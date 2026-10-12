# capstone_sql GGUF 계약 (버전 2)

학습 저장소의 exporter(`src/exporter.py`)가 쓰는 GGUF 파일과, 이 C 엔진이 그 파일로 하는 계산의 약속이다.
구조 하이퍼파라미터와 토크나이저 데이터는 파일 안에 있으므로 엔진은 설정 파일을 따로 두지 않는다. 파일에 담을 수
없는 **계산 규칙**은 이 문서와 `fixtures/v<N>/`의 기대값으로 고정한다.

## 1. 버전 규칙

- 버전은 exporter의 `CONTRACT_VERSION`(`src/tokenizer/contract.py`)과 엔진의 `CONTRACT_VERSION`(`engine/loader.c`)이
  같은 값이어야 한다. 엔진은 지원하지 않는 버전의 파일을 오류로 거부한다.
- **버전을 올리는 경우**: 아래 2~6절 중 하나라도 바뀔 때. 예: RoPE 구현 변경, 정규화 위치 변경, 특수 토큰 추가,
  pre-tokenizer 규칙 변경, 위치 토큰 도입, 메타데이터 키나 텐서 이름 변경.
- **버전을 올리지 않는 경우**: 값만 바뀔 때. 모델 크기, vocab 크기, seed·merge 내용, eps·theta 값은 메타데이터로
  전달되므로 같은 버전이다(예: 1단계 → 2단계 모델 전환).
- 버전을 올리면 학습 저장소에서 `python -m src.export_fixtures --new-model`로 `fixtures/v<새 버전>/`을 만든다.
  이전 버전 fixture는 엔진이 그 버전을 지원하는 동안 지우지 않는다.

| 버전 | 내용 |
| --- | --- |
| 1 | 이전 계약. 지금은 거부하므로 원래 체크포인트에서 다시 내보낸다 |
| 2 | 특수 토큰 ID, seed·merge 시작 ID를 메타데이터로 전달 (현재) |

## 2. 파일 형식

[GGUF 명세](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)의 little-endian v3 컨테이너, 텐서는 전부
F32(`GGML_TYPE_F32=0`), alignment 32. 다른 구조, F16·양자화, 다른 메타데이터 계약은 거부한다. `capstone_sql`은
사용자 정의 구조라 llama.cpp에서 실행되지 않는다.

**메타데이터 19개** (전부 필수)

| 키 | 타입 | 내용 |
| --- | --- | --- |
| `general.architecture` | string | `capstone_sql` |
| `general.alignment` | uint32 | 32 |
| `capstone_sql.contract_version` | uint32 | 2 |
| `capstone_sql.vocab_size`, `dim`, `layers`, `heads`, `ffn_dim`, `context` | uint32 | 구조 |
| `capstone_sql.eps`, `theta` | float32 | RMSNorm epsilon, RoPE theta |
| `capstone_sql.tokenizer.seeds` | string 배열 | seed 토큰 (ID = `seed_base` + 순서) |
| `capstone_sql.tokenizer.merges` | uint32 배열 | `[a, b, new_id, ...]`, 학습 순서 |
| `capstone_sql.tokenizer.pad_id`, `bos_id`, `eos_id`, `sep_id` | uint32 | 특수 토큰 ID |
| `capstone_sql.tokenizer.seed_base`, `merge_base` | uint32 | seed·merge 시작 ID |

제약: `vocab_size >= 256`, `dim % heads == 0`, head 차원(`dim / heads`)은 짝수, `layers <= 256`, `context <= 4096`.
바이트 ID 0~255는 고정이고, 특수 ID는 서로 달라야 하며 바이트·seed·merge 범위와 겹치면 안 된다. merge의 `new_id`는
`merge_base`부터 1씩 증가하고, `a`·`b`는 그보다 앞서 정의된 ID여야 한다.

**텐서** (이름은 PyTorch `state_dict` 그대로. 데이터는 `[out, in]` row-major이고 GGUF 차원 표기만 fastest-first로 뒤집는다)

| 텐서 | shape |
| --- | --- |
| `embedding.embedding.weight` | `[vocab_size, dim]` |
| `blocks.N.attn_norm.weight`, `blocks.N.ffn_norm.weight` | `[dim]` |
| `blocks.N.attn.qkv_proj.weight` | `[3*dim, dim]`, 행 순서 Q → K → V |
| `blocks.N.attn.out_proj.weight` | `[dim, dim]` |
| `blocks.N.ffn.w1.weight` (gate), `blocks.N.ffn.w3.weight` (up) | `[ffn_dim, dim]` |
| `blocks.N.ffn.w2.weight` (down) | `[dim, ffn_dim]` |
| `final_norm.weight` | `[dim]` |
| `lm_head.weight` | `[vocab_size, dim]` |

학습 모델이 임베딩과 출력층 가중치를 공유해도 파일에는 두 텐서로 따로 들어간다. 엔진은 공유 여부를 가정하지 않는다.

## 3. 모델 계산

선형층에 bias는 없다. `x`는 위치별 `dim` 벡터, `W x`는 `[out, in]` 행렬 곱이다.

```text
h = embedding[token_id]                                     # 위치 0..T-1
각 블록 N = 0..layers-1 (Pre-LN):
    a = RMSNorm(h, attn_norm)
    q, k, v = split(qkv_proj · a, 3)                        # Q, K, V 순서로 dim 씩
    q, k = RoPE(q, position), RoPE(k, position)             # V 는 회전하지 않음
    head 별: softmax(q·kᵀ / sqrt(head_dim), causal) · v      # 위치 i 는 0..i 만 참조
    h = h + out_proj · concat(heads)
    f = RMSNorm(h, ffn_norm)
    h = h + w2 · (silu(w1 · f) * (w3 · f))                  # SwiGLU, silu(x) = x · sigmoid(x)
logits = lm_head · RMSNorm(h[T-1], final_norm)              # 마지막 위치만
```

- **RMSNorm**: `x / sqrt(mean(x²) + eps) * weight`. eps는 `capstone_sql.eps`.
- **head 분할**: `dim`을 head 순서대로 `head_dim`씩 자른다(head 0이 앞쪽).
- **RoPE**: head 안에서 **인접한 두 차원** `(2i, 2i+1)`을 한 쌍으로 회전한다(앞뒤 절반을 짝짓는 방식이 아니다).
  각도는 `position / theta^(2i / head_dim)`, `position`은 0부터 센다.
  `(x0, x1) → (x0·cos − x1·sin, x0·sin + x1·cos)`.

## 4. 토크나이저 (byte-level BPE)

ID 배치: `0~255` 바이트, 특수 토큰(`pad_id`…`sep_id`), seed(`seed_base`부터), merge(`merge_base`부터).
입력은 ASCII만 받으며 비ASCII는 오류다. 텍스트를 앞에서부터 다음 순서로 조각낸다.

1. **seed 우선**: 현재 위치에서 시작하는 seed 중 가장 긴 것이 단어 경계(앞뒤가 `[A-Za-z0-9_]`가 아님)에 맞으면
   그 seed 하나가 토큰이다. `stock`은 `stockholm` 안에서 매칭되지 않는다.
2. 아니면 조각을 자른다: **알파벳 연속**은 한 조각, 숫자·공백·그 밖의 문자는 **한 글자씩** 한 조각.
3. 잘린 조각 전체가 seed와 같으면 그 seed다(예: `stock_`의 `stock`은 1번에서 `_` 때문에 경계가 안 맞지만 여기서
   seed가 된다).
4. 그 외 조각은 바이트 ID로 바꾼 뒤 **merge를 학습 순서대로 하나씩** 조각 전체에 적용한다. 한 merge는 왼쪽부터
   겹치지 않게 적용한다. merge는 조각 경계를 넘지 않는다.

토큰 ID → 바이트 복원: 바이트는 그 자체, seed는 문자열, merge는 `a`와 `b`의 바이트를 이어 붙인 것이다. 특수 토큰과
쓰이지 않는 ID에는 바이트가 없다.

## 5. 입력 형식

```text
[bos_id] encode(lowercase(question)) [sep_id]
```

질문은 ASCII 소문자로 바꾼 뒤 4절로 인코딩한다. 이 길이가 `context`를 넘으면 오류다.

## 6. 디코딩

- greedy. 매 단계 전체 prefix로 마지막 위치 logits를 계산한다(KV cache 없음).
- **후보**: `eos_id`와 바이트 복원이 가능한 ID(바이트, seed, merge). PAD·BOS·SEP·미사용 ID는 나오지 않는다.
- **동점**: `eos_id` 우선, 그다음 작은 ID.
- **종료**: `eos_id`가 선택되면 `eos`, 생성 수가 `max_new_tokens`에 닿으면 `max_new_tokens`, 다음 forward 전에
  시퀀스가 `context`를 채우면 `context`.
- 출력은 생성된 토큰의 바이트를 이어 붙인 것이다(`eos_id`는 포함하지 않음).

## 7. fixture

`fixtures/v<N>/`은 학습 저장소의 `python -m src.export_fixtures`가 만든다.

| 파일 | 내용 |
| --- | --- |
| `model.gguf` | 작은 무작위 모델 (vocab 320, dim 16, 2층, context 32). 이 버전의 계약으로 쓴 파일 |
| `cases.txt` | Python이 계산한 기대값: 토큰화, 마지막 위치 logits, greedy 생성 결과와 종료 이유 |
| `manifest.json` | 계약 버전과 두 파일의 sha256 |

`cases.txt`는 탭으로 구분하고, 문자열은 UTF-8 바이트를 hex로 적는다(빈 문자열은 빈 칸).

```text
encode    <text>    <token id ...>
logits    <token id ...>    <logit ...>                       # vocab_size 개
generate  <max_new_tokens>    <question>    <output>    <eos | max_new_tokens | context>
```

엔진은 `make test`에서 이 값과 비교한다. logits 허용 오차는 `atol = rtol = 2e-5`이고, 토큰 ID와 생성 바이트는
정확히 같아야 한다. 학습 저장소는 같은 model.gguf로 기대값을 다시 계산해 커밋된 값과 비교하므로, 계산이 바뀌었는데
버전을 올리지 않으면 그쪽 CI가 실패한다.
