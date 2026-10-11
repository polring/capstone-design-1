# GGUF Exporter / C inference engine

## 구현 범위

기존 #9의 EXL1 확인용 코드를 GGUF v3와 실제 C forward로 교체했다.
무작위 가중치 기준으로 Python/C logits와 반복 생성 결과를 비교한다.
학습된 체크포인트가 아직 없어 SQL 정확도는 검증하지 않았다.

`src/inference.py`는 팀의 `TokenEmbedding`, `TransformerDecoderBlock`,
`RMSNorm`을 사용하여 전체 모델을 감싼 검증용 모델이다.
전체 학습 모델이 완성되면 이 문서의 체크포인트 계약으로 연결하거나
Exporter의 이름 대응을 수정해야 한다. 파라미터 수만 같은 모델은 호환되지 않는다.

## 실행 (저장소 루트, Windows PowerShell)

PyTorch, NumPy, pytest 및 GCC가 필요하다. 독립 GGUF 검증에는
`pip install -r requirements-inference.txt`도 실행한다.

```powershell
python -m src.exporter --dummy --output dummy.gguf
gcc -std=c11 -O2 -Wall -Wextra -Werror src/loader.c -lm -o loader.exe
.\loader.exe dummy.gguf --encode "SELECT stock FROM items;"
.\loader.exe dummy.gguf --logits 257 104 105 259
.\loader.exe dummy.gguf --generate "how many mouse units are left in stock?" 20
python -m pytest tests/test_inference.py -q
```

Linux에서는 `-o loader`로 빌드하고 `./loader`로 실행한다.
`--dummy`는 dim=16, heads=2, layers=2, ffn_dim=32의 무작위 모델이다.
더미 기본 BPE는 병합 없이 바이트와 seed만 사용한다.
학습된 BPE를 사용하려면 `--merges bpe_merges.json`을 지정한다.
더미 출력은 SQL이거나 유효한 UTF-8 문자열이라는 보장이 없다.

## 실제 체크포인트 계약

```python
from dataclasses import asdict
import torch
from src.inference import ModelConfig, InferenceModel

config = ModelConfig()  # 최종 학습 설정을 정확히 전달
model = InferenceModel(config)
# 학습 루프에서 이 모델을 학습한 뒤 저장
torch.save({"config": asdict(config), "state_dict": model.state_dict()}, "model.pt")
```

```powershell
python -m src.exporter --checkpoint model.pt --merges bpe_merges.json --output model.gguf
.\loader.exe model.gguf --generate "how many mouse units are left in stock?" 64
```

최종 학습 모델이 다른 이름이나 `model_state_dict` 키를 쓰면 자동으로
추측하지 않고 어댑터를 추가한다. 현재 코드는 키와 shape를 엄격히 검사한다.
가중치가 공유된 모델은 현재 독립 embedding/lm_head 텐서 두 개로 내보낸다.

## 파일 형식과 이름 대응

[GGUF 공식 명세](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)를 따른
little-endian v3 컨테이너, F32(`GGML_TYPE_F32=0`), alignment=32를 사용한다.
Python writer에는 GGUF 라이브러리가 필요하지 않고 C에는 외부 라이브러리가 없다.

| 설정 | 메타데이터 |
| --- | --- |
| 구조 | `general.architecture = capstone_sql` |
| 계약 버전 | `capstone_sql.contract_version = 2` |
| 단어장/차원/층/헤드/FFN/문맥 | `capstone_sql.vocab_size/dim/layers/heads/ffn_dim/context` |
| RMSNorm/RoPE | `capstone_sql.eps/theta` |
| seed 토큰 | `capstone_sql.tokenizer.seeds`: string array |
| 순서 있는 BPE 병합 | `capstone_sql.tokenizer.merges`: uint32 array `[a,b,new_id,...]` |
| 특수 토큰 ID | `capstone_sql.tokenizer.pad_id/bos_id/eos_id/sep_id`: uint32 |
| seed/병합 시작 ID | `capstone_sql.tokenizer.seed_base/merge_base`: uint32 |

계약 버전 2는 이 19개 메타데이터 항목을 사용한다. seed 개수는 배열 길이에서
읽고 특수 토큰 및 seed/병합 시작 ID는 파일에서 읽는다. 기존 계약 버전 1은
명시적으로 거절하므로 원래 체크포인트와 병합 파일을 새 exporter로 다시 내보낸다.
임의의 GGUF를 받는
범용 엔진이 아니며 다른 구조, F16/양자화, 다른 메타데이터 계약은 거절한다.
`capstone_sql`은 사용자 정의 구조이므로 llama.cpp에서 자동 실행되지 않는다.

### 토크나이저 설정 내보내기

기본값은 현재 `src/tokenizer/bpe.py`의 seed 목록과 특수 토큰 순서에서 가져온다.
다른 학습 토크나이저를 연결할 때는 `--tokenizer-config tokenizer-config.json`을
추가한다. 파일은 아래 형식이며 **가중치를 학습할 때의 실제 토큰 대응**과 일치해야 한다.
ID를 임의로 바꾸는 것만으로 기존 가중치가 새 토크나이저에 맞춰지는 것은 아니다.

```json
{
  "seeds": ["SELECT", "WHERE", "AND", "OR"],
  "pad_id": 256,
  "bos_id": 257,
  "eos_id": 258,
  "sep_id": 259,
  "seed_base": 260,
  "merge_base": 264
}
```

위는 형식 예시일 뿐 1단계 기존 모델의 설정이 아니다. 실제 병합 파일의 새 ID는
`merge_base`부터 순서대로 증가해야 한다. 바이트 ID 0~255는 byte-level BPE 정의상
고정이며, 특수 ID 중복·바이트/seed/병합 범위 충돌·vocab 초과는 양쪽에서 거절한다.
seed가 38개일 필요는 없다. 현재 영어 ASCII 전처리와 질문 소문자화·BOS/SEP 프롬프트
방식은 그대로이므로 SentencePiece 또는 공개 Llama 모델 지원이 추가된 것은 아니다.

텐서 이름은 Python state_dict 그대로다. 텐서 데이터는 PyTorch의
`[out,in]` row-major이며 GGUF 차원 설명만 fastest-first 순서로 뒤집는다.

| 텐서 | Python shape |
| --- | --- |
| `embedding.embedding.weight` | `[vocab_size, dim]` |
| `blocks.N.attn_norm.weight`, `blocks.N.ffn_norm.weight` | `[dim]` |
| `blocks.N.attn.qkv_proj.weight` | `[3*dim, dim]`, Q→K→V 순서 |
| `blocks.N.attn.out_proj.weight` | `[dim, dim]` |
| `blocks.N.ffn.w1.weight`, `blocks.N.ffn.w3.weight` | `[ffn_dim, dim]` |
| `blocks.N.ffn.w2.weight` | `[dim, ffn_dim]` |
| `final_norm.weight` | `[dim]` |
| `lm_head.weight` | `[vocab_size, dim]` |

## C 계산 흐름

### CLI 함수 구조 (2026-10-06 가독성 리팩터링)

기존의 한 파일 빌드 명령과 GGUF 계약을 유지하면서 `src/loader.c` 내부를
역할별 함수로 분리했다. 기능이나 SQL 지원 범위를 추가한 작업은 아니다.

| 역할 | 주요 함수 |
| --- | --- |
| 프로그램 시작 및 정리 | `main`, `release_model`, `release_tokenizer` |
| CLI 옵션 해석 및 분기 | `parse_cli_options`, `dispatch_cli_command` |
| 모드별 실행 | `run_logits_command`, `run_encode_command`, `run_generation_command` |
| 모델 파일 읽기 | `read_model_file`, `read_gguf_header`, `read_model_metadata` |
| 텐서 로드 및 검증 | `read_tensor_descriptors`, `read_tensor_values`, `validate_model_weights` |
| 토큰 바이트 복원 | `build_token_pieces`, `build_merged_token_piece` |
| 질문 준비 및 토큰화 | `normalize_question`, `prepare_prompt_tokens`, `encode_text`, `encode_piece` |
| 추론 단계 실행 | `forward`, `run_decoder_block`, `compute_last_token_logits` |
| Attention 및 FFN | `compute_queries_keys_values`, `compute_causal_attention`, `apply_attention_residual`, `apply_ffn_residual` |
| 다음 토큰 생성 | `generate_tokens`, `select_next_token` |
| 추론 임시 메모리 | `create_workspace`, `release_workspace` |

`ModelConfig`는 구조 설정, `Tokenizer`는 BPE와 토큰 바이트,
`InferenceWorkspace`는 계산용 임시 버퍼를 각각 묶는다.
텐서 이름과 파일의 메타데이터 이름은 변경하지 않았다.

처음 코드를 읽을 때는 `main` → `dispatch_cli_command` → 모드별 실행 함수를
따라가고, 생성 모드에서는 `prepare_prompt_tokens` → `generate_tokens` →
`forward` 순서로 확인한다. 파일 로드는 `load_model`에서 단계별 함수를 따라간다.

복잡한 조건은 `is_model_config_compatible`, `is_tensor_data_in_bounds`,
`has_valid_merge_references` 등으로 분리했다. 단락 평가로 잘못된 배열 접근과
0으로 나누기를 막는 순서는 유지한다. 주석은 GGUF 바이트 순서와 정렬,
Q/K RoPE, causal mask, softmax, BPE 병합 순서, EOS 동점 정책을 설명한다.

### 수식 및 생성 흐름

```text
GGUF 설정/전체 가중치 로드
→ 소문자 질문을 팀 BPE로 encode
→ [BOS=257] 질문 [SEP=259]
→ Embedding
→ 각 블록: RMSNorm → packed QKV → 인접 차원 RoPE
  → causal scaled attention → 출력 projection → residual
  → RMSNorm → SwiGLU → residual
→ final RMSNorm → lm_head → 마지막 위치 logits
→ greedy 선택 → EOS=258 또는 최대 길이까지 반복 → 토큰 바이트 출력
```

파일에서 읽은 PAD/BOS/SEP ID와 병합 미정의 ID는 생성 후보에서 제외하고
EOS와 실제 복원 가능한 토큰만 고른다. 동점은 EOS 우선, 그 외 낮은 ID 우선이다.
입력은 1단계 영어 범위에 맞춰 ASCII만 지원하며 비ASCII는 명시적으로 거절한다.
숫자는 한 글자씩, seed는 단어 경계를 확인하고, 알파벳 조각에는 순서대로
BPE 병합을 적용한다. 이는 팀 regex tokenizer의 ASCII 동작과 비교 검증한다.

Baseline은 배치 1이고 매 생성 단계 전체 prefix를 다시 계산한다.
KV cache, 샘플링, 양자화, Unicode 입력, SQL 문법 제약은 후속 작업이다.
파일 및 개별 할당은 1 GiB 제한, layers<=256, context<=4096이며
각 할당 합계에 대한 전역 메모리 예산 관리 기능은 아직 없다.

## 검증과 다음 연동

`tests/test_inference.py`는 GCC 경고를 오류로 처리하여 빌드한다.
동일한 더미 가중치로 전체 Python/C logits를 `atol=rtol=2e-5` 기준 비교하고,
다른 설정, 토큰 생성 반복, EOS/문맥 종료, BPE 및 잘못된 입력/파일도 검증한다.
선택적 `gguf.GGUFReader` 검사로 모든 텐서 값의 파일 호환성을 독립 확인한다.

최종 학습 체크포인트가 제공되면 이름/설정/토크나이저 계약을 확인한 뒤
실제 질문의 Python/C logits와 SQL 결과를 추가 검증한다. 이는 현재 더미 검증과 구분한다.

2026-10-03 로컬 검증 결과: 새 추론 테스트 25개 포함 전체 99개 통과.
CPU PyTorch 2.14.1, GCC, GGUFReader 0.19.0으로 실행했다.

2026-10-06 리팩터링 검증: C11/GCC `-O2 -Wall -Wextra -Werror` 빌드와
포맷 검사를 통과했다. 전체 160개 테스트 중 Python 및 파일 검증 107개는
통과했지만, C 실행이 필요한 53개는 Windows 애플리케이션 제어 정책
(`WinError 4551`)이 새 실행 파일의 시작을 차단하여 실행 결과를 검증하지 못했다.
이 53개에는 리팩터링 회귀 테스트 13개가 포함되어 있다.
보안 설정을 변경하지 않았으며, 전체 통과나 변경 전후 CLI 출력 일치를
확인한 것으로 취급하면 안 된다. 승인된 실행 환경에서 다음 명령으로 재검증한다.

```powershell
python -m pytest tests/ -q
```

2026-10-07 계약 v2 검증: 전체 **191개 테스트 통과**. 이번 실행에서는 C 실행 파일이
정상 시작되어 이전에 차단됐던 회귀 테스트도 수행했다. seed 개수 0/2/45, 변경된
특수 토큰 ID와 seed/merge 시작 ID, EOS 종료, 잘못된 ID 범위·충돌·병합 참조,
계약 v1 거절을 검증했다. C11/GCC `-O2 -Wall -Wextra -Werror -pedantic` 빌드와
Black/clang-format 검사도 통과했다.

전체 실행 중 기존 `stock_` 토큰화 불일치를 발견해 수정했다. Python은 regex로
분할한 조각도 seed 목록에서 다시 조회하므로 C도 동일하게 처리한다. `stock_`,
`_stock`, `1stock` 회귀 테스트를 포함했다. 이 결과는 더미 가중치 및 팀 BPE 계약
검증이며 공개 사전학습 모델을 실행한 결과는 아니다.
