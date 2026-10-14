# GGUF Exporter

학습·배포 체크포인트를 C 추론 엔진이 읽는 GGUF 파일로 내보낸다. C 엔진과 파일 형식·계산 규칙(계약, `CONTRACT.md`)은
별도 저장소 [capstone-sql-cli](https://github.com/polring/capstone-sql-cli)에 있다.

`src/inference/gguf_model.py`는 팀의 `TokenEmbedding`, `TransformerDecoderBlock`,
`RMSNorm`을 사용하여 전체 모델을 감싼 내보내기용 모델(`InferenceModel`)이다.
학습 모델(`TextToSQLModel`)과 블록 구성이 같고 최상위 파라미터 이름만 다르므로,
학습·배포 체크포인트는 이름만 바꿔 그대로 불러온다(아래 "학습·배포 체크포인트 내보내기").
파라미터 수만 같은 모델은 호환되지 않는다.

## 실행 (저장소 루트, Windows PowerShell)

```powershell
python -m src.exporter --checkpoint models/stage1/model.pt --output runs/stage1/model.gguf
python -m src.exporter --dummy --output dummy.gguf
python -m pytest tests/inference/test_gguf_model.py tests/test_contract_fixtures.py -q
```

`--dummy`는 dim=16, heads=2, layers=2, ffn_dim=32의 무작위 모델이다.
더미 기본 BPE는 병합 없이 바이트와 seed만 사용한다.
학습된 BPE를 사용하려면 `--tokenizer tokenizer.json`(seed + 병합 규칙)을 지정한다.
더미 출력은 SQL이거나 유효한 UTF-8 문자열이라는 보장이 없다.

## 학습·배포 체크포인트 내보내기

`python -m src.release`가 만든 배포 모델이나 학습 체크포인트(`runs/.../best.pt`)는 바로 내보낸다.
`--tokenizer`를 생략하면 체크포인트와 같은 폴더의 `tokenizer.json`(없으면 체크포인트에 기록된 경로)을
쓴다. 학습할 때 쓴 토크나이저와 다르면 토큰 ID가 어긋나므로 다른 파일을 직접 지정하지 않는다.

```powershell
python -m src.exporter --checkpoint models/stage1/model.pt --output runs/stage1/model.gguf
```

`load_checkpoint`는 `model_cfg` + `model` 형식(학습·배포 체크포인트)을 알아보고 다음과 같이 바꾼다.

| 학습·배포 체크포인트 (`TextToSQLModel`) | 내보내기 모델 (`InferenceModel`) |
|---|---|
| `model_cfg.num_layers`, `num_heads`, `max_seq_len` | `config.layers`, `heads`, `context` |
| `tok_emb.*` | `embedding.*` |
| `norm.*` | `final_norm.*` |
| 그 밖의 `blocks.*`, `lm_head.weight` | 같은 이름 |

`eps`(1e-6)와 `theta`(10000)는 학습 모델의 기본값과 같아 체크포인트에 없어도 된다.
`padding_idx`는 추론 결과에 영향이 없어 버린다.
가중치가 공유된(weight tying) 임베딩·출력층은 독립 embedding/lm_head 텐서 두 개로 내보낸다.

## 내보내기 전용 체크포인트 계약

`InferenceModel`을 직접 학습하거나 만든 경우 `config` + `state_dict` 형식으로 저장한다.
이 형식에는 토크나이저 위치가 없으므로 `--tokenizer`가 필요하다.

```python
from dataclasses import asdict
import torch
from src.inference.gguf_model import ModelConfig, InferenceModel

config = ModelConfig()
model = InferenceModel(config)
torch.save({"config": asdict(config), "state_dict": model.state_dict()}, "model.pt")
```

```powershell
python -m src.exporter --checkpoint model.pt --tokenizer tokenizer.json --output model.gguf
```

두 형식 모두 키와 shape를 엄격히 검사한다. 다른 이름을 쓰는 체크포인트는 추측하지 않고 오류를 낸다.

## 토크나이저 설정 내보내기

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

## 계약 버전과 fixture

계약 버전은 `src/tokenizer/contract.py`의 `CONTRACT_VERSION` 하나로 정하고 exporter가 파일에 쓴다.
C 엔진은 `tests/fixtures/contract/v<N>/`의 사본으로만 검증되므로, 계산 규칙(CONTRACT.md 3~6절)을 바꾸면 다음 순서를 따른다.

1. `CONTRACT_VERSION`을 올리고 `python -m src.export_fixtures --new-model`로 `tests/fixtures/contract/v<새 버전>/`을 만든다.
2. C 엔진 저장소에 그 폴더를 `fixtures/v<새 버전>/`으로 복사하고, 엔진의 `CONTRACT_VERSION`·계산·`CONTRACT.md`를 고쳐
   `make test`를 통과시킨다.
3. 엔진이 먼저 병합된 뒤에 새 버전 태그로 release를 올린다(release workflow가 엔진 main으로 생성 확인을 한다).

`tests/test_contract_fixtures.py`는 커밋된 fixture 모델로 기대값을 다시 계산해 비교한다. 계산이 바뀌었는데 1번을
빠뜨리면 이 테스트가 실패한다. 값만 바뀌는 변경(모델 크기, vocab, seed·merge 내용)은 버전을 올리지 않는다.

## 배포 (GitHub Release)

`stage<N>-contract<V>` 태그(시험용은 `-rc1` 같은 접미사, prerelease로 올라감)를 push하면
`.github/workflows/release.yml`이 `models/stage<N>/model.pt`를 내보내 C 엔진 저장소의 main으로 SQL 생성까지 확인한 뒤
`model.gguf`, `model_info.json`, `shop.db`, `SHA256SUMS`를 release에 올린다. 태그의 계약 버전이
`CONTRACT_VERSION`과 다르면 실패한다. `.gguf`는 git에 커밋하지 않는다(fixture 모델만 예외).

## 검증

`tests/inference/test_gguf_model.py`는 내보낸 GGUF를 `exporter.read_gguf`로 다시 읽어 가중치·병합 규칙·logits가 원래 모델과
같은지 확인한다. 배포 모델(`models/stage1/model.pt`)은 계약 규칙의 greedy 생성(`export_fixtures.reference_generate`)이
학습 모델과 같은 SQL을 내는지 확인한다.
선택적 `gguf.GGUFReader` 검사로 모든 텐서 값의 파일 호환성을 독립 확인한다.
C 엔진의 Python 대조(fixture)·오류 경로 테스트와 검증 기록은 C 엔진 저장소에 있다.
