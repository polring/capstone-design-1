# GGUF Exporter

학습·배포 체크포인트를 C 추론 엔진이 읽는 GGUF 파일로 내보낸다. C 엔진과 파일 형식·계산 규칙(계약, `CONTRACT.md`)은
별도 저장소 [capstone-sql-cli](https://github.com/polring/capstone-sql-cli)에 있다.

`src/inference/gguf_model.py`는 팀의 `TokenEmbedding`, `TransformerDecoderBlock`,
`RMSNorm`을 사용하여 전체 모델을 감싼 내보내기용 모델(`InferenceModel`)이다.
학습 모델(`TextToSQLModel`)과 블록 구성이 같고 최상위 파라미터 이름만 다르므로,
학습·배포 체크포인트는 이름을 바꾸고 사용하지 않는 어휘 행을 제외하여 불러온다(아래 "학습·배포 체크포인트 내보내기").
파라미터 수만 같은 모델은 호환되지 않는다.

## C 입력 전처리

`src/inference/c/normalizer.c`·`normalizer.h`는 Python의
`src/inference/wordnet_normalizer.py`에 대응하는 C 구현이다. 기존
`models/stage3/normalizer.json`을 그대로 읽으며 Python·NLTK 없이 실행한다.
GGUF 내보내기에 자동으로 들어가는 기능은 아니므로 C 구동기의 **BPE 호출 전**에 연결한다.
아래 C 코드는 별도 `capstone-sql-cli` 저장소로 전달하는 로컬 이식 작업물이다.
`src/inference/c/`는 현재 이 저장소에서 미추적 상태이므로 새로 clone하면 포함되지 않는다.
아래 빌드 예시는 그 파일들을 로컬에 준비한 경우에 사용한다. 별도 CLI 저장소에 연결·반영되었는지는 이 문서의 로컬 점검으로 확인하지 않았다.

C99 컴파일러와 **PCRE2 8-bit 개발 라이브러리**가 필요하다. Ubuntu에서는
`libpcre2-dev`·`pkg-config`, Windows에서는 MinGW/MSVC용 PCRE2를 설치한다.
PCRE2를 사용하는 이유는 사전에 들어 있는 lookbehind·이름 있는 캡처 등 정규식 규칙을 실행하기 위해서다.
공유 라이브러리로 링크했다면 실행 환경에도 PCRE2 DLL/공유 라이브러리를 제공해야 한다.

```sh
cmake -S src/inference/c -B build/normalizer "-DNORMALIZER_MAP_PATH=models/stage3/normalizer.json"
cmake --build build/normalizer --config Release
# 실행 파일 위치는 CMake 생성기에 따라 build/normalizer/Release/일 수도 있다.
build/normalizer/normalizer_cli "record count of transactions"
# row count of orders
```

직접 컴파일하는 Linux 예시:

```sh
# 기본 사전 경로는 실행할 때의 작업 폴더에 있는 normalizer.json이다.
cc -std=c99 -O2 src/inference/c/normalizer.c src/inference/c/normalizer_cli.c \
  $(pkg-config --cflags --libs libpcre2-8) -o normalizer_cli
```

일반적인 순차 CLI에서는 `normalizer_process(question)` 한 번만 호출하면 된다.

```c
#include "normalizer.h"

const char *input = normalizer_process(question);
/* 기존 BPE에 input을 전달한다. 별도의 초기화·종료·free 호출은 없다. */
```

첫 호출에서 현재 작업 폴더의 `normalizer.json`을 읽어 재사용하며, 프로세스 종료 시 사전과 결과 메모리를
자동 해제한다. 사전 경로는 빌드 시 `NORMALIZER_MAP_PATH`로 바꿀 수 있다. 실패 시 로그를 남기고
원문을 사용한다. 첫 사전 로딩에 실패하면 해당 실행에서는 전처리 없이 계속 진행한다.
반환 문자열은 수정하거나 해제하지 않고 **다음 호출 전에** 토크나이저에서 사용해야 한다.
이 자동 관리 함수는 동시 호출을 지원하지 않는다. 입력은 NULL이 아닌 문자열이어야 한다.

외부에는 `normalizer_process()`만 공개한다. 사전 로딩·치환·메모리 정리는
`normalizer.c`의 내부 함수로 분리되어 있다. 입출력은 NUL로 끝나는 UTF-8 문자열이다.
전처리는 단어·구문·비교 기호를 치환하고,
숫자·날짜·따옴표·DB 값·명시적 LIKE 조각을 보호한다. 원문에서 같은 위치에 겹치는 치환은 긴 구문이 우선하며,
치환된 결과를 재치환하지 않는다. 사전 형식은 버전 1이며, 구문 치환문은 현재 배포 사전에서 사용하는
일반 문자열과 `\g<name>` 캡처 참조를 지원한다. 다른 Python 전용 정규식 문법이나 치환 이스케이프는 별도 대응이 필요하다.
사전은 최대 16 MiB까지 읽는다. WordNet 라이선스 원본은 `licenses/WORDNET_LICENSE.txt`에서 관리하고,
배포할 때는 `normalizer.json`과 `WORDNET_LICENSE.txt`를 함께 제공한다. `models/stage3/`에는 배포용 사본을 둔다.

전처리 도중 오류가 나면 부분적으로 바뀐 결과를 버리고 **원문을 반환**한다.
오류는 현재 작업 폴더의 `normalizer.log`에 추가 기록하며,
빌드 시 `NORMALIZER_LOG_PATH`로 경로를 바꿀 수 있다. 로그 파일을 열 수 없으면 `stderr`에 기록한다.
원문은 다른 작업 메모리보다 먼저 복사해 이후 메모리 부족에도 반환할 수 있게 했다.
최초 원문 복사조차 불가능하면 호출자가 전달한 원문 포인터를 그대로 반환한다.

C 코드의 빌드·테스트 의존성은 별도 C 구동기 저장소에서 관리한다.
이 저장소의 Python 테스트와 CI에는 PCRE2 설치나 C 테스트를 추가하지 않는다.

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
| `model_cfg.valid_vocab_size` (없거나 `None`이면 `vocab_size`) | `config.vocab_size` |
| `tok_emb.*` | `embedding.*` |
| `norm.*` | `final_norm.*` |
| 그 밖의 `blocks.*`, `lm_head.weight` | 같은 이름 |

`eps`(1e-6)와 `theta`(10000)는 학습 모델의 기본값과 같아 체크포인트에 없어도 된다.
`padding_idx`는 추론 결과에 영향이 없어 버린다.
가중치가 공유된(weight tying) 임베딩·출력층은 독립 embedding/lm_head 텐서 두 개로 내보낸다.
`valid_vocab_size`가 할당된 어휘 수보다 작으면 두 텐서에서 사용하지 않는 뒤쪽 행을 제외한다.
따라서 GGUF에서도 학습 모델과 같은 토큰 ID만 출력하며, 기존 C 엔진의 파일 형식을 그대로 사용한다.

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
