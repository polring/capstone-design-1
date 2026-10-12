# CLI 저장소 분리 계획

작성: 2026-10-12 · 1단계 구현: 2026-10-12 (`refactor/cli-split` 브랜치)

C 추론 엔진과 C 후처리기를 별도 저장소(이하 CLI 저장소)로 분리하는 계획이다. 먼저 같은 저장소 안에서 의존성을
끊고(1단계), 분리 직전 검증을 거쳐 히스토리를 보존해 떼어낸다(2단계). 안정화 기간은 두지 않는다.

## 1. 범위

| 대상 | 현재 위치 | 분리 후 |
| --- | --- | --- |
| C 추론 엔진 | `src/loader.c` | CLI 저장소 |
| C 후처리기 | `src/postprocessor/c/` | CLI 저장소 |
| Python 후처리기 (semantic dict 생성 포함) | `src/postprocessor/python/` | 메인 저장소에 남김 |
| 학습·평가·Python 추론, exporter | `src/` | 메인 저장소에 남김 |

## 2. 원칙

- **두 저장소가 공유하는 것은 계약이고, 코드나 설정 파일이 아니다.** CLI 저장소에 `config.py` 사본을 두지 않는다.
  구조 하이퍼파라미터는 exporter가 GGUF 메타데이터에 쓰고 `loader.c`가 파일에서 읽으므로, 2단계 모델(8층·384차원)도
  CLI 코드 수정 없이 돈다. 사본을 두면 원본이 둘이 되어 언젠가 어긋난다.
- **CLI에 넘기는 모델 파일은 `model.gguf` 하나다.** gguf에 가중치, 구조 하이퍼파라미터(`vocab_size`, `dim`, `layers`,
  `heads`, `ffn_dim`, `context`, `eps`, `theta`), 토크나이저(seed, merge, 특수 토큰 id)가 모두 들어 있다. `model.pt`,
  `tokenizer.json`, `model_info.json`은 메인 저장소의 원본으로만 둔다. 후처리기는 여기에 `shop.db`와
  `semantic_dict_*.bin`이 더 필요하다.
- **gguf는 파생물이라 git에 넣지 않는다.** 메인 저장소가 GitHub Release에 올리고 CLI 저장소가 태그로 받아 간다.
- **gguf에 담을 수 없는 것은 문서, 버전, 테스트로 묶는다.** 계산 규칙(RoPE 차원 쌍, Pre-LN 순서, SwiGLU gate/up 순서,
  임베딩-출력층 가중치 공유), BPE 알고리즘(병합 순서, 동점 처리, 바이트 fallback), 입력 형식(`<bos> 질문 <sep>`, 소문자
  변환), 디코딩 규칙(greedy, `<eos>` 종료, 최대 길이)은 어긋나도 파일은 정상적으로 로드되고 결과만 조용히 틀린다.
  그래서 계약 문서, 계약 버전, fixture로 고정한다.
- **버전을 맞추는 일은 사람의 기억에 맡기지 않는다.** 계산 결과가 바뀌었는데 버전을 올리지 않으면 메인 CI가 실패하고,
  지원하지 않는 버전의 파일은 `loader.c`가 거부한다.

## 3. 현재 결합 상태

| 항목 | 상태 | 처리 |
| --- | --- | --- |
| C 코드의 Python import | 없음. C만으로 빌드된다 | 그대로 |
| 계약 버전 | `src/exporter.py:51`과 `src/loader.c:508`에 숫자 `2`가 따로 하드코딩돼 있다 | 1단계 3번 |
| Python↔C 일치 테스트 | `tests/test_inference.py`가 gcc로 `loader.c`를 빌드해 torch 모델과 실시간 비교한다. 분리 후에는 돌 수 없다 | 1단계 4~6번 |
| 계약 문서 | 없음. 계산 규칙이 코드에만 있다 | 1단계 2번 |
| 모델 배포 | `.gguf`는 커밋하지 않고 각자 `src.exporter`로 만든다 | 1단계 7번 |
| CI | `pytest` job 하나에서 C 테스트도 함께 돈다 | 1단계 8번 |
| C 관련 문서 | `docs/exporter-loader.md`, `docs/postprocessor-c.md`, `CONVENTION.md`의 C 기준. `postprocessor-c.md`에 옛 경로(`..\..\..\data\shop.db`)와 로컬 절대 경로 링크(`file:///C:/Users/...`)가 있다 | 1단계 9번 |

## 4. 분리 후 구조와 배포 흐름

```
메인 저장소 (학습 · 원본)
  models/stage<N>/  model.pt, tokenizer.json, model_info.json   ← git에 커밋 (원본)
  태그 push (예: stage1-contract2)
    → GitHub Actions: src.exporter 로 model.gguf 생성
    → GitHub Release 첨부: model.gguf, model_info.json, shop.db, SHA256SUMS
        │
        ▼
CLI 저장소 (실행)
  engine/, postprocessor/      C 소스, Makefile 하나로 빌드
  CONTRACT.md                  계약 문서
  fixtures/v<N>/               작은 테스트 모델과 기대값 (git에 커밋, 수 KB)
  scripts/fetch-model <태그>    release에서 받아 sha256 확인 → models/
  models/                      .gitignore
```

- 메인 저장소 담당자는 학습 → `src.release` → 커밋 → 태그 push까지만 한다. export와 release 업로드는 workflow가 한다.
- CLI 저장소 담당자는 새 태그가 나오면 `fetch-model`로 받는다. 태그 이름에 단계와 계약 버전이 들어 있어, 어떤 모델이
  어떤 계약으로 도는지 태그만 보고 알 수 있다.
- CLI 테스트는 다운로드 없이 fixture로 돈다. 실제 모델은 직접 실행하거나 데모할 때만 받는다.
- **파일 크기**: `models/stage1/model.pt`는 20MB다. 계획서 파라미터 수로 계산하면 2단계(15M)는 약 60MB, 3단계(27.4M)는
  약 110MB로, GitHub의 git 파일당 100MB 제한에 걸린다. 3단계부터는 `model.pt`도 release로 옮긴다(release 첨부는 파일당
  2GB). 그때 `src.release`가 release용 묶음까지 만들도록 고친다.

## 5. 1단계: 같은 저장소 안에서 분리

9개 작업을 모두 구현했다. 진행 중 발견해 함께 고친 것: C 후처리기가 `-Werror`에서 빌드되지 않던 문제(쓰지 않는
`my_strdup`), `make clean`이 macOS·Linux에서 `nul` 파일을 만들던 문제, 후처리기의 저장소 밖 기본 경로
(`../../../data/shop.db`, `../../../data/semantic_dict_*.bin`) 제거.

| # | 작업 | 완료 조건 |
| --- | --- | --- |
| 1 | 최상위 `cli/`를 만들고 `src/loader.c` → `cli/engine/`, `src/postprocessor/c/` → `cli/postprocessor/`로 옮긴다. 둘 다 `cli/Makefile` 하나로 빌드한다 | `make -C cli` 성공 |
| 2 | **계약 문서** `cli/CONTRACT.md`: GGUF 메타데이터 키, 텐서 이름과 shape, 계산 규칙, BPE 알고리즘, 입력 형식, 디코딩 규칙. 현재 상태를 버전 2로 기록한다 | 이 문서만 읽고 C 엔진을 다시 구현할 수 있는 수준 |
| 3 | 계약 버전을 `src/tokenizer/contract.py`의 `CONTRACT_VERSION` 하나로 모으고 exporter가 이 값을 쓰게 한다 | Python 쪽 하드코딩 제거 |
| 4 | **fixture 생성기** `python -m src.export_fixtures` → `cli/fixtures/v2/`: 작은 `model.gguf`(현재 테스트의 vocab 320·dim 16 모델), 토큰화 케이스(문장 → id), 첫 스텝 logits, greedy 생성 결과, `manifest.json`(계약 버전, 생성 commit, sha256) | 커밋된 fixture가 재생성 결과와 비트 단위로 같음 |
| 5 | **Python 없이 도는 C 테스트**: `make -C cli test`가 `loader`의 `--encode`, `--logits`(허용 오차 포함), `--generate` 결과를 fixture와 비교한다 | torch 없는 환경에서 통과 |
| 6 | **버전 누락 방지 테스트**(메인): fixture를 재생성해 커밋된 것과 비교하고, 값이 바뀌었는데 `CONTRACT_VERSION`이 같으면 실패한다 | 계산 방식을 일부러 바꿨을 때 실패하는 것 확인 |
| 7 | **release workflow** `.github/workflows/release.yml`: `stage*-contract*` 태그 push 시 `models/stage<N>/model.pt`를 export해 release에 첨부하고 `SHA256SUMS`를 만든다. 저장소를 나누기 전에도 쓸 수 있다 | 태그 하나로 release 생성 확인 |
| 8 | **CI 분리**: 기존 `pytest` job과 새 `cli` job(gcc + `make -C cli test`, Python 없음). `cli/`가 `src/`나 `data/`를 참조하면 실패하는 grep 검사를 추가한다 | 두 job 모두 통과 |
| 9 | **문서 이동**: `docs/exporter-loader.md`의 C 부분, `docs/postprocessor-c.md`, `CONVENTION.md`의 C 코드 작성 기준을 `cli/` 아래로 옮기고, 옛 경로와 절대 경로 링크를 고친다. README 디렉터리 트리를 갱신한다 | `cli/` 밖에 C 관련 문서가 남지 않음 |

## 6. 2단계: 저장소 분리

1단계 PR이 병합되면 분리 직전에 한 번 끝까지 확인한다.

- 시험 태그(`stage1-contract2-rc1`)를 push해 release가 만들어지는지 확인한다.
- release에서 받은 `model.gguf`로 C 엔진과 Python 추론(`src.run --input`)의 SQL이 같은지 확인한다.
- 계산 규칙을 일부러 한 줄 바꿔 `tests/test_contract_fixtures.py`가 실패하는지 확인하고 되돌린다.

그다음 진행한다.

1. `git subtree split --prefix=cli -b cli-split`으로 히스토리를 보존한 채 새 저장소로 push한다.
2. 새 저장소에 CI(빌드 + fixture 테스트 + clang-format 검사)와 `scripts/fetch-model`을 둔다.
3. 메인 저장소에서 `cli/`를 지우고, `test_inference.py`의 Python↔C 실시간 비교를 없앤다. 그 역할은 1단계 6번 테스트가
   맡는다. fixture 생성기 출력 경로는 메인 저장소 안(`tests/fixtures/contract/`)으로 바꾼다.
4. 메인 저장소의 C 관련 CI(gcc 설치 전제)와 루트 `.clang-format`을 정리한다.
5. `test_inference.py`에 있는 C 엔진 오류 경로 테스트(잘못된 파일·인자 거부 등)는 torch로 만든 모델을 바이트
   변조해 쓰므로, CLI 저장소에서는 fixture `model.gguf`를 변조하는 방식으로 옮긴다. 옮기기 전에는 분리 후 이
   검사가 사라진다.

## 7. 운영 규칙 (분리 후)

- **계약이 바뀔 때**: 메인에서 `CONTRACT_VERSION`을 올리고 fixture를 재생성해 PR → 태그 push로 release → CLI 저장소에서
  새 fixture를 `fixtures/v<N>/`에 복사하고 새 버전을 지원하는 PR. CLI는 지원하는 버전 목록을 명시하고, 그 밖의 버전은
  지금처럼 에러로 거부한다.
- **버전을 올리는 경우**: 계산 규칙, BPE 알고리즘, 입력 형식, 디코딩 규칙, GGUF 메타데이터 키나 텐서 배치가 바뀔 때.
  예: 위치 토큰 도입, RoPE 구현 변경, 특수 토큰 추가.
- **버전을 올리지 않는 경우**: 모델 크기, vocab 크기, seed·merge 내용만 바뀔 때. 2단계 모델 전환은 GGUF 메타데이터로
  처리되므로 보통 여기에 해당한다.
- **SQL 문법이 바뀔 때**(2단계의 AND/OR, 비교 연산자, LIKE, ORDER BY, LIMIT): 모델 계약과 별개로 후처리기 parser를 함께
  고쳐야 한다. 단계 전환 체크리스트에 이 항목을 넣는다.

## 8. 남은 결합

- **후처리기 Python·C 동작 일치**: C 후처리기는 Python 구현(`src/postprocessor/python/`)과 같은 결과를 내도록
  만들어졌고 `test_postprocessor.c`가 Python 테스트 케이스를 옮겨 왔다. 분리 후에는 한쪽만 바뀌어도 잡히지 않는다.
  계속 맞출지, C 쪽을 기준으로 둘지 정해야 한다.
- **`semantic_dict_*.bin`**: GloVe(수백 MB)를 받아 Python으로 만드는 파일이라 release workflow에 넣지 않았다.
  semantic 교정을 배포에 쓸 거면 한 번 만들어 release에 수동으로 첨부한다.

## 9. 정할 것

| 항목 | 제안 |
| --- | --- |
| 새 저장소 위치와 이름 | 같은 조직(`polring`) 아래 |
| CLI 저장소 담당자 | 계약 변경 PR을 받아 줄 사람 한 명 |
