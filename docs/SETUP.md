# 환경 설정 가이드

저장소를 클론한 뒤 실행 환경을 만들고 각 단계를 재현하는 방법이다. 설계 이유는
[계획서](text-to-sql-llm-project-plan.md), 진행 상황과 수치는 [진행 보고서](text-to-sql-stage1-report.md)에
있고, 함수 단위 입출력은 각 소스 파일의 docstring에 있다.

**모든 명령은 저장소 루트에서 실행한다.** 경로가 현재 디렉터리 기준 상대 경로(`data/...`, `models/...`)라서
다른 위치에서 실행하면 파일을 찾지 못하거나 엉뚱한 곳에 만든다.

## 1. 요구 사항

- Python 3.10 이상 (개발 환경 3.13)
- NVIDIA GPU (학습용, 개발 환경 RTX 4070 Super). 추론·평가는 CPU로도 된다(`--cpu`).
- DB·SQL·질문 생성(`src/data/db_gen.py`, `src/data/stage1/` 중 `sql_gen`·`question_gen`·`question_gen_auto`)과
  `src/tokenizer/bpe.py`는 표준 라이브러리만 쓴다. Dataset·모델·학습·추론·테스트에는 아래 가상환경이 필요하다.

## 2. 가상환경과 PyTorch 설치

```
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

마지막 줄이 `2.14.0+cu130 True NVIDIA GeForce ...`처럼 `True`를 출력해야 한다.

- `requirements.txt`가 PyTorch CUDA 빌드 인덱스(`--extra-index-url`)를 가리킨다. Windows용 CUDA 빌드는 일반
  PyPI에 없으므로 `pip install torch`를 따로 하지 말고 반드시 `requirements.txt`로 설치한다.
- `False`가 나오면 드라이버와 빌드 태그가 맞지 않는 경우가 대부분이다. `nvidia-smi`의 "CUDA Version"(드라이버가
  지원하는 최대 버전) 이하인 빌드 태그(`cuNNN`, https://download.pytorch.org/whl/ 에서 확인)로
  `requirements.txt`의 인덱스와 torch 버전을 바꾼다.
- 이후 `python`은 `.venv\Scripts\python`(또는 `.venv\Scripts\activate` 후의 셸)을 뜻한다.

## 3. 폴더 구조와 경로

경로는 모두 `src/config.py`에 있고, 현재 단계(`STAGE`)에 따라 `stage<N>` 폴더를 가리킨다. 역할은 **폴더 위치**로 정해진다.

```
data/stage1/
├── db/          shop.db, holdout.json
├── sql/         train.json, eval_indist.json, eval_holdout.json, gen_report.json
├── train/       학습 쌍 (*.json 전부가 학습 데이터·BPE 코퍼스)
├── eval/        평가셋마다 폴더 하나: pairs.json (질문, SQL) + sql.json (그 SQL 의 메타데이터)
└── tokenizer/   seed_tokens.json (seed 토큰, 사람이 편집), tokenizer.json (BPE 학습 결과)
data/wordnet/    이름 교체용 영단어 목록 (단계 공통)
models/stage1/   배포 모델: model.pt, tokenizer.json, model_info.json
data_raw/stage1/ 질문 생성 라운드·중간 산출물 (커밋 안 함)
runs/stage1/     학습 실행·평가·추론 결과 (커밋 안 함)
```

- **모든 실행 명령은 읽는 파일과 쓰는 위치를 인자로 바꿀 수 있고, 생략하면 위 기본값을 쓴다.** 인자 목록은
  각 명령의 `--help`에 있다.
- 단계마다 다른 코드는 `src/data/stage<N>/`에 있다(SQL 생성, 질문 생성, 라벨 정제, 이름 교체, 채점 규칙 `sql_rules.py`).

## 4. 클론 직후 할 일

DB, SQL, 질문 쌍, 토크나이저, 배포 모델이 모두 커밋되어 있어 학습 없이 바로 쓸 수 있다.

```
python -m pytest tests/ -v                                   # 단위 테스트
python -m src.run --input "what city does ashley live in?"   # 배포 모델로 질문 → SQL → DB 실행 결과
python -m src.run --evaluation                               # 배포 모델로 평가셋 전부 채점
python -m src.train --overfit 100                            # 과적합 테스트 (GPU, 약 40초)
python -m src.train                                          # 본 학습 + 평가 (GPU, 약 10분)
```

데이터를 처음부터 다시 만들거나 새로 추가할 때만 5~7절을 따른다.

## 5. DB·SQL 생성

코드와 시드로 바이트 단위까지 같은 결과가 나온다.

```
python -m src.data.db_gen build              # data/stage1/db/shop.db, holdout.json
python -m src.data.db_gen test               # 분포·결정성·price 등호·홀드아웃 분리 검사
python -m src.data.stage1.sql_gen build --cap-nonid 9999 --cap-id 9999   # data/stage1/sql/train.json, eval_indist.json
python -m src.data.stage1.sql_gen verify     # 완료 조건 검사
python -m src.data.stage1.sql_gen test       # 결정성·홀드아웃 리터럴 미포함·학습/평가 분리 검사
python -m src.data.stage1.sql_gen holdout    # data/stage1/sql/eval_holdout.json (미등장 값 평가 SQL)
```

주요 옵션: `--seed`(기본 0), `--out`(출력 폴더, 기본 `data/stage1/db` / `data/stage1/sql`), `sql_gen`의
`--db-dir`(읽을 DB 폴더), `db_gen`의 `--preset small|large`(기본 `large`)와 `--name-style single|two_word`
(`--preset small`은 80 / 40 / 150행 원안 스키마, `two_word`는 두 단어 이름). `db_gen space`는 SQL 공간과 ID 조건
비중을, `summary`는 분포·할당량 표를 출력한다. 결정성 판정은 SQLite 파일 바이트가 아니라 행 내용 기준 해시로 한다.

## 6. 질문 생성

LLM 응답은 결정적이지 않아서 같은 명령을 다시 돌려도 같은 문장이 나오지 않는다. 같은 규칙과 검증을 통과하는
동등한 데이터가 나온다는 의미로만 재현된다. 프롬프트 규칙은 `src/data/stage1/question_gen.py`의 `RULES_BLURB`
(약한 모델용은 `small_model_rules_blurb()`), 검증 항목은 `check_question()`·`soft_flags()`에 있다.

### 방법 A — 사람이 LLM 세션에 전달 (`question_gen`)

```
python -m src.data.stage1.question_gen make-prompts --out data_raw/stage1/<라운드> --n 200 --batch-size 25 --seed N \
    --exclude-dir data_raw/stage1/<이전 라운드>,...   # 이전 라운드 폴더 전부 (SQL 중복 방지)
```

1. `prompts/batch_NN.txt`를 **배치마다 서로 독립된 LLM 세션**에 하나씩 전달하고, 응답 JSON을
   `responses/batch_NN.json`으로 저장한다. 한 세션이 여러 배치를 처리하면 문장 표현이 획일화된다.
2. 검증과 다양성 진단:
   ```
   python -m src.data.stage1.question_gen validate --out data_raw/stage1/<라운드>         # 통과분 → pairs.json
   python -m src.data.stage1.question_gen template-check --out data_raw/stage1/<라운드>   # 표현 반복 비율 진단
   ```
- `--exclude-dir`에는 `pilot_sql.json`이 있는 원본 라운드 폴더(`data_raw/` 아래)를 넘긴다.
- 입력 SQL은 `--train`(기본 `data/stage1/sql/train.json`)으로 바꾼다. 평가용 질문은 `--train data/stage1/sql/eval_indist.json`
  또는 `eval_holdout.json`처럼 평가 SQL 파일을 넘긴다. `question_gen_auto`도 같은 옵션을 쓴다.
- 약한/로컬 모델용 프롬프트는 `make-prompts-small`(배치 크기 작음, 말투 지시가 더 구체적, `--questions-per-sql`).

### 방법 B — 로컬 모델 자동 호출 (`question_gen_auto`)

Ollama 서버가 `127.0.0.1:11434`에서 실행 중이고 모델을 받아 둔 상태(`ollama pull qwen3:14b`)여야 한다.

```
python -m src.data.stage1.question_gen_auto --model qwen3:14b --n 128 --batch-size 8 --questions-per-sql 8 --seed 0
```

샘플링 → 프롬프트 → 모델 호출 → JSON 복구 → 검증 → 다양성 진단을 한 번에 하며, 기본 출력은
`data_raw/stage1/auto_pilot/`이다. 중간에 끊겨도 같은 명령을 다시 실행하면 완료된 배치는 건너뛴다.

### 결과를 코퍼스에 넣기

1. 통과한 `pairs.json`을 직접 읽어 품질을 확인한 뒤 옮긴다.
   - 학습용: `data/stage1/train/<이름>.json` — 이 폴더의 `*.json`은 전부 학습 데이터와 토크나이저 코퍼스가 된다.
   - 평가용: `data/stage1/eval/<이름>/pairs.json`으로 옮기고 메타데이터를 만든다. 평가셋은 이 폴더만 있으면
     `run`·`train`·`release`가 자동으로 찾는다.
     ```
     python -m src.run --build-sql <이름> --sql data/stage1/sql/eval_indist.json   # 그 평가셋의 SQL 파일
     ```
2. 라벨 충돌 검사 후, 충돌이 있으면 양쪽 쌍을 모두 제거한다(기본: 학습 쌍 전부 + 분포 내 평가셋).
   ```
   python -m src.data.stage1.question_gen corpus-check
   ```
3. 라벨 규칙(계획서 4-5 L2~L4)과 어긋나는 학습 쌍을 제거한다. 제거 목록은
   `data_raw/stage1/clean_ambiguous_removed.json`에 누적된다.
   ```
   python -m src.data.stage1.clean_ambiguous           # dry-run
   python -m src.data.stage1.clean_ambiguous --apply
   ```
4. 학습 코퍼스가 바뀌었으면 7절의 토크나이저를 다시 학습한다.

## 7. 토크나이저·학습·배포

### 토크나이저

```
python -m src.tokenizer.bpe    # seed_tokens.json + 학습 쌍 → data/stage1/tokenizer/tokenizer.json, round-trip·미등장 이름 토큰 수 출력
python -m src.data.dataset     # 전체 쌍 토큰화 길이 통계, round-trip, collate_fn/loss_mask 확인
```

- 코퍼스와 seed 목록이 같으면 `tokenizer.json`은 항상 같은 파일로 나온다.
- seed 토큰(SQL 키워드, 테이블·컬럼 이름, 범주형 값)은 `seed_tokens.json`에서 관리한다. 토큰 ID가 목록 순서로
  정해지므로 순서를 바꾸거나 추가하면 토크나이저와 모델을 다시 학습해야 한다.
- 입력·출력 바꾸기: `--train-files`, `--seeds`, `--vocab-size`, `--out`.

### 학습

```
python -m src.train --overfit 100          # 학습 예시 100개로 EM 100% 도달 확인
python -m src.train                        # 본 학습(이름 교체 40% 기본) → runs/stage1/ffn683_swap40/, 끝나면 평가셋 전부 측정
python -m src.train --name-swap-ratio 0    # 이름 교체 없는 기준 실행 → runs/stage1/ffn683/
.venv\Scripts\tensorboard --logdir runs    # 학습 곡선
```

- 산출물: `runs/stage1/<이름>/`에 `best.pt`(검증 EM 최고 시점 체크포인트, 모델 설정 포함), `tokenizer.json`
  (학습에 쓴 토크나이저 사본), `metrics.json`(epoch별 기록, 평가 요약), `eval_<평가셋>.json`(요약 + 틀린 사례 전체),
  `tb/`(TensorBoard 로그).
- 검증셋은 학습 쌍에서 SQL 단위로 5%를 떼어 내고, 체크포인트 선택·조기 종료는 이것으로만 한다. 평가셋은 학습이
  끝난 뒤 한 번만 측정한다.
- 주요 옵션: `--epochs`(기본 25), `--patience`(기본 10), `--lr`, `--batch-size`, `--ffn-dim`(config를 고치지
  않고 FFN 크기만 바꿔 실험), `--seed`, `--run-name`. 입력 바꾸기: `--train-files`(여러 개), `--tokenizer`,
  `--eval-dir`, `--db`. 최종 평가할 평가셋 고르기: `--eval-sets`(기본 전부). 실행 폴더 직접 지정: `--out`
  (기본 `runs/stage<N>/<run-name>/`).
- 모델 구조는 `src/config.py`의 `STAGE` 값으로 정해진다.
- 이름 교체: 기본 학습이 매 epoch 이름 조건 쌍의 40%를 가짜 이름으로 바꾼다(`--name-swap-ratio`로 비율 변경, 0이면 끔).
  가짜 이름에서 영단어를 걸러 내는 데 `data/wordnet/english_words.txt`(WordNet 3.0 표제어, 라이선스는 같은 폴더)를
  쓴다. 다른 목록은 `--wordlist`로 지정한다(목록이 다르면 가짜 이름과 결과가 달라진다).
- 고정 파일 방식: 파일을 만들어 `--train-files`로 학습한다. 이미 이름을 바꾼 파일이면 학습 중 이름 교체는 자동으로 꺼진다.
  ```
  python -m src.data.stage1.name_swap --ratio 0.4   # → data_raw/stage1/name_swap_40/train_pairs.json (커밋 안 함)
  python -m src.train --train-files data_raw/stage1/name_swap_40/train_pairs.json --run-name name_swap_40
  ```

### 배포 모델 만들기

```
python -m src.release --ckpt runs/stage1/ffn683_swap40/best.pt    # → models/stage1/
```

- `model.pt`(가중치·모델 설정·단계만, 학습 인자와 로컬 경로 없음), `tokenizer.json`(이 모델이 학습에 쓴
  토크나이저), `model_info.json`(출처 체크포인트, 학습 설정 요약, 검증 EM, 내보낸 파일로 다시 측정한 평가 결과)을 쓴다.
- 단계마다 최종 모델 하나만 커밋한다. 다시 내보내면 덮어쓴다. `--no-eval`로 평가를 생략할 수 있다.

## 8. 추론·평가 (`run.py`)

추론과 평가는 `python -m src.run` 하나로 한다. 모델 로딩·채점 함수는 `src/evaluation.py`(학습·배포와 같은 함수)에 있다.
학습 없이도 커밋된 배포 모델(`models/stage<N>/model.pt`)로 바로 실행된다.

### 입력과 모드에 따른 동작

| `--input` | 모드 | 동작 | 결과 |
| --- | --- | --- | --- |
| 생략 | `--test` 또는 생략 | 대화형: 질문을 한 줄씩 입력 | 화면 출력 |
| 생략 | `--evaluation` | 이 단계의 평가셋 전부 채점 | 평가셋마다 `eval_<이름>.json` |
| 생략, `--eval-sets a b` | `--evaluation` 또는 생략 | 고른 평가셋만 채점 | 평가셋마다 `eval_<이름>.json` |
| 생략, `--eval-sets a b` | `--test` | 고른 평가셋을 채점 없이 배치 추론 | 평가셋마다 예측 JSON |
| 질문 문자열 | `--test` 또는 생략 | 단건 추론 | 화면: SQL + DB 실행 결과 |
| 질문 문자열 | `--evaluation` | 정답이 없다는 안내 후 단건 추론 | 화면 |
| JSON 파일 (정답 없음) | `--test` 또는 생략 | 배치 추론 | 예측 JSON |
| JSON 파일 (정답 없음) | `--evaluation` | 안내 후 배치 추론 | 예측 JSON |
| 평가셋 이름·폴더, 정답 있는 JSON | `--evaluation` 또는 생략 | 채점: EM, 다중 정답 EM, 실행 정확도, 신뢰구간, 계층·WHERE 컬럼별, 오답 분류 | 화면 요약 + `eval_<이름>.json` |
| 평가셋 이름·폴더, 정답 있는 JSON | `--test` | 채점 없이 배치 추론 | 예측 JSON (정답 함께 기록) |

- `--input`이 존재하는 폴더면 평가셋 폴더, `.json`이면 파일, 평가셋 이름과 같으면 평가셋, 그 밖에는 질문 문자열로 본다.
- JSON 형식: `[{"question": ..., "sql": ...}]` 또는 `["질문", ...]`. 같은 폴더에 `sql.json`이 있으면 계층별 집계에 쓴다.
- 예측 JSON의 각 항목: `question`, `pred`(생성 SQL, 실패하면 null), `stage_format`(그 단계 SQL 형식인지), 입력에 있었다면 `sql`.
- 질문은 학습 데이터처럼 소문자로 바꿔 넣는다. 생성은 greedy이며 제약 디코더는 없다.

### 인자

| 인자 | 생략하면 |
| --- | --- |
| `--stage N` | `config.STAGE`. 기본 모델·평가셋·DB·채점 규칙을 이 단계로 정한다 |
| `--model` | `models/stage<N>/model.pt`. 폴더를 주면 그 안의 `model.pt`, 없으면 `best.pt` |
| `--eval-sets` | `--input` 없이 쓸 때 처리할 평가셋 이름들. 생략하면 전부 (`--input`과 함께 쓸 수 없음) |
| `--output` (`--out`) | 평가: `runs/` 안의 체크포인트면 그 폴더, 배포 모델이면 `runs/stage<N>/release_eval/`. 추론: `runs/stage<N>/predictions/<입력 이름>.json`. `.json`을 주면 그 파일(평가셋이 여러 개면 한 파일로 묶음), 폴더를 주면 그 안에 저장 |
| `--tokenizer` | 모델 폴더의 `tokenizer.json` |
| `--db` | `data/stage<N>/db/shop.db` |
| `--eval-dir` | `data/stage<N>/eval/` |
| `--batch-size` | 512 |
| `--show` | 출력할 오답·예측 예시 수 (10) |
| `--no-exec`, `--cpu`, `--no-amp` | DB 실행 끄기, CPU 사용, bf16 끄기 |

### 예시

```
python -m src.run                                              # 대화형
python -m src.run --input "what city does ashley live in?"     # 단건
python -m src.run --evaluation                                 # 평가셋 전부
python -m src.run --eval-sets indist holdout                   # 고른 평가셋만
python -m src.run --input holdout --output results/holdout.json
python -m src.run --input questions.json --test                # 정답 없는 질문 목록 배치 추론
python -m src.run --model runs/stage1/ffn683_swap40 --evaluation   # 직접 학습한 체크포인트 평가
python -m src.run --list-sets                                  # 평가셋 목록
python -m src.run --build-sql indist --sql data/stage1/sql/eval_indist.json   # 평가셋 sql.json 생성
```

- 결과 파일은 기본으로 `runs/` 아래(커밋 안 함)에 저장된다. `models/`는 커밋 폴더라 결과를 쓰지 않는다.
- `--stage 2`처럼 아직 모델이 없는 단계를 주면 안내 후 종료한다.

## 9. 재현 확인 기준

| 명령 / 파일 | 기대 결과 |
| --- | --- |
| `python -m src.tokenizer.bpe` | 병합 726/726, round-trip 실패 0건, 미등장 이름 평균 분할 토큰 ≈ 3.52, `tokenizer.json`이 커밋된 파일과 같음 |
| `python -m src.data.dataset` | 시퀀스 길이 22~64(평균 38.3), 컨텍스트 128 초과 0건, `loss_mask` 검증 통과 |
| `python -m pytest tests/ -v` | 전체 통과 |
| `python -m src.train --overfit 100` | EM 100% 도달 |
| `python -m src.train` (seed 0) | 파라미터 4,985,600, 최고 검증 EM 24 epoch 0.990, 분포 내 EM ≈ 0.967, 미등장 값 EM ≈ 0.907 |
| `python -m src.train --name-swap-ratio 0` (seed 0) | 이름 교체 없는 기준: 분포 내 EM ≈ 0.955, 미등장 값 EM ≈ 0.712 |
| `python -m src.run --evaluation` | `models/stage1/model_info.json`의 평가 결과와 같음 (분포 내 0.967, 미등장 값 0.907) |
| `data/stage1/sql/gen_report.json` | 구조별 할당량이 층화 샘플링 방침(계획서 5-3)대로, ID 조건 비중 30% 이하 |

## 10. 커밋 대상

- **커밋**: `src/`, `tests/`, `docs/`, `CLAUDE.md`, `README.md`, `requirements.txt`, `data/stage<N>/`
  (DB, SQL, 학습 쌍, 평가셋, seed 토큰, 토크나이저), `data/wordnet/`, `models/stage<N>/`(단계별 최종 배포 모델 하나).
- **커밋 안 함** (`.gitignore`): `.venv/`, `data_raw/`(LLM 라운드 원본·중간 산출물), `qwen3_14b_test/`,
  `runs/`(학습·평가·추론 결과).
