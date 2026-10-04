# 환경 설정 가이드

저장소를 클론한 뒤 실행 환경을 만들고 각 단계를 재현하는 방법이다. 설계 이유는
[계획서](text-to-sql-llm-project-plan.md), 진행 상황과 수치는 [진행 보고서](text-to-sql-stage1-report.md)에
있고, 함수 단위 입출력은 각 소스 파일의 docstring에 있다.

**모든 명령은 저장소 루트에서 실행한다.** 스크립트가 현재 디렉터리 기준 상대 경로(`data/...`,
`bpe_merges.json`)를 쓰므로, 다른 위치에서 실행하면 `data/`가 엉뚱한 곳에 생긴다.

## 1. 요구 사항

- Python 3.10 이상 (개발 환경 3.13)
- NVIDIA GPU (학습용, 개발 환경 RTX 4070 Super)
- 데이터 생성 스크립트(`src/data/generator/*.py`)와 `bpe.py`는 표준 라이브러리만 쓴다. Dataset·모델·학습·
  테스트에는 아래 가상환경이 필요하다.

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

## 3. 클론 직후 할 일

DB, SQL, 검증된 질문 쌍은 커밋되어 있으므로 다시 만들 필요가 없다. 토크나이저 산출물만 만들면 바로 학습할
수 있다.

```
python -m src.tokenizer.bpe      # bpe_merges.json 생성 (항상 같은 결과)
python -m src.data.dataset       # Dataset 자체 테스트 (파일 저장 없음)
python -m pytest tests/ -v       # 단위 테스트
python -m src.train --overfit 100    # 과적합 테스트 (약 40초)
python -m src.train                  # 본 학습 + 평가 (이름 교체 40% 기본 적용, 약 10분)
```

데이터를 처음부터 다시 만들거나 새로 추가할 때만 4~6절을 따른다.

## 4. DB·SQL 생성

코드와 시드로 바이트 단위까지 같은 결과가 나온다.

```
python src/data/generator/db_gen.py build    # data/shop.db, data/holdout.json
python src/data/generator/db_gen.py test     # 분포·결정성·price 등호·홀드아웃 분리 검사
python src/data/generator/sql_gen.py build --cap-nonid 9999 --cap-id 9999   # data/sql_train.json, data/sql_eval_indist.json
python src/data/generator/sql_gen.py verify  # 완료 조건 검사
python src/data/generator/sql_gen.py test    # 결정성·홀드아웃 리터럴 미포함·학습/평가 분리 검사
python src/data/generator/sql_gen.py holdout # data/sql_eval_holdout.json (미등장 값 평가 SQL)
```

주요 옵션: `--seed`(기본 0), `--out`(기본 `./data`), `db_gen.py`의 `--preset small|large`(기본 `large`)와
`--name-style single|two_word`(`--preset small`은 80 / 40 / 150행 원안 스키마, `two_word`는 두 단어 이름). `db_gen.py space`는
SQL 공간과 ID 조건 비중을, `summary`는 분포·할당량 표를 출력한다. 결정성 판정은 SQLite 파일 바이트가 아니라
행 내용 기준 해시로 한다.

## 5. 질문 생성

LLM 응답은 결정적이지 않아서 같은 명령을 다시 돌려도 같은 문장이 나오지 않는다. 같은 규칙과 검증을 통과하는
동등한 데이터가 나온다는 의미로만 재현된다. 프롬프트 규칙은 `question_gen.py`의 `RULES_BLURB`
(약한 모델용은 `small_model_rules_blurb()`), 검증 항목은 `check_question()`·`soft_flags()`에 있다.

### 방법 A — 사람이 LLM 세션에 전달 (`question_gen.py`)

```
python src/data/generator/question_gen.py make-prompts --out data_raw/<라운드> --n 200 --batch-size 25 --seed N \
    --exclude-dir data_raw/pilot,data_raw/pilot2,...   # 이전 라운드 폴더 전부 (SQL 중복 방지)
```

1. `prompts/batch_NN.txt`를 **배치마다 서로 독립된 LLM 세션**에 하나씩 전달하고, 응답 JSON을
   `responses/batch_NN.json`으로 저장한다. 한 세션이 여러 배치를 처리하면 문장 표현이 획일화된다.
2. 검증과 다양성 진단:
   ```
   python src/data/generator/question_gen.py validate --out data_raw/<라운드>         # 통과분 → pilot_train_pairs.json (폴더명 eval* 이면 eval_pairs.json)
   python src/data/generator/question_gen.py template-check --out data_raw/<라운드>   # 표현 반복 비율 진단
   ```
- `--exclude-dir`에는 `pilot_sql.json`이 있는 원본 라운드 폴더(`data_raw/` 아래)를 넘긴다. `data/`의 병합
  폴더에는 이 파일이 없어 쓸 수 없다.
- 입력 SQL은 `--train`(기본 `data/sql_train.json`)으로 바꾼다. 평가용 질문은 `--train data/sql_eval_indist.json`
  또는 `data/sql_eval_holdout.json`처럼 평가 SQL 파일을 넘긴다. `question_gen_auto.py`도 같은 옵션을 쓴다.
- 약한/로컬 모델용 프롬프트는 `make-prompts-small`(배치 크기 작음, 말투 지시가 더 구체적, `--questions-per-sql`).

### 방법 B — 로컬 모델 자동 호출 (`question_gen_auto.py`)

Ollama 서버가 `127.0.0.1:11434`에서 실행 중이고 모델을 받아 둔 상태(`ollama pull qwen3:14b`)여야 한다.

```
python src/data/generator/question_gen_auto.py --model qwen3:14b --n 128 --batch-size 8 --questions-per-sql 8 --seed 0
```

샘플링 → 프롬프트 → 모델 호출 → JSON 복구 → 검증 → 다양성 진단을 한 번에 하며, 기본 출력은
`data_raw/auto_pilot/`이다. 중간에 끊겨도 같은 명령을 다시 실행하면 완료된 배치는 건너뛴다.

### 결과를 코퍼스에 넣기

1. 통과한 쌍 파일을 직접 읽어 품질을 확인한 뒤 `data/` 아래 폴더로 옮긴다.
   - 학습용: `data/pilot_<이름>/` — `pilot*` 폴더는 학습 데이터와 토크나이저 코퍼스에 자동으로 포함된다.
   - 평가용: `data/eval_<이름>/` — **절대 `pilot*`로 이름 붙이지 않는다.** 평가 코드는 `eval_*/eval_pairs.json`만
     읽으므로, 평가 라운드는 원본 폴더부터 `eval*`로 이름 붙여 `validate`가 `eval_pairs.json`으로 저장하게 한다.
2. 라벨 충돌 검사 후, 충돌이 있으면 양쪽 쌍을 모두 제거한다.
   ```
   python src/data/generator/question_gen.py corpus-check --data-dir data
   ```
3. 라벨 규칙(계획서 4-5 L2~L4)과 어긋나는 학습 쌍을 제거한다. 제거 목록은
   `data_raw/clean_ambiguous_removed.json`에 누적된다.
   ```
   python -m src.data.clean_ambiguous           # dry-run
   python -m src.data.clean_ambiguous --apply
   ```
4. 학습 코퍼스가 바뀌었으면 6절의 토크나이저와 Dataset을 다시 돌린다.

## 6. 토크나이저·Dataset

```
python -m src.tokenizer.bpe    # data/pilot*/ 코퍼스로 병합 726개 학습 → bpe_merges.json, round-trip·미등장 이름 토큰 수 출력
python -m src.data.dataset     # 전체 쌍 토큰화 길이 통계, round-trip, collate_fn/loss_mask 확인
```

코퍼스가 같으면 `bpe_merges.json`은 항상 같은 파일로 나온다.

## 7. 학습

```
python -m src.train --overfit 100          # 학습 예시 100개로 EM 100% 도달 확인
python -m src.train                        # 본 학습(이름 교체 40% 기본) → runs/stage<N>_ffn<D>_swap40/, 끝나면 평가셋 전부 측정
python -m src.train --name-swap-ratio 0    # 이름 교체 없는 기준 실행 → runs/stage<N>_ffn<D>/
python -m src.evaluation --ckpt runs/stage1_ffn683_swap40/best.pt   # 저장된 체크포인트만 다시 평가
.venv\Scripts\tensorboard --logdir runs    # 학습 곡선
```

- 각 실행 폴더에 학습 때 쓴 `bpe_merges.json` 사본이 저장되고, `evaluation`·`predict`는 이 사본을 먼저 쓴다.
  그래서 코퍼스가 바뀌어 루트의 `bpe_merges.json`을 다시 만들어도 이전 체크포인트를 그대로 평가할 수 있다.

- 학습된 모델 직접 써 보기: 질문을 SQL로 바꾸고 `shop.db`에서 실행한 결과까지 출력한다. 기본 체크포인트는
  위 본 학습이 저장하는 `runs/stage1_ffn683_swap40/best.pt`이고, 질문은 소문자로 바꿔 넣는다.
  ```
  python -m src.predict                                   # 대화형 (빈 줄로 종료)
  python -m src.predict -q "what city does ashley live in?" -q "..."
  python -m src.predict --ckpt runs/stage1_ffn683/best.pt --no-exec   # 다른 체크포인트(이름 교체 없는 기준), 실행 생략
  ```
- 산출물: `runs/<이름>/best.pt`(검증 EM 최고 시점 체크포인트, 모델 설정 포함), `metrics.json`(epoch별 기록,
  평가 요약), `eval_indist.json`·`eval_holdout.json`·`eval_holdout_qwen.json`(평가셋별 요약 + 틀린 사례 전체), `tb/`(TensorBoard 로그).
- 평가 지표: EM(95% 신뢰구간 포함), 실행 정확도(`shop.db` 실행 결과 비교), 계층·WHERE 컬럼별 EM, 오답의
  틀린 부분(테이블·WHERE 컬럼·SELECT·리터럴). `src/evaluation.py`의 `--sets`로 평가셋을 고른다.
- 주요 옵션: `--epochs`(기본 25), `--patience`(기본 10), `--lr`, `--batch-size`, `--ffn-dim`(config를 고치지
  않고 FFN 크기만 바꿔 실험), `--seed`, `--run-name`.
- 모델 구조는 `src/config.py`의 `STAGE` 값으로 정해진다.
- 이름 교체: 기본 학습이 매 epoch 이름 조건 쌍의 40%를 가짜 이름으로 바꾼다(`--name-swap-ratio`로 비율 변경, 0이면 끔).
  고정 파일로 실험하려면 파일을 만든 뒤 `--train-file`로 학습한다.
- **WordNet 파일이 필요하다:** 가짜 이름에서 영단어를 걸러 내려고 NLTK WordNet 데이터
  (`%APPDATA%\nltk_data\corpora\wordnet.zip`)를 표준 라이브러리 `zipfile`로 읽는다. NLTK 패키지는 필요
  없다. 이 파일이 없으면 기본 학습이 `영단어 목록 없음` 오류로 멈추고, 다른 위치나 한 줄에 한 단어인 텍스트
  파일은 `--wordlist`로 지정한다(목록이 다르면 가짜 이름과 결과가 달라진다). 이름 교체 없이 학습하려면
  `--name-swap-ratio 0`.
  ```
  python -m src.data.name_swap --ratio 0.4   # 고정 파일 방식 → data_raw/name_swap_40/train_pairs.json (커밋 안 함)
  python -m src.train --train-file data_raw/name_swap_40/train_pairs.json --run-name name_swap_40
  ```

## 8. 재현 확인 기준

| 명령 / 파일 | 기대 결과 |
| --- | --- |
| `python -m src.tokenizer.bpe` | 병합 726/726, round-trip 실패 0건, 미등장 이름 평균 분할 토큰 ≈ 3.52 |
| `python -m src.data.dataset` | 시퀀스 길이 22~64(평균 38.3), 컨텍스트 128 초과 0건, `loss_mask` 검증 통과 |
| `python -m pytest tests/ -v` | 전체 통과 |
| `python -m src.train --overfit 100` | EM 100% 도달 |
| `python -m src.train` (seed 0) | 파라미터 4,985,600, 최고 검증 EM 24 epoch 0.990, 분포 내 EM ≈ 0.967, 미등장 값 EM ≈ 0.907 |
| `python -m src.train --name-swap-ratio 0` (seed 0) | 이름 교체 없는 기준: 분포 내 EM ≈ 0.955, 미등장 값 EM ≈ 0.712 |
| `data/sql_gen_report.json` | 구조별 할당량이 층화 샘플링 방침(계획서 5-3)대로, ID 조건 비중 30% 이하 |

## 9. 커밋 대상

- **커밋**: `src/`, `tests/`, `docs/`, `CLAUDE.md`, `README.md`, `requirements.txt`, 그리고 `data/`의
  `shop.db`, `holdout.json`, `sql_*.json`, `sql_gen_report.json`, `pilot_merged/pilot_train_pairs.json`,
  `eval_indist_merged/eval_pairs.json`, `eval_holdout_merged/eval_pairs.json`, `eval_qwen_holdout_merged/eval_pairs.json`.
- **커밋 안 함** (`.gitignore`): `.venv/`, `data_raw/`(LLM 라운드 원본·중간 산출물), `qwen3_14b_test/`,
  `bpe_merges.json`(3절 명령으로 재생성), `runs/`(학습 산출물).
