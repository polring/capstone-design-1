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

## 3. 저장소 구조와 경로

경로는 모두 `src/config.py`에 있고, 현재 단계(`STAGE`)에 따라 `stage<N>` 폴더를 가리킨다. 역할은 **폴더 위치**로 정해진다.

- **모든 실행 명령은 읽는 파일과 쓰는 위치를 인자로 바꿀 수 있고, 생략하면 아래 기본값을 쓴다.** 인자 목록은 각 명령의 `--help`에 있다.
- 단계마다 다른 코드는 `src/data/stage<N>/`에, 단계마다 다른 데이터는 `data/stage<N>/`에 있다. 함수 단위 입출력은 각 파일의 docstring에 있다.

### 코드 (`src/`)

```
src/
├── config.py                 단계별 모델 구조, 학습 설정, 데이터·모델 경로 (STAGE 하나로 전환)
├── train.py                  학습 루프 → runs/stage<N>/<run>/ (7절)
├── release.py                학습 체크포인트 → 배포 모델 models/stage<N>/ (7절)
├── run.py                    추론·평가 통합 명령: 단건 / JSON 배치 / 대화형, --test / --evaluation (8절)
├── demo.py                   파이프라인 단계별 입력 → 출력 보기 (9절)
├── exporter.py               체크포인트 → GGUF (C 추론 엔진용, docs/exporter-loader.md)
├── models/
│   ├── embedding.py          TokenEmbedding, RoPE
│   ├── transformer.py        RMSNorm, Attention, SwiGLU FFN, 디코더 블록
│   └── model.py              전체 모델 조립, greedy 생성
├── tokenizer/
│   └── bpe.py                바이트 레벨 BPE 토크나이저 (Tokenizer)
├── inference/
│   ├── loading.py            체크포인트·배포 모델 불러오기, 짝이 맞는 토크나이저 찾기
│   ├── generate.py           질문 → SQL greedy 생성 (배치), 디코딩
│   └── gguf_model.py         GGUF 내보내기용 체크포인트 계약 모델 (exporter.py 가 사용)
├── evaluation/
│   ├── scoring.py            채점·집계: EM, 다중 정답 EM, 규칙 일치 EM, 실행 정확도, 신뢰구간
│   └── eval_sets.py          평가셋 읽기(폴더 방식 / 합친 방식), 평가 결과 위치
├── data/
│   ├── dataset.py            <bos> 질문 <sep> SQL <eos> 토큰화, loss_mask, collate_fn
│   ├── db_gen.py             SQLite DB·미등장 값 생성 (스키마가 같은 단계 공통)
│   ├── stage1/
│   │   ├── sql_gen.py            SQL 생성 (학습 / 분포 내 / 미등장 값)
│   │   ├── question_gen.py       LLM 질문 생성 (사람이 세션에 전달) · 검증
│   │   ├── question_gen_auto.py  로컬 Ollama 모델 자동 호출
│   │   ├── clean_ambiguous.py    라벨 규칙(L2~L4) 위반 학습 쌍 제거
│   │   ├── name_swap.py          이름 교체 (가짜 이름 생성)
│   │   └── sql_rules.py          1단계 채점 규칙 (SQL 분해, 오답 분류, 다중 정답, 규칙 일치)
│   └── stage2/
│       ├── schema.py             컬럼 종류별 허용 연산자·정렬 가능 여부·리터럴 표기
│       ├── sql_ast.py            SQL 구조(Query / Cond), 정규형 렌더링, 역파싱, 문법 규칙 검사
│       ├── sql_gen.py            SQL 생성 (학습 / 분포 내 / 구조 홀드아웃 / 미등장 값)
│       ├── compose.py            질문 조합기: SQL 구조 → 질문 (문장 틀 × 표현 사전 × 유사어), 학습 쌍 생성
│       ├── clean_ambiguous.py    2단계 라벨 규칙 판정 (L1·L2·L3·L4·L14)
│       ├── question_gen.py       LLM 질문: 외부 말투 평가 / 학습 질문, 배포 묶음, 응답 검증, 학습 폴더로 옮기기
│       ├── synonym_swap.py       LLM 학습 질문의 epoch 마다 유사어 교체
│       ├── name_swap.py          2단계 이름 교체 (IN 의 모든 이름, LIKE 조각)
│       ├── sql_rules.py          2단계 채점 규칙 (IN 값·조건 순서 무시 EM, 오답 분류, 다중 정답, 규칙 일치)
│       └── eval_build.py         2단계 평가셋 만들기 (→ data/stage2/eval/pairs.json 하나)
└── postprocessor/            생성된 SQL 의 테이블·컬럼 이름을 DB 스키마로 교정
    └── python/               Python 구현 (docs/postprocessor_guide.md, sqlglot 필요)
```

### 데이터와 결과

```
data/stage1/
├── db/          shop.db, holdout.json (미등장 값) — 2·3단계도 이 DB 를 쓴다
├── sql/         train.json, eval_indist.json, eval_holdout.json, gen_report.json
├── train/       pairs.json (학습 쌍. 이 폴더의 *.json 전부가 학습 데이터·BPE 코퍼스)
├── eval/        평가셋마다 폴더 하나: indist/, holdout/, holdout_qwen/ (pairs.json + sql.json)
└── tokenizer/   seed_tokens.json (seed 토큰, 사람이 편집), tokenizer.json (BPE 학습 결과)
data/stage2/
├── sql/         train.json, eval_indist.json, eval_struct.json, eval_holdout.json, gen_report.json
├── composer/    frames.json (문장 틀), lexicon.json (표현 사전), synonyms.json (틀 유사어)
├── handout/     다른 플랫폼 배포 묶음 안내문 틀 (handout_template.md, handout_fixed_template.md)
├── train/       composed.json (조합기 쌍), stage1_pairs.json (1단계 학습 쌍 사본), llm.json (LLM 학습 질문)
├── eval/        pairs.json + sql.json 하나에 평가셋 전부, 쌍마다 set (eval_build 가 만든다, 6절 "2단계 평가셋")
└── tokenizer/   seed_tokens.json, tokenizer.json
data/wordnet/    이름 교체용 영단어 목록 (단계 공통, 라이선스 포함)
models/stage1/, models/stage2/   단계별 배포 모델: model.pt, tokenizer.json, model_info.json
data_raw/stage<N>/  질문 생성 라운드·중간 산출물 (커밋 안 함). 2단계: 외부 말투 평가(ext_*), LLM 학습 질문(llm_*), 배포 묶음, 사람 작성 세트(human/)
runs/stage<N>/   학습 실행·평가·추론 결과 (커밋 안 함)
experiments/     단계 전환 전 실험 코드·데이터·체크포인트 (커밋 안 함, 결과는 보고서에만 기록)
```

### 테스트와 문서

```
tests/           pytest — src/ 와 같은 구조 (tests/models/, tests/data/stage2/, …), 마커는 4절
docs/
├── text-to-sql-llm-project-plan.md   전체 설계서 (단계 계획, 평가 지표, 전환 기준)
├── text-to-sql-stage1-report.md      1단계 진행 보고서
├── text-to-sql-stage2-report.md      2단계 진행 보고서
├── text-to-sql-label-rules.md        라벨 규칙 L1~L17
├── SETUP.md                          이 문서
├── CONVENTION.md                     협업·커밋 규칙
├── exporter-loader.md                GGUF 내보내기와 C 추론 엔진
└── postprocessor_guide.md            Python SQL 후처리
```

## 4. 클론 직후 할 일

DB, SQL, 질문 쌍, 토크나이저, 배포 모델이 모두 커밋되어 있어 학습 없이 바로 쓸 수 있다.

```
python -m pytest tests/ -v                                   # 전체 테스트 (unit + integration)
python -m src.run --input "what city does ashley live in?"   # 배포 모델로 질문 → SQL → DB 실행 결과
python -m src.run --evaluation                               # 배포 모델로 평가셋 전부 채점
python -m src.demo all                                       # 예시 하나로 파이프라인 각 단계의 입력 → 출력 보기 (9절)
python -m src.train --overfit 100                            # 과적합 테스트 (GPU, 약 40초)
python -m src.train                                          # 본 학습 + 평가 (GPU, 약 10분)
```

데이터를 처음부터 다시 만들거나 새로 추가할 때만 5~7절을 따른다.

### 테스트 분류와 선택 실행

모든 테스트에는 `unit` / `integration` 중 **정확히 하나**를 지정한다.

분류는 import 한 모듈 개수가 아니라 **assert 가 무엇을 검증하는지**로 정한다.

- `unit`: 한 함수·클래스의 동작 검증. 다른 모듈은 입력 준비용으로 쓰거나 대체(scripted forward 등)해도 된다.
  외부 프로세스·컴파일·저장소 데이터 파일(`data/…`) 없이 메모리와 `tmp_path` 만 쓴다. 모델 forward 검증은 unit 이다.
- `integration`: 구성요소 간 왕복·일치 검증(exporter↔loader, Python↔C, tokenize↔split, 배치 생성↔단일 생성),
  학습 동작(backward·optimizer step), C 엔진 빌드·subprocess·pytester, 저장소 데이터 파일 사용.

파일 전체가 같은 분류면 import 다음에 `pytestmark = pytest.mark.unit` 또는
`pytestmark = pytest.mark.integration`을 넣는다. 혼합 파일은 각 테스트 함수나 클래스에
`@pytest.mark.unit` / `@pytest.mark.integration`을 붙인다. 파일에 unit을 붙인 상태에서
함수에 integration을 붙이면 덮어쓰기가 아니라 이중 분류가 되므로 허용하지 않는다.

```sh
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration
python -m pytest tests/ --collect-only -m "not unit and not integration"
```

마지막 명령은 선택된 테스트가 0개여야 한다. 이때 pytest 종료 코드는 5
(`NO_TESTS_COLLECTED`)이므로 이 진단 명령 자체를 성공 코드 0을 요구하는 CI 작업으로 쓰지 않는다.

`tests/conftest.py`가 `-m` 필터 적용 **이전**에 누락·이중 분류를 검사하며,
`--strict-markers`가 미등록 마커·오타를 오류로 처리한다. 따라서 unit만 실행하는 CI에서도
수집 대상에 있는 마커 누락 테스트는 조용히 제외되지 않고 실패한다. 다만 명령에서 제외한
디렉터리는 수집하지 않으므로 전체 정책을 검사하려면 `tests/` 전체를 대상으로 실행한다.

수집 정책의 회귀 테스트는 `tests/test_marker_policy.py`에 있으며, 별도의 임시 pytest
프로세스를 실행하므로 integration으로 분류한다.

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

### 2단계 SQL

```
python -m src.data.stage2.sql_gen build      # data/stage2/sql/ 생성 + 검증 (약 2분, 결정성 확인용 재생성 포함)
python -m src.data.stage2.sql_gen verify     # 실행·정규형 왕복·규칙·미등장 리터럴·분할 비중복·1단계 겹침·구조 홀드아웃·상한·커버리지·결정성
python -m src.data.stage2.sql_gen summary    # 분포 표
```

주요 옵션: `--out`(기본 `data/stage2/sql`), `--db-dir`(읽을 DB 폴더, 기본 `data/stage1/db`), `--stage1-dir`(겹침 금지용
1단계 쌍 폴더, 기본 `data/stage1`), `--seed`, `--n-train` / `--n-eval-indist` / `--n-eval-struct`(SQL 수), `--cell-floor`.
`verify`와 `summary`는 `--out`의 `gen_report.json`에 적힌 설정·DB 위치를 읽는다. 문법·분할·결과는 [2단계 보고서](text-to-sql-stage2-report.md)에 있다.

### 2단계 질문 조합기

```
python -m src.data.stage2.compose sample --n 20                 # 학습 SQL 에서 질문 예시 출력 (--split eval: 평가 전용 표현 우선)
python -m src.data.stage2.compose check                         # 커버리지·값 복사·사전 충돌·유사어·eval 누출·결정성 (phrase 포함)
python -m src.data.stage2.compose dump --n 300 --out data_raw/stage2/compose_pilot.json   # 검수용 표본
python -m src.data.stage2.compose build                         # 학습 쌍 → data/stage2/train/composed.json + stage1_pairs.json
```

주요 옵션: `--sql`(SQL 레코드 파일, 기본 `data/stage2/sql/train.json`, `check`는 4개 분할), `--frames` / `--lexicon` /
`--synonyms`(기본 `data/stage2/composer/`, `--synonyms ''`면 유사어를 쓰지 않음), `--seed`, `--epoch`(같은 SQL의 다른 문장),
`--wildcard-ratio`(LIKE 표현 중 와일드카드 표기 `{v}*`·`%{v}%` 등을 고를 확률, 기본 0.1).
`build`: `--epochs`(SQL당 문장 수, 기본 3), `--out`(기본 `data/stage2/train/composed.json`), `--stage1-pairs`(같은 폴더에
`stage1_pairs.json`으로 복사할 1단계 학습 쌍, 기본 `data/stage1/train/pairs.json`, `''`면 복사 안 함). 학습 폴더의 `*.json`
전부가 학습 데이터·BPE 코퍼스이므로 두 파일이 함께 쓰인다. 1단계 쌍은 2단계 라벨 규칙 판정(`clean_ambiguous.drop_reason`)과 어긋나는 것을 빼고 복사하고 뺀 목록은 `data_raw/stage2/stage1_pairs_removed.json`에 둔다. `check`는 조합기 문장이 이 판정과 어긋나면(0.1% 초과) 실패한다.
`build`가 쓰는 쌍의 `sql`은 IN 값을 질문에 나온 순서로 쓴 표기(라벨 규칙 L8)이고, `sql_group`은 정규형 SQL이다.

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
3. 라벨 규칙(L2~L4, [라벨 규칙 문서](text-to-sql-label-rules.md))과 어긋나는 학습 쌍을 제거한다. 제거 목록은
   `data_raw/stage1/clean_ambiguous_removed.json`에 누적된다.
   ```
   python -m src.data.stage1.clean_ambiguous           # dry-run
   python -m src.data.stage1.clean_ambiguous --apply
   ```
4. 학습 코퍼스가 바뀌었으면 7절의 토크나이저를 다시 학습한다.

### 2단계 외부 말투 평가 질문 (`stage2.question_gen`)

계획서 3-4 "2단계 이후 평가 질문"의 방식 A(SQL → 질문). 평가 분할 SQL을 주고 SQL 하나마다 페르소나 5개(keyword, chat,
nonnative, verbose, business)의 질문을 하나씩 받는다. 프롬프트에는 스키마·값 표기·SQL의 뜻만 있고 조합기 틀·표현은 없다.

```
python -m src.data.stage2.question_gen make-prompts --out data_raw/stage2/ext_r1_indist --split indist --n 120 --seed 0
#   → <out>/prompts/batch_XX.txt (배치 하나 = 새 LLM 세션 하나), <out>/round_sql.json
#   플랫폼마다 같은 프롬프트를 새 세션에 붙여넣고 응답을 <out>/responses/<플랫폼>/batch_XX.json 으로 저장
python -m src.data.stage2.question_gen validate --out data_raw/stage2/ext_r1_indist
#   → <out>/pairs_<플랫폼>.json, <out>/validation_report.json (플랫폼 × 페르소나별 통과율·실패 사유·확인용 flag)
```

주요 옵션: `--split indist|struct|holdout`(SQL을 뽑을 평가 분할), `--n`(SQL 수, 모양별로 돌아가며 뽑음), `--batch-size`(기본 20),
`--exclude-dir`(이전 라운드 폴더들, 쉼표 구분, 그 SQL은 뽑지 않음). 검증: 값·LIKE 조각·LIMIT n 포함(소문자로 비교, 추론이
소문자로 넣으므로), 숫자 단어·밑줄 컬럼 이름·SQL 키워드 금지, IN 값 위치로 라벨(`sql` = 질문 순서, `sql_group` = 정규형) 결정,
같은 SQL 안 중복, 다른 SQL과 같은 질문(양쪽 제거), 라벨 규칙 판정(`src/data/stage2/clean_ambiguous.py`, 버린 수는 페르소나·사유별로 보고서에). 검증한 라운드는 아래 "2단계 평가셋"의 `eval_build ext`로 평가셋 폴더가 된다.

학습 질문은 같은 도구로 학습 SQL에서 만든다(평가와 다른 페르소나: plain, spoken, command, context, shorthand). 평가 SQL 라운드는 학습에 넣지 않는다.

```
python -m src.data.stage2.question_gen make-prompts --out data_raw/stage2/llm_r1 --split train --n 200 --seed 0
python -m src.data.stage2.question_gen validate --out data_raw/stage2/llm_r1
python -m src.data.stage2.question_gen to-train --out data_raw/stage2/llm_r1 --platform claude
#   → data/stage2/train/llm.json 에 넣는다. 쌍마다 round(라운드 폴더 이름)와 source(llm:<플랫폼>:<페르소나>)를 적고, 같은 라운드·플랫폼이
#     이미 있으면 바꾼다(질문은 소문자로, 기존 학습 쌍과 질문이 같고 SQL이 다른 쌍은 뺀다). 옮긴 뒤 BPE를 다시 만든다
```

다른 플랫폼(GPT, Gemini 등)을 쓰는 팀원에게는 프롬프트 대신 배포 묶음을 준다. 그 LLM이 안내문의 BRIEF(스키마·SQL 범위·라벨 규칙)를 읽고
프롬프트를 직접 쓰고, 파일럿으로 고친 뒤 배치마다 새 채팅에서 SQL당 질문 5개를 만든다(안내문 틀 `data/stage2/handout/handout_template.md`).
학습 SQL과 평가 SQL을 섞어 한 묶음으로 준다. 배포 파일의 id는 묶음 안 번호라 받는 쪽은 어느 것이 평가용인지 모른다.
배치를 플랫폼별로 나눠 맡겨도 된다(`handout/`을 플랫폼 폴더로 나누고 README의 배치 범위를 고친다). `validate`는 플랫폼마다 응답한 배치만 본다.

```
python -m src.data.stage2.question_gen make-handout --out data_raw/stage2/llm_r3 --n 800 --batch-size 25 --seed 2     --exclude-dir data_raw/stage2/llm_r1,data_raw/stage2/llm_r2     --eval indist:80,struct:80,holdout:40 --eval-from data_raw/stage2/ext_r1 --eval-prefix data_raw/stage2/ext_r2
#   → <out>/handout/ (보낼 폴더: README.md, sql/pilot.txt, sql/batch_XX.txt), <out>/handout_map.json (묶음 id → 라운드·원래 id)
#     라운드 폴더: <out>/ (학습), <out>/pilot/, data_raw/stage2/ext_r2_<분할>/ (평가, --eval-from 라운드와 같은 SQL에서 뽑음)
python -m src.data.stage2.question_gen collect-handout --out data_raw/stage2/llm_r3 --platform gpt --responses <돌려받은 폴더>   # 플랫폼마다 한 번씩
#   → 응답을 라운드별 responses/<플랫폼>/ 으로 나누고(```json 코드 블록은 벗긴다) 라운드마다 validate (파일럿은 프롬프트 버전별 <out>/pilot/responses/gpt_vN/)
#   그 뒤 학습분은 플랫폼마다 to-train --out data_raw/stage2/llm_r3 --platform <플랫폼>, 평가분은 eval_build ext (→ ext_r2_<분할>_<플랫폼>)
```

프롬프트를 정해 주는 묶음: `--prompt`로 이전 배포에서 받은 최종 프롬프트를 배치 파일마다 넣는다. 받는 쪽은 프롬프트를 쓰지 않고 배치 파일 하나를
새 세션에 통째로 넣기만 한다(파일럿 없음, 안내문 틀 `data/stage2/handout/handout_fixed_template.md`). `--n 0`이면 평가 SQL만 담는다.

```
python -m src.data.stage2.question_gen make-handout --out data_raw/stage2/gpt_r3 --n 0 --batch-size 25 --seed 3 \
    --prompt "data_raw/stage2/gpt result/prompt_final.txt" \
    --eval indist:80,struct:80,holdout:40 --eval-from data_raw/stage2/ext_r2 --eval-prefix data_raw/stage2/ext_r3
#   → <out>/handout/ (README.md, sql/batch_01~08.txt), <out>/handout_map.json, data_raw/stage2/ext_r3_<분할>/
python -m src.data.stage2.question_gen collect-handout --out data_raw/stage2/gpt_r3 --platform gpt --responses <돌려받은 폴더>
#   → eval_build ext (→ ext_r3_<분할>_gpt)
```

### 2단계 평가셋 (`stage2.eval_build`)

`data/stage2/eval/pairs.json` + `sql.json` 하나에 평가셋을 모두 담고 쌍마다 `set`(평가셋 이름)을 적는다. 불러올 때는 `set`으로 나뉘어
평가셋마다 따로 채점한다(`--eval-sets`에 이 이름을 쓴다). 명령마다 그 평가셋만 바꾸고 나머지는 둔다. `ext`를 라운드 전부로 돌리면
응답이 없어진 `ext_*` 평가셋은 지운다. 손으로 고치지 않고 다시 실행한다. 질문은 추론처럼 소문자로 저장하고,
쌍의 `sql`은 IN 값을 질문 순서로, `sql_group`은 정규형으로 쓴다(sql.json 은 `sql_group`으로 찾는다).

```
python -m src.data.stage2.eval_build              # 아래 셋 전부
python -m src.data.stage2.eval_build composer     # indist / struct / holdout / phrase: 평가 SQL 의 조합기 문장 (seed 1)
python -m src.data.stage2.eval_build ext          # ext_<라운드>_<플랫폼>: data_raw/stage2/ext_*/pairs_<플랫폼>.json
python -m src.data.stage2.eval_build human        # human: data_raw/stage2/human/ 에서 질문을 채운 문항
```

| 평가셋 | 내용 |
| --- | --- |
| `indist`, `struct`, `holdout` | 평가 분할 SQL마다 train 조합기 문장 하나(학습 문장은 seed 0, 평가는 seed 1). 처음 보는 SQL·구조·값 |
| `phrase` | indist SQL에서 eval 자리 하나(틀 / 조건 키 / 정렬 키)만 eval 표현으로 쓴 문장. 처음 보는 표현 |
| `ext_<라운드>_<플랫폼>` | 외부 말투 평가 라운드의 검증 통과 쌍(페르소나 포함). dev / 봉인 구분은 2단계 보고서 |
| `human` | 사람 작성 세트. 팀원이 `sql_list_<작성자>.json`(또는 .csv)의 `question`을 채우면 다시 실행해 추가한다. 빈 문항은 건너뛰고, 값 검사(`question_gen.check_question`)에 걸린 문항은 빼고 출력한다. 라벨 규칙 판정은 빼지 않는다(규칙 일치 EM이 뺀다) |

`ext`와 `human`은 커밋하지 않는 `data_raw/`를 읽으므로, 원본 라운드가 있는 곳에서만 다시 만들 수 있다.

## 7. 토크나이저·학습·배포

### 토크나이저

```
python -m src.tokenizer.bpe    # seed_tokens.json + 학습 쌍 → data/stage1/tokenizer/tokenizer.json, round-trip·미등장 이름 토큰 수 출력
python -m src.demo dataset     # 전체 쌍 토큰화 길이 통계, collate_fn/loss_mask 확인 (9절)
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
- 검증셋은 학습 쌍에서 SQL 단위로(`sql_group`이 있으면 그것으로) 5%를 떼어 내고, 체크포인트 선택·조기 종료는 이것으로만 한다. 평가셋은 학습이
  끝난 뒤 한 번만 측정한다.
- 주요 옵션: `--epochs`(기본 25), `--patience`(기본 10), `--lr`, `--batch-size`, `--ffn-dim`(config를 고치지
  않고 FFN 크기만 바꿔 실험, 기본 round(8 × dim / 3)), `--layers` / `--dim` / `--heads`(모델 크기 실험, 기본은 단계 프리셋),
  `--seed`, `--run-name`, `--save-every N`(N epoch마다 `epoch<N>.pt`도 저장, 학습량별 평가 추이 확인용). 입력 바꾸기: `--train-files`(여러 개), `--tokenizer`,
  `--eval-dir`, `--db`. 최종 평가할 평가셋 고르기: `--eval-sets`(기본 전부). 실행 폴더 직접 지정: `--out`
  (기본 `runs/stage<N>/<run-name>/`).
- 모델 구조와 기본 경로는 `src/config.py`의 `STAGE` 값으로 정해진다. 기본은 2이고, 1단계 명령은 환경 변수로 하나만 바꿔 실행한다
  (`run.py`는 `--stage 1`). `src/data/stage1/`·`src/data/stage2/`의 데이터 스크립트는 `STAGE`와 상관없이 자기 단계 경로를 쓴다:
  ```
  python -m src.data.stage2.compose build                                       # 2단계 학습 쌍
  python -m src.tokenizer.bpe                                                   # data/stage2/tokenizer/tokenizer.json (vocab 2048)
  python -m src.train --epochs 15 --llm-repeat 3 --run-name <이름>              # → runs/stage2/<이름>/
  TEXT_TO_SQL_STAGE=1 python -m src.train                                       # 1단계 (5M, data/stage1/)
  ```
  PowerShell에서는 `$env:TEXT_TO_SQL_STAGE=1` 을 먼저 실행한다. 2·3단계는 1단계 DB·holdout(`data/stage1/db/`)을 함께 쓴다
  (`config.DB_STAGE`). 테스트는 1단계 배포 모델·데이터 기준이라 `tests/conftest.py`가 `TEXT_TO_SQL_STAGE=1`로 돈다.
- 모델 최대 길이(`SEQ_LEN`)를 넘는 학습 쌍은 학습에서 빼고 개수를 출력한다.
- 이름 교체: 기본 학습이 매 epoch 이름 조건 쌍의 40%를 가짜 이름으로 바꾼다(`--name-swap-ratio`로 비율 변경, 0이면 끔).
  가짜 이름에서 영단어를 걸러 내는 데 `data/wordnet/english_words.txt`(WordNet 3.0 표제어, 라이선스는 같은 폴더)를
  쓴다. 다른 목록은 `--wordlist`로 지정한다(목록이 다르면 가짜 이름과 결과가 달라진다).
  2단계는 `src/data/stage2/name_swap.py`가 IN 의 이름 전부와 LIKE 조각까지 바꾼다. 고정 파일
  명령은 1단계용만 있다.
- LLM 학습 질문(2단계, `source`가 `llm:*`): 매 epoch 이름 교체 전에 질문의 일상 단어를 조합기 유사어 사전(`synonyms.json`)의
  대체어로 바꾼다(`src/data/stage2/synonym_swap.py`, 확률 기본 0.5 = 사전의 `prob`, `--llm-synonym-prob 0`이면 끔). 값 검사나 라벨 규칙에
  걸리는 문장은 원래 문장을 쓴다. `--llm-repeat N`은 검증셋을 뗀 뒤 LLM 쌍을 N번 넣는다(사본마다 epoch마다 다르게 바뀐다).
  ```
  python -m src.train --epochs 15 --llm-repeat 3 --run-name <이름>
  ```
- 고정 파일 방식: 파일을 만들어 `--train-files`로 학습한다. 이미 이름을 바꾼 파일이면 학습 중 이름 교체는 자동으로 꺼진다.
  ```
  python -m src.data.stage1.name_swap --ratio 0.4   # → data_raw/stage1/name_swap_40/train_pairs.json (커밋 안 함)
  TEXT_TO_SQL_STAGE=1 python -m src.train --train-files data_raw/stage1/name_swap_40/train_pairs.json --run-name name_swap_40
  ```

### 배포 모델 만들기

```
python -m src.release --ckpt runs/stage2/stage2_llm3/best.pt                         # → models/stage2/
TEXT_TO_SQL_STAGE=1 python -m src.release --ckpt runs/stage1/ffn683_swap40/best.pt   # → models/stage1/
```

- `model.pt`(가중치·모델 설정·단계만, 학습 인자와 로컬 경로 없음), `tokenizer.json`(이 모델이 학습에 쓴
  토크나이저), `model_info.json`(출처 체크포인트, 학습 설정 요약, 검증 EM, 내보낸 파일로 다시 측정한 평가 결과)을 쓴다.
- 단계마다 최종 모델 하나만 커밋한다. 다시 내보내면 덮어쓴다. `--no-eval`로 평가를 생략할 수 있다.

### GGUF 내보내기 (C 추론 엔진용)

```
python -m src.exporter --checkpoint models/stage1/model.pt --output runs/stage1/model.gguf
```

- 배포 모델·학습 체크포인트를 그대로 받고, 토크나이저는 체크포인트 옆의 `tokenizer.json`을 쓴다(`--tokenizer`로 변경).
- 내보내기 세부와 계약 버전 변경 절차는 [exporter-loader.md](exporter-loader.md), C 엔진 빌드·실행과 파일 형식(CONTRACT.md)은
  C 엔진 저장소 [capstone-sql-cli](https://github.com/polring/capstone-sql-cli)에 있다. 결과 `.gguf`는 커밋하지 않는다.
- 배포용 GGUF는 `stage<N>-contract<V>` 태그를 push하면 GitHub Actions가 만들어 release에 올린다.

## 8. 추론·평가 (`run.py`)

추론과 평가는 `python -m src.run` 하나로 한다. 모델 불러오기·생성은 `src/inference/`, 채점·평가셋 관리는
`src/evaluation/`에 있고 학습·배포와 같은 함수를 쓴다.
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
| 평가셋 이름·폴더, 정답 있는 JSON | `--evaluation` 또는 생략 | 채점: EM, 다중 정답 EM, 규칙 일치 EM, 실행 정확도, 신뢰구간, 계층별·집계 기준별, 오답 분류 | 화면 요약 + `eval_<이름>.json` |
| 평가셋 이름·폴더, 정답 있는 JSON | `--test` | 채점 없이 배치 추론 | 예측 JSON (정답 함께 기록) |

- `--input`이 존재하는 폴더면 평가셋 폴더, `.json`이면 파일, 평가셋 이름과 같으면 평가셋, 그 밖에는 질문 문자열로 본다.
- JSON 형식: `[{"question": ..., "sql": ...}]` 또는 `["질문", ...]`. 같은 폴더에 `sql.json`이 있으면 계층별 집계에 쓴다.
- 예측 JSON의 각 항목: `question`, `pred`(생성 SQL, 실패하면 null), `stage_format`(그 단계 SQL 형식인지), 입력에 있었다면 `sql`.
- 질문은 학습 데이터처럼 소문자로 바꿔 넣는다. 생성은 greedy이며 제약 디코더는 없다.
- 채점 규칙은 단계마다 `src/data/stage<N>/sql_rules.py`다. 1단계 EM은 문자열 일치이고 WHERE 컬럼별로 나눠 본다. 2단계 EM은
  IN 값 순서와 AND·OR 조건 순서를 무시하고(라벨 규칙 L8, 리터럴 표기는 정규형과 같아야 함), 분할·작성자·페르소나·구조 조합·
  미등장 값 모양·eval 표현 자리·테이블·조건(수·연결어·IN·LIKE)·정렬·LIMIT별로 나눠 본다(값이 둘 이상인 기준만 출력). 규칙 일치 EM은
  2단계 라벨 규칙 판정(`stage2/clean_ambiguous.drop_reason`)이 어긋났다고 보는 문항을 뺀다.

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
python -m src.run --model runs/stage2/<실행>/best.pt --evaluation   # 직접 학습한 체크포인트 평가
python -m src.run --list-sets                                  # 평가셋 목록
python -m src.run --stage 1 --build-sql indist --sql data/stage1/sql/eval_indist.json   # 1단계 평가셋 sql.json 생성 (--sql 여러 개 가능)
python -m src.run --stage 1 --evaluation                       # 1단계 배포 모델, 1단계 평가셋·채점 규칙
```

- 결과 파일은 기본으로 `runs/` 아래(커밋 안 함)에 저장된다. `models/`는 커밋 폴더라 결과를 쓰지 않는다.
- `--stage 2`처럼 아직 모델이 없는 단계를 주면 안내 후 종료한다.

## 9. 단계별 입출력 보기 (`demo.py`)

파이프라인의 각 단계가 무엇을 받아 무엇을 내놓는지 직접 보여 준다. 확인·시연용이라 파일을 쓰지 않는다.
질문을 SQL로 바꿔 쓰거나 평가하는 것은 `run.py`(8절)다.

### 하위 명령

`--text`(질문)와 `--sql`을 주면 그 예시를 보여 주고, 주지 않으면 오른쪽 기준으로 예시를 고른다.

| 하위 명령 | 입력 → 출력 | `--text` / `--sql`을 주면 | 주지 않으면 |
| --- | --- | --- | --- |
| `db` | DB 테이블 구조와 행 | `--sql`이 읽는 테이블과 WHERE 조건에 맞는 행 | 테이블마다 앞쪽 `--n`행, 미등장 값 |
| `sql` | (테이블, SELECT, WHERE 컬럼, 값) → SQL → 실행 결과 | `--sql`을 SQL 파일(학습·분포 내·미등장 값)에서 찾은 생성 정보와 분할 | `--sql-file`에서 무작위 `--n`개 |
| `questions` | SQL → LLM이 쓴 질문들 → 검증 결과 | `--sql`의 질문 전부 (학습·평가 쌍에서 찾고 출처 표시) | 질문이 있는 SQL 무작위 `--n`개 |
| `clean` | (질문, SQL) → 라벨 규칙 판정 | 그 쌍의 SELECT 모호 여부와 통과/제거 | 학습 쌍 전체의 제거 대상 수, SELECT 모호 문항 무작위 `--n`개 |
| `name-swap` | (질문, SQL) → 가짜 이름으로 바꾼 쌍 | 그 쌍을 바꾼 결과 | 학습 쌍 전체에 `--ratio`로 적용한 통계와 예시 `--n`개 |
| `tokenizer` | 텍스트 → 사전 분할(seed 표시) → 토큰 ID·종류 → 디코딩 | 질문(과 SQL)을 토큰화 | 기본 질문 `what city does ashley live in?` |
| `dataset` | (질문, SQL) → `<bos> 질문 <sep> SQL <eos>` ID, 위치별 input → target·loss 여부, 전체 길이 통계 | 그 쌍 | 기본 질문과 `SELECT city FROM customers WHERE name = 'ashley'` |
| `embedding` | 토큰 ID → 임베딩, RoPE 회전 | 그 질문 | 기본 질문 |
| `block` | 임베딩 → 디코더 블록 전체. `--layer` 블록은 RMSNorm → q·k·v → RoPE → attention(많이 보는 토큰) → residual → SwiGLU를 직접 계산하고 블록 `forward`와 일치 확인 | 그 질문 | 기본 질문, 블록 0 |
| `model` | 마지막 블록 출력 → 최종 RMSNorm → 출력층 → logits, `<sep>` 다음 토큰 확률, 정답의 teacher forcing loss | 그 질문(과 정답 SQL의 loss) | 기본 질문, loss 생략 |
| `generate` | greedy 생성을 한 토큰씩 (스텝마다 고른 토큰·확률, 2위 후보) → 생성 SQL → DB 실행 | 그 질문(과 정답 비교) | 기본 질문 |
| `score` | (질문, 정답 SQL, 생성 SQL) → 평가셋 문항 하나의 채점 기록: EM, 다중 정답, 실행 정확도, 라벨 불일치, 틀린 부분, 집계 위치 | 그 쌍. 생성 SQL은 `--pred`, 없으면 모델이 생성 | 기본 질문과 기본 SQL |
| `all` | 아래 흐름 전체 | `--sql`과 `--text`로 진행 | 기본 SQL과, 그 SQL에 LLM이 쓴 첫 질문 |

### `all`: 예시 하나를 끝까지 따라가기

각 단계의 결과가 다음 단계의 입력이 된다.

```
--sql ─→ db (이 SQL이 읽는 행)
     ─→ sql (생성 정보·분할) ─→ questions (이 SQL의 LLM 질문들)
                                   │ 첫 질문 (또는 --text)
                                   ▼
        clean (라벨 판정) · name-swap (학습용 이름 교체) · dataset (학습 입력 ID·loss_mask)
                                   │
        tokenizer ─→ <bos> 질문 <sep> ─→ embedding ─→ 임베딩 ─→ block ─→ 마지막 블록 출력
                                   ─→ model (출력층 → logits, 한 번에 계산한 결과와 일치 확인, 정답 loss)
                                   ─→ generate (생성 SQL → DB 실행) ─→ score (정답과 채점)
```

### 예시

```
python -m src.demo all                                                          # 기본 SQL 하나를 끝까지
python -m src.demo all --sql "SELECT * FROM customers WHERE name = 'harriet'"   # 미등장 개체 SQL
python -m src.demo questions --sql "SELECT stock FROM items WHERE item_name = 'adapter'"
python -m src.demo tokenizer --text "how many mouse units are left in stock?"
python -m src.demo block --text "is there a customer named harriet?" --layer 3
python -m src.demo generate --text "what is the price of the laptop?" --model runs/stage1/ffn683_swap40/best.pt
python -m src.demo name-swap --n 5                                              # 학습 쌍 전체에 적용한 통계
python -m src.demo score --text "do we have a customer named harriet" --sql "SELECT * FROM customers WHERE name = 'harriet'" --pred "SELECT name FROM customers WHERE name = 'harriet'"
```

- 공통 인자: `--n`(예시 수), `--seed`(무작위 예시 고르기). 읽는 파일은 하위 명령마다 `--db`, `--holdout`,
  `--sql-file`, `--pairs`, `--train-files`, `--eval-dir`, `--wordlist`, `--tokenizer`, `--model`, `--pred`로 바꾼다(생략하면
  config 기본값). 하위 명령별 인자는 `python -m src.demo <하위 명령> --help`, 예시 기준 요약은 `python -m src.demo --help`.
- 모델을 쓰는 단계는 배포 모델(`models/stage<N>/model.pt`)과 그 토크나이저를 CPU에서 쓴다.
- 출력의 `※` 줄은 그 단계가 무엇을 하는지와 나온 값을 어떻게 읽는지 설명한다. 값에 따라 설명이 달라진다(예: `generate`는
  2위 후보 확률이 1% 이상인 토큰을 "헷갈림"으로 표시, `score`는 틀린 부분과 다중 정답·실행 정확도의 의미).
  `--quiet`를 주면 설명 없이 값만 출력한다(오답을 여러 개 비교할 때).

## 10. 재현 확인 기준

1단계 명령은 `TEXT_TO_SQL_STAGE=1`(`run.py`는 `--stage 1`)로 실행한다.

| 명령 / 파일 | 기대 결과 |
| --- | --- |
| `python -m src.tokenizer.bpe` | 병합 726/726, round-trip 실패 0건, 미등장 이름 평균 분할 토큰 ≈ 3.52, `tokenizer.json`이 커밋된 파일과 같음 |
| `python -m src.demo dataset` | 시퀀스 길이 22~64(평균 38.3), 컨텍스트 128 초과 0건 |
| `python -m src.demo block` | 마지막 줄 "블록 forward 결과와 같음: True" |
| `python -m pytest tests/ -v` | 전체 통과 |
| `python -m src.train --overfit 100` | EM 100% 도달 |
| `python -m src.train` (seed 0) | 파라미터 4,985,600, 최고 검증 EM 24 epoch 0.990, 분포 내 EM ≈ 0.967, 미등장 값 EM ≈ 0.907 |
| `python -m src.train --name-swap-ratio 0` (seed 0) | 이름 교체 없는 기준: 분포 내 EM ≈ 0.955, 미등장 값 EM ≈ 0.712 |
| `python -m src.run --stage 1 --evaluation` | `models/stage1/model_info.json`의 평가 결과와 같음 (분포 내 0.967, 미등장 값 0.907) |
| `data/stage1/sql/gen_report.json` | 구조별 할당량이 층화 샘플링 방침(계획서 5-3)대로, ID 조건 비중 30% 이하 |
| `python -m src.data.stage2.sql_gen verify` | "통과" (결정성 포함, 다시 만든 결과가 커밋된 `data/stage2/sql/`과 같음) |

## 11. 커밋 대상

- **커밋**: `src/`, `tests/`, `docs/`, `CLAUDE.md`, `README.md`, `requirements.txt`, `data/stage<N>/`
  (DB, SQL, 학습 쌍, 평가셋, seed 토큰, 토크나이저), `data/wordnet/`, `models/stage<N>/`(단계별 최종 배포 모델 하나).
- **커밋 안 함** (`.gitignore`): `.venv/`, `data_raw/`(LLM 라운드 원본·중간 산출물),
  `runs/`(학습·평가·추론 결과), `experiments/`(단계 전환 전 실험 코드·데이터·체크포인트).
