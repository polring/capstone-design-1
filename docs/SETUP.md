# 환경 설정 가이드

저장소를 클론한 뒤 실행 환경을 만들고 각 단계를 재현하는 방법이다. 설계 이유는
[계획서](text-to-sql-llm-project-plan.md), 현재 진행 상황과 수치는 [3단계 보고서](text-to-sql-stage3-report.md)에
있고, 함수 단위 입출력은 각 소스 파일의 docstring에 있다.

**모든 명령은 저장소 루트에서 실행한다.** 경로가 현재 디렉터리 기준 상대 경로(`data/...`, `models/...`)라서
다른 위치에서 실행하면 파일을 찾지 못하거나 엉뚱한 곳에 만든다.

## 1. 요구 사항

- Python 3.12 또는 3.13 (CI 3.12, 개발 환경 3.13)
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

GPU 학습 환경에서는 마지막 줄이 `2.14.0+cu130 True NVIDIA GeForce ...`처럼 출력되어야 한다.
CPU 추론 환경에서는 GPU 확인 명령 대신 `python -c "import torch; print(torch.__version__)"`로 설치를 확인한다.

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
│   ├── wordnet_normalizer.py 사전 기반 선택형 질문 전처리
│   └── gguf_model.py         GGUF 내보내기용 체크포인트 계약 모델 (exporter.py 가 사용)
├── evaluation/
│   ├── scoring.py            채점·집계: EM, 다중 정답 EM, 규칙 일치 EM, 실행 정확도, 신뢰구간
│   └── eval_sets.py          평가셋 읽기(폴더 방식 / 합친 방식), 평가 결과 위치
├── data/
│   ├── dataset.py            <bos> 질문 <sep> SQL <eos> 토큰화, loss_mask, collate_fn
│   ├── db_gen.py             SQLite DB·미등장 값 생성 (스키마가 같은 단계 공통)
│   ├── json_io.py            JSON·gzip JSON 공통 입출력
│   ├── splitting.py          SQL·대비 묶음 단위 학습/검증 분할과 누수 검사
│   ├── integrity.py          데이터·배포 파일의 해시 기록
│   ├── preprocessing/        WordNet·도메인 전처리 사전 생성
│   ├── stage1/
│   │   ├── sql_gen.py            SQL 생성 (학습 / 분포 내 / 미등장 값)
│   │   ├── question_gen.py       LLM 질문 생성 (사람이 세션에 전달) · 검증
│   │   ├── question_gen_auto.py  로컬 Ollama 모델 자동 호출
│   │   ├── clean_ambiguous.py    라벨 규칙(L2~L4) 위반 학습 쌍 제거
│   │   ├── name_swap.py          이름 교체 (가짜 이름 생성)
│   │   └── sql_rules.py          1단계 채점 규칙 (SQL 분해, 오답 분류, 다중 정답, 규칙 일치)
│   ├── stage2/
│   │   ├── schema.py             컬럼 종류별 허용 연산자·정렬 가능 여부·리터럴 표기
│   │   ├── sql_ast.py            SQL 구조(Query / Cond), 정규형 렌더링, 역파싱, 문법 규칙 검사
│   │   ├── sql_gen.py            SQL 생성 (학습 / 분포 내 / 구조 홀드아웃 / 미등장 값)
│   │   ├── compose.py            질문 조합기: SQL 구조 → 질문 (문장 틀 × 표현 사전 × 유사어), 학습 쌍 생성
│   │   ├── clean_ambiguous.py    2단계 라벨 규칙 판정 (L1·L2·L3·L4·L14)
│   │   ├── question_gen.py       LLM 질문: 외부 말투 평가 / 학습 질문, 응답 검증, 학습 폴더로 옮기기
│   │   ├── name_swap.py          2단계 이름 교체 (IN 의 모든 이름, LIKE 조각)
│   │   ├── sql_rules.py          2단계 채점 규칙 (IN 값·조건 순서 무시 EM, 오답 분류, 다중 정답, 규칙 일치)
│   │   └── eval_build.py         2단계 평가셋 만들기 (→ data/stage2/eval/pairs.json 하나)
│   └── stage3/                날짜·집계·그룹·서브쿼리용 SQL/질문 생성·검사·증강
│       ├── schema.py / sql_ast.py   단계별 스키마와 SQL 구조, LIKE 해석
│       ├── sql_gen.py / compose.py SQL 생성과 질문 조합·유의어 증강
│       ├── clean_ambiguous.py / sql_rules.py  라벨 검사와 채점 규칙
│       ├── name_swap.py / name_pools.py / like_sampling.py  이름·검색 조각 증강
│       └── eval_build.py / llm_batches.py / contrast.py     평가·LLM·대비 데이터 생성
└── postprocessor/            생성된 SQL 의 테이블·컬럼 이름을 DB 스키마로 교정
    └── python/               Python 후처리 참고 구현 (sqlglot 필요, C 구현은 별도 CLI 저장소)
```

### 데이터와 결과

```
data/stage1/
├── db/          shop.db, holdout.json (미등장 값)
├── sql/         train.json, eval_indist.json, eval_holdout.json, gen_report.json
├── train/       pairs.json (학습 쌍. 이 폴더의 *.json 전부가 학습 데이터·BPE 코퍼스)
├── eval/        평가셋마다 폴더 하나: indist/, holdout/, holdout_qwen/ (pairs.json + sql.json)
└── tokenizer/   seed_tokens.json (seed 토큰, 사람이 편집), tokenizer.json (BPE 학습 결과)
data/stage2/
├── db/          1단계와 내용이 같은 DB·holdout 사본 (2단계 실행 기본 경로)
├── sql/         train.json, eval_indist.json, eval_struct.json, eval_holdout.json, gen_report.json
├── composer/    frames.json (문장 틀), lexicon.json (표현 사전), synonyms.json (틀 유사어)
├── train/       composed.json (조합기 쌍), stage1_pairs.json (1단계 학습 쌍 사본), llm.json (LLM 학습 질문)
├── eval/        pairs.json + sql.json 하나에 평가셋 전부, 쌍마다 set (eval_build 가 만든다, 6절 "2단계 평가셋")
└── tokenizer/   seed_tokens.json, tokenizer.json
data/stage3/
├── db/          날짜·두 단어 이름을 포함한 shop.db와 holdout.json
├── sql/         일반·집계 계열을 통합한 SQL 원본과 생성 보고서
├── composer/    일반·고급 질문 틀과 표현·유의어 사전
├── train/       pairs.json.gz (통합 학습 쌍)
├── validation/  pairs.json (고정 검증 쌍)
├── eval/        14개 평가셋 통합 pairs.json, sql.json, external_sources.json
├── tokenizer/   seed_tokens.json, tokenizer.json
├── preprocessing/normalizer.json  입력 전처리 사전
└── dataset_info.json, name_pools.json  데이터 명세와 이름 풀

data/common/wordnet/    이름 교체용 영단어 목록 77,503개 (단계 공통, 라이선스는 licenses/WORDNET_LICENSE.txt)
models/stage1/, models/stage2/, models/stage3/   단계별 배포 모델: model.pt, tokenizer.json, model_info.json
models/stage3/                 normalizer.json과 배포용 WORDNET_LICENSE.txt도 포함
licenses/                      외부 데이터 라이선스 원본
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
├── text-to-sql-stage3-report.md      현재 3단계 결과와 코드 구성
├── text-to-sql-label-rules.md        라벨 규칙 L1~L29와 정책 반영 기록
├── SETUP.md                          이 문서
├── CONVENTION.md                     협업·커밋 규칙
├── exporter-loader.md                GGUF 내보내기와 C 입력 전처리 연결
└── repo-split-plan.md                C 구동기 저장소 분리의 계획·완료 기록
```

## 4. 클론 직후 할 일

DB, SQL, 질문 쌍, 토크나이저, 배포 모델이 모두 커밋되어 있어 학습 없이 바로 쓸 수 있다.

```
python -m pytest tests/ -v                                   # 전체 테스트 (unit + integration)
python -m src.run --input "how many orders are there?"       # 기본 3단계 모델로 질문 → SQL → DB 실행 결과
python -m src.run --evaluation                               # 배포 모델로 평가셋 전부 채점
python -m src.demo generate                                  # 기본 3단계 SQL 생성 과정 보기 (전체 시연은 9절)
python -m src.train --overfit 100                            # 기본 3단계 과적합 테스트 (GPU)
python -m src.train                                          # 기본 3단계 27.4M 프리셋 본 학습 + 평가 (GPU)
```

데이터를 처음부터 다시 만들거나 새로 추가할 때만 5~7절을 따른다.
배포 모델과 같은 15.7M 구조·고정 데이터로 학습하려면 아래의 "3단계 실행" 예시를 따른다.

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
python -m src.data.db_gen build --stage 1    # data/stage1/db/shop.db, holdout.json
python -m src.data.db_gen test --stage 1     # 분포·결정성·price 등호·홀드아웃 분리 검사
python -m src.data.stage1.sql_gen build --cap-nonid 9999 --cap-id 9999   # data/stage1/sql/train.json, eval_indist.json
python -m src.data.stage1.sql_gen verify     # 완료 조건 검사
python -m src.data.stage1.sql_gen test       # 결정성·홀드아웃 리터럴 미포함·학습/평가 분리 검사
python -m src.data.stage1.sql_gen holdout    # data/stage1/sql/eval_holdout.json (미등장 값 평가 SQL)
```

위 명령은 1단계 재생성 예시다. `db_gen`은 `--stage` 생략 시 현재 기본인 3단계 경로·설정을 사용한다.
주요 옵션: `--seed`(기본 0), `--out`(출력 폴더, 위 예시에서는 `data/stage1/db` / `data/stage1/sql`), `sql_gen`의
`--db-dir`(읽을 DB 폴더), `db_gen`의 `--preset small|large|aggregate`(1·2단계 기본 `large`, 3단계 기본 `aggregate`)와 `--name-style single|two_word`
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
     python -m src.run --stage 1 --build-sql <이름> --sql data/stage1/sql/eval_indist.json   # 그 평가셋의 SQL 파일
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

### 3단계 실행

학습·추론·BPE·DB 생성·배포는 공통 실행 코드에서 `--stage 3`을 받는다.
단계별 파일을 복사해 실행하지 않는다. `--stage 1`, `--stage 2`도 같은 인자 형식이다.
공유된 DB·토크나이저·고정 분할을 그대로 사용하려면 아래 학습 명령부터 실행한다.

기존 모델을 바로 사용하려면 `models/stage3/`의 15.7M 모델을 불러온다.
같은 폴더에 토크나이저와 최종 전처리 사전·WordNet 라이선스를 포함했다.
전처리는 `--wordnet-map`을 지정할 때 적용하며, 모델 정보에 원문·전처리 평가를 구분해 기록한다.

```sh
python -m src.run --stage 3 --wordnet-map models/stage3/normalizer.json --input "record count of transactions"
```

```sh
python -m src.train --stage 3 --layers 8 --dim 384 --heads 6 --ffn-dim 1024 --epochs 15 --steps-per-epoch 10845 --run-name example
python -m src.run --stage 3 --model runs/stage3/example/best.pt --evaluation
```

위 학습 명령은 배포 모델과 같은 15.7M 구조다. 크기 인자를 생략한 `--stage 3` 학습 기본값은 계획서의 27M 구조다.

DB·SQL·질문·평가·BPE 생성 도구는 입력을 새로 만들 때 사용한다.
생성 결과를 비교하려면 `--out` 또는 `--eval-dir`를 `data_raw/stage3/` 아래로 지정해 공유본과 별도로 저장한다.

BPE의 기본 입력은 `data/stage3/train/pairs.json.gz` 하나다.
GitHub 파일 크기 제한에 맞춰 압축했으며 학습기·BPE·전처리 사전 생성기가 직접 읽는다. 별도 압축 해제는 필요 없다.
학습은 이 파일의 321,041쌍과 `data/stage3/validation/pairs.json`의 고정 검증 16,216쌍을 자동으로 사용한다.
`--val-ratio`로 다시 나누지 않으며, SQL·대비 묶음이 양쪽에 걸치면 실행을 거부한다.
다른 입력은 `--train-files <학습.json> --val-files <검증.json>`으로 지정한다.
명시하는 학습 입력도 `.json`과 `.json.gz`를 모두 지원한다.
고정 검증을 지정하지 않은 별도 학습 입력은 기존처럼 SQL·대비 묶음 단위로 분리한다.
파일별 개수·해시·출처는 `data/stage3/dataset_info.json`에 기록한다.

전체 단계의 데이터·배포 모델과 공용 데이터의 무결성 기준은 `data/integrity.json`에서 관리한다.
공통 테스트 `python -m pytest tests/data/test_dataset_bundle.py -q`는 파일 추가·삭제·내용 변경과
gzip 압축 해제 후의 해시를 검사한다. 정당한 데이터·모델 변경을 검토한 뒤
`python -m src.data.integrity`로 기준을 갱신하고 변경 파일과 함께 커밋한다.
테스트 실패를 없애기 위한 기준 갱신은 하지 않는다.
줄바꿈도 해시 대상이다. `.gitattributes`의 `-text`는 기존 작업 파일의 줄바꿈을 소급 변경하지
않으므로, 기준 생성 전 작업 파일과 Git 저장 바이트가 같은지 `git ls-files --eol data models`로
확인한다. 1·2단계의 기존 JSON은 Git에 저장된 LF 바이트를 기준으로 유지한다.
평가는 `data/stage3/eval/pairs.json` 하나에 조합기 17,676쌍과 외부 LLM 2,000쌍을 합친 총 19,676쌍이다.
`set` 필드로 14개 평가셋을 구분해 각각 채점한다. 외부 평가는 기존 GPT 표현 1,000쌍과 사람 말투를 참고한 LLM 표현 1,000쌍이다.
SQL 메타데이터는 `eval/sql.json`, 외부 질문 생성 기록은 `eval/external_sources.json`에 둔다.
SQL 원본도 일반 조회·집계 계열을 합쳐 `sql/train.json` 100,000개, `eval_indist.json` 3,000개,
`eval_struct.json` 2,978개, `eval_holdout.json` 8,707개로 둔다. `query_kind`가 있으면 집계 계열이다.
문항 틀·표현은 `composer/`, DB와 미등장 값은 `db/`, 현재 BPE는 `tokenizer/`,
최종 전처리 사전은 `preprocessing/normalizer.json`이다.
생성 프롬프트와 문형 출처 기록은 커밋하지 않는 `data_raw/stage3/composer/`의
`llm_brief.txt`·`template_provenance.json`에 둔다. 새 환경에서 LLM 질문을 생성하려면
안내문을 준비하고 `--brief`로 지정한다. 출처 기록 검사는 로컬 파일이 있을 때만 실행한다.
이름 교체 원본은 `name_pools.json`이며, DB에 없는 교체 후보는 학습 시 계산한다.
후보 목록 내보내기(`python -m src.data.stage3.name_pools`)는 `data_raw/stage3/name_pools/replacement_pools.json`에 저장한다.
과거 버전은 로컬 `data_raw/stage3/layout_backup_20261018/original_stage3.zip`에 보관하며 GitHub 공유 대상에서 제외한다.
현재 공유본의 학습·평가에는 이 백업과 원본 LLM 응답, 로컬 실험 스크립트가 필요하지 않다.
3단계 학습 기본값은 이름 교체 40%, LLM 쌍(source `llm:*`) 3번(`--llm-repeat 3`),
LLM 질문의 epoch별 유의어 교체(확률 = `data/stage3/composer/synonyms.json`의 `prob` 0.5, `--llm-synonym-prob 0`이면 끔).
특정 라운드의 반복 비중은 `--llm-round-repeat ROUND=N`으로 따로 지정한다.
통합본의 기본 반복 비중은 `dataset_info.json`에서 읽으며, numbers 수정본(`llm_numbers_r10_600`)은 1회다.
`--llm-round-repeat llm_numbers_r10_600=2`처럼 명시하면 그 설정이 우선한다.
반복은 검증 분리 뒤에 적용하므로 고유 문항과 검증셋은 유지된다.
지정한 라운드가 학습에 없으면 오타·설정 누락으로 보고 실행을 거부한다.
반복 설정을 바꾼 체크포인트 이어 받기는 허용하지 않으며 새 학습으로 비교한다.
데이터량이 다른 실행의 학습량을 맞추려면 `--steps-per-epoch N`으로 epoch당 학습 배치 수를
제한한다. 예: `--epochs 15 --steps-per-epoch 10845`는 총 162,675 step이며 lr 스케줄도 이 길이를 쓴다.
각 epoch는 섞인 학습 배치 중 앞 N개를 사용한다. 사용할 수 있는 배치 수보다 큰 값은 거부한다.
조합기 문장의 epoch별 유의어 교체는 기본으로 끄고 `--composer-synonym-prob 0.5`로 켠다(조합기 30만 쌍 기준 epoch당 약 90초 추가).
LIKE 값 교체는 `--like-sampling balanced`로 토큰 빈도를 고려한 후보 선택을 켤 수 있다(기본은 기존 `markov`).
검증 분리·LLM 반복·길이 필터 후의 학습 SQL만으로 토큰 빈도를 계산하고, 같은 글자 수의 흔한 단일 토큰 /
희귀 단일 토큰(학습 SQL 출현 100회 이하) / 여러 토큰 조각 중 사용 가능한 후보군을 같은 확률로 선택한다.
후보는 평가 문장과 무관한 소문자 ASCII 조합이며 현재 데이터의 1~3글자를 지원한다. 더 긴 조각은 교체를 건너뛴다.
질문·SQL 값은 함께 바뀌고 LIKE 방향·글자 수·범위·분할 메타데이터를 유지한다. 교체 확률은 기존 `--name-swap-ratio`를 쓴다.
설정·후보군 크기는 실행 폴더의 `like_sampling.json`, 교체 통계는 학습 로그의 `like_bucket_*`에 기록한다.
이 옵션은 3단계 본 학습에서만 지원한다. 방식이 다른 체크포인트의 `--resume`은 거부하며 방식 비교는 `--init`으로 시작한다.

추가 의미 대비 보강은 기존 학습용 조합기로 생성한다. 평가 표현·라벨 규칙·토크나이저를 바꾸지 않는다.
기본 생성량은 출력 컬럼 2,000·비교 연산자 2,000·정렬/개수 2,000·LIKE 방향 1,500쌍이다.

```powershell
python -m src.data.stage3.contrast build --out data_raw/stage3/my_contrast
```

`build`는 지정한 폴더에 검증된 질문·SQL 및 대비 묶음 명세를 저장한다.
`--source-sql`, `--corpus-files`, `--round`, `--*-groups`로 원본 SQL·중복 확인 대상 질문·라운드명·생성량을 지정한다.
기존 r9 실험의 고정 라운드 반영 작업은 Git에서 제외되는 로컬 실험 코드로 분리했다.

```powershell
python experiments/stage3/prepare_contrast_r9.py build
python experiments/stage3/prepare_contrast_r9.py prepare
```

이전 r9 재현 명령은 정리 전 압축본을 별도 작업 공간에 복원한 후에 사용한다.
이 로컬 스크립트의 `build`는 `data_raw/stage3/contrast_r9_1500/`에 생성하고,
`prepare`는 검수된 numbers/intent/no-invention 8,000쌍과 새 7,500쌍을 학습 입력 폴더에 추가하고,
SQL·대비 묶음을 함께 나눈 `data/stage3/splits/contrast_r9_seed0/{train,validation}.json`을 만든다.
현재 통합본은 이 과거 분할을 사용하지 않는다. 원본은 `data_raw/stage3/layout_backup_20261018/original_stage3.zip`에 있다.
동일 산출물이 있으면 덮어쓰지 않고 종료한다. 다른 실험은 `--out`, `--split-dir`로 경로를 지정한다.
입력/출력 해시와 라운드별 분할 수는 `split_report.json`에 남는다.

`experiments/stage3/run_contrast_experiment.py`는 이 분할로 5.8M·15 epoch·균형 LIKE 교체를 실행하고,
기본 평가 8개와 GPT 외부 말투 평가(기존 5.8M도 같은 외부 셋으로 평가)를 이어 실행한다.
사전 확인은 `--dry-run`이며, 진행 상태는 실행 폴더 `status.json`, 결과 비교는
`comparison.json`·`comparison.csv`·`weakness_comparison.json`이다. 재분할했으므로 이전 검증 EM과 직접 비교하지 않는다.
3단계 전용 코드와 공통 코드의 구분은 [3단계 보고서](text-to-sql-stage3-report.md#code-layout)에 정리했다.

## 7. 토크나이저·학습·배포

### 토크나이저

```
python -m src.tokenizer.bpe    # 기본 3단계 seed + 학습 쌍 → data/stage3/tokenizer/tokenizer.json, round-trip·미등장 이름 토큰 수 출력
python -m src.demo dataset     # 전체 쌍 토큰화 길이 통계, collate_fn/loss_mask 확인 (9절)
```

- 코퍼스와 seed 목록이 같으면 `tokenizer.json`은 항상 같은 파일로 나온다.
- seed 토큰(SQL 키워드, 테이블·컬럼 이름, 범주형 값)은 `seed_tokens.json`에서 관리한다. 토큰 ID가 목록 순서로
  정해지므로 순서를 바꾸거나 추가하면 토크나이저와 모델을 다시 학습해야 한다.
- 입력·출력 바꾸기: `--train-files`, `--seeds`, `--vocab-size`, `--out`.

### 학습

```
python -m src.train --overfit 100          # 학습 예시 100개로 EM 100% 도달 확인
python -m src.train                        # 기본 3단계 프리셋 본 학습 → runs/stage3/<실행>/, 끝나면 평가셋 전부 측정
python -m src.train --name-swap-ratio 0    # 기본 3단계에서 이름 교체 없는 기준 실행
.venv\Scripts\tensorboard --logdir runs    # 학습 곡선
```

- 산출물: `runs/stage<N>/<이름>/`에 `best.pt`(검증 EM 최고 시점 체크포인트, 모델 설정 포함), `tokenizer.json`
  (학습에 쓴 토크나이저 사본), `metrics.json`(epoch별 기록, 평가 요약), `eval_<평가셋>.json`(요약 + 틀린 사례 전체),
  `tb/`(TensorBoard 로그), `last.pt`(마지막 epoch 가중치 + 옵티마이저·난수 상태, `--resume`용).
- 검증셋은 기본 3단계 통합본에서는 고정 파일을 사용한다. 고정 검증이 없는 입력은 SQL·대비 묶음 단위로
  5%를 분리한다. 체크포인트 선택·조기 종료는 검증셋만 쓰고 평가셋은 학습 후 측정한다.
- 주요 옵션: `--epochs`(1·2단계 기본 25, 3단계 15, overfit 2000), `--patience`(기본 10), `--lr`, `--batch-size`, `--ffn-dim`(config를 고치지
  않고 FFN 크기만 바꿔 실험, 기본 round(8 × dim / 3)), `--layers` / `--dim` / `--heads`(모델 크기 실험, 기본은 단계 프리셋),
  `--seed`, `--run-name`, `--save-every N`(N epoch마다 `epoch<N>.pt`도 저장, 학습량별 평가 추이 확인용). 입력 바꾸기: `--train-files`(여러 개), `--tokenizer`,
  `--val-files`, `--eval-dir`, `--db`. 최종 평가할 평가셋 고르기: `--eval-sets`(기본 전부). 실행 폴더 직접 지정: `--out`
  (기본 `runs/stage<N>/<run-name>/`).
- 중단된 학습 이어 하기: 처음과 같은 인자에 `--resume runs/stage<N>/<run>/last.pt`를 붙인다. 다음 epoch부터 같은 lr 스케줄로 계속하고
  옵티마이저·증강 난수까지 복원한다. `best.pt`·`epoch<N>.pt`도 받지만 가중치만 있어 AdamW 상태는 새로 시작한다. `--init`은 lr 스케줄을
  처음부터 다시 시작하는 별개 기능이다.
- 모델 구조와 기본 경로는 `src/config.py`의 `STAGE` 값으로 정해지며 기본은 3이다.
  학습·추론·BPE·DB 생성은 `--stage N`으로 단계를 선택한다. `src/data/stage<N>/`의 데이터 스크립트는
  `STAGE`와 상관없이 자기 단계 경로를 쓴다:
  ```
  python -m src.data.stage2.compose build                                       # 2단계 학습 쌍
  python -m src.tokenizer.bpe --stage 2                                         # data/stage2/tokenizer/tokenizer.json (vocab 2048)
  python -m src.train --stage 2 --epochs 15 --llm-repeat 3 --run-name <이름>     # → runs/stage2/<이름>/
  python -m src.train --stage 1                                                # 1단계 (5M, data/stage1/)
  ```
  실행 DB·holdout 기본 경로는 `data/stage<N>/db/`다. 2단계 DB는 1단계와 내용이 같은 사본이며,
  2단계 SQL 생성기의 입력 기본값은 계속 `data/stage1/db/`다. 3단계 DB에는 날짜와 두 단어 이름이 포함된다.
  `demo.py`는 `--stage` 대신 환경 변수를 따른다. PowerShell에서 이전 단계를 보려면
  `$env:TEXT_TO_SQL_STAGE='1'` 후 실행하고, 끝나면 `Remove-Item Env:TEXT_TO_SQL_STAGE`로 기본값을 복원한다.
  테스트는 `tests/conftest.py`에서 기본 환경 변수를 1로 설정하고, 2·3단계 테스트는 해당 단계 경로를 직접 사용한다.
- 모델 최대 길이(`SEQ_LEN`)를 넘는 학습 쌍은 학습에서 빼고 개수를 출력한다.
- 이름 교체: 기본 학습이 매 epoch 이름 조건 쌍의 40%를 가짜 이름으로 바꾼다(`--name-swap-ratio`로 비율 변경, 0이면 끔).
  1·2단계는 마르코프 이름 생성기를 쓰며, 가짜 이름에서 영단어를 걸러 내는 데 `data/common/wordnet/english_words.txt`(WordNet 3.0 표제어, 라이선스는 `licenses/WORDNET_LICENSE.txt`)를
  쓴다. 다른 목록은 `--wordlist`로 지정한다(목록이 다르면 가짜 이름과 결과가 달라진다).
  목록은 WordNet 3.0의 noun/verb/adj/adv 표제어 중 알파벳 단어를 소문자로 정렬한 것이다.
  저장소 내 라이선스 원본은 `licenses/WORDNET_LICENSE.txt`로 관리하고, 목록·전처리 사전을 별도로 배포할 때는 함께 복사한다.
  전처리 사전 생성기는 출력 폴더마다 라이선스를 중복 생성하지 않으며, `models/stage3/`에는 배포용 사본을 유지한다.
  2단계는 `src/data/stage2/name_swap.py`가 IN 의 이름 전부와 LIKE 조각까지 바꾼다. 고정 파일
  명령은 1단계용만 있다.
  3단계는 `name_pools.json`으로 만든 DB와 겹치지 않는 두 단어 후보를 사용하고, 상품 범주를 유지한다.
  LIKE 조각은 마르코프 또는 `--like-sampling balanced`의 토큰 빈도별 후보로 교체한다.
- LLM 학습 질문(2·3단계, `source`가 `llm:*`): 매 epoch 이름 교체 전에 질문의 일상 단어를 조합기 유사어 사전(`synonyms.json`)의
  대체어로 바꾼다(각 단계 `src/data/stage<N>/compose.py`의 `SynonymSwapper`, 확률 기본 0.5 = 사전의 `prob`, `--llm-synonym-prob 0`이면 끔). 값 검사나 라벨 규칙에
  걸리는 문장은 원래 문장을 쓴다. `--llm-repeat N`은 검증셋을 뗀 뒤 LLM 쌍을 N번 넣는다(사본마다 epoch마다 다르게 바뀐다).
  `--composer-synonym-prob P`(기본 0 = 끔)를 주면 조합기 문장(`source`가 `composer*`)에도 같은 교체를 확률 P로 적용한다.
  조합기 문장은 만들 때 틀에 한 번 적용됐으므로, epoch마다 다른 대체어를 보여 주는 용도다. 1단계 쌍은 바꾸지 않는다.
  ```
  python -m src.train --epochs 15 --llm-repeat 3 --run-name <이름>
  ```
- 고정 파일 방식: 파일을 만들어 `--train-files`로 학습한다. 이미 이름을 바꾼 파일이면 학습 중 이름 교체는 자동으로 꺼진다.
  ```
  python -m src.data.stage1.name_swap --ratio 0.4   # → data_raw/stage1/name_swap_40/train_pairs.json (커밋 안 함)
  python -m src.train --stage 1 --train-files data_raw/stage1/name_swap_40/train_pairs.json --run-name name_swap_40
  ```

### 배포 모델 만들기

```
python -m src.release --ckpt runs/stage2/stage2_llm3/best.pt                         # → models/stage2/
python -m src.release --stage 1 --ckpt runs/stage1/ffn683_swap40/best.pt  # 1단계 배포 모델 갱신 예시
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

### 선택적 단어·구문 입력 전처리 (3단계)

기존 모델·토크나이저·학습/평가 데이터는 유지하고, `--wordnet-map`을 지정할 때만
학습에 없던 단어를 학습 단어로 치환한다. 구문 프로필은 이미 등장한 단어의 조합과 약어도 정규화한다.
원문과 정답은 채점에 그대로 사용하며
평가 출력의 `preprocessing`에 변환 질문·치환 단어·경로 거리와 유사도를 남긴다.
옵션을 생략하면 기존 추론이다. 옵션을 켜고 출력 경로를 생략하면 별도 `wordnet_<variant>/`에 저장한다.

```powershell
python -m src.data.preprocessing.wordnet_map --train-files data/stage3/train/pairs.json.gz --wordnet data_raw/stage3/wordnet_expression_pilot/wordnet.zip --db data/stage3/db/shop.db --out data_raw/stage3/preprocessing/wordnet
python -m src.run --stage 3 --model runs/stage3/size_5m_numbers_r10_once_fixed_s0/best.pt --input "show merchandise names" --wordnet-map data_raw/stage3/preprocessing/wordnet/synonyms.json --no-exec
python -m src.run --stage 3 --model runs/stage3/size_5m_numbers_r10_once_fixed_s0/best.pt --eval-sets advanced_phrase --wordnet-map data_raw/stage3/preprocessing/wordnet/path.json --output runs/stage3/wordnet_cli_check
```

생성기는 기존 치환표 폴더 덮어쓰기를 거부하므로 다시 생성할 때는 새 출력 폴더를 지정한다.
원본 WordNet은 공식 NLTK 데이터 저장소의 `packages/corpora/wordnet.zip`이며
`wordnet/data.*`, `wordnet/*.exc`, `wordnet/LICENSE`가 들어 있는 배포본이다.
실행 의존성은 표준 라이브러리와 생성된 치환표뿐이고 NLTK 설치는 필요 없다.
`synonyms.json`은 같은 synset, `path.json`은 명사·동사의 상/하위어 연결 최대 2단계까지 허용한다.
모든 뜻 중 최소 거리, 학습 등장 빈도, 철자 순으로 정하며 문맥에 맞는 뜻을 추론하지 않는다.
숫자·날짜·따옴표 문자열·DB에 등록된 이름/범주값을 보호한다.
`wordnet_v3`는 starts/begins/ends with/in, contains, prefix/suffix, 와일드카드 표기 뒤의
명시적인 검색 조각도 보호하고, `name ending x`, `finishing in x`, `x anywhere in`도 처리한다.
v1·v2와 WordNet 치환표는 같고 값 보호만 보완한 것이다.
따옴표나 DB에 없는 새로운 이름 및 인식하지 못하는 문체의 LIKE 조각은 완전히 보호하지 못할 수 있다.
전처리 점수는 외부 사전을 이용한 시스템 점수로 보고하고, 원문 모델 점수와 따로 비교한다.

도메인 명사·집계 구문·약어 전처리는 `domain_v4` 프로필을 사용한다.
`src.data.preprocessing.domain_wordnet_map`은 상품·고객·주문의 WordNet 뜻을 기준으로 명사 후보를 제한하고,
같은 거리로 여러 테이블에 연결되는 단어는 유지한다. 주문은 상업적 의미의 뜻풀이가 있을 때만 후보가 된다.
도메인 의미 앵커와 구문 규칙은 사람이 정의한 외부 지식이며 문맥 해석 모델은 사용하지 않는다.
원문에서 가장 긴 구문을 우선해 한 번 치환하고 이름·검색값 보호는 기존과 같다.

| 프로필 | 처리 |
|---|---|
| `domain.json` | 도메인 명사 치환만 |
| `phrases.json` | 집계 구문 12개 규칙, qty/avg/asc/desc 포함 |
| `combined.json` | 도메인 명사 + 구문·약어 |
| `combined_symbols.json` | 위 처리 + 비교 기호 6종을 자연어로 확장 |

```powershell
python -m src.data.preprocessing.domain_wordnet_map --train-files data/stage3/train/pairs.json.gz --wordnet data_raw/stage3/wordnet_expression_pilot/wordnet.zip --db data/stage3/db/shop.db --out data_raw/stage3/preprocessing/domain
python -m src.run --stage 3 --model runs/stage3/size_5m_numbers_r10_once_fixed_s0/best.pt --wordnet-map data_raw/stage3/preprocessing/domain/combined.json --input "record count of transactions" --no-exec
python -m src.run --stage 3 --model runs/stage3/size_5m_numbers_r10_once_fixed_s0/best.pt --wordnet-map data/stage3/preprocessing/normalizer.json --eval-sets advanced_phrase --output runs/stage3/domain_cli_check
```

실제 CPU 확인: `record count of transactions` → `row count of orders` → `SELECT COUNT(*) FROM orders`.
`avg qty of orders with order number = 11303` → `average quantity of orders with order number equal to 11303`
→ `SELECT AVG(quantity) FROM orders WHERE order_id = 11303`.
빌더는 평가셋을 읽지 않는다. 구문 규칙은 이미 관측한 평가의 약점을 참고한 개발 개선이므로
같은 평가에서의 개선을 봉인 평가나 모델 자체의 어휘 일반화로 주장하지 않는다.

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
python -m src.run --input "how many orders are there?"         # 기본 3단계 단건
python -m src.run --evaluation                                 # 평가셋 전부
python -m src.run --eval-sets indist holdout                   # 고른 평가셋만
python -m src.run --input holdout --output results/holdout.json
python -m src.run --input questions.json --test                # 정답 없는 질문 목록 배치 추론
python -m src.run --stage 2 --model runs/stage2/<실행>/best.pt --evaluation   # 직접 학습한 2단계 체크포인트 평가
python -m src.run --list-sets                                  # 평가셋 목록
python -m src.run --stage 1 --build-sql indist --sql data/stage1/sql/eval_indist.json   # 1단계 평가셋 sql.json 생성 (--sql 여러 개 가능)
python -m src.run --stage 1 --evaluation                       # 1단계 배포 모델, 1단계 평가셋·채점 규칙
```

- 결과 파일은 기본으로 `runs/` 아래(커밋 안 함)에 저장된다. `models/`는 커밋 폴더라 결과를 쓰지 않는다.
- 1·2·3단계 배포 모델을 제공한다. 선택한 모델 파일이 없으면 오류를 안내하고 종료한다.

## 9. 단계별 입출력 보기 (`demo.py`)

이 절의 전체 시연 예시는 1단계 기준이다. 현재 `demo all`·`demo model`은 3단계에서
할당 어휘 용량 4,096과 실제 어휘 수 3,534의 차이를 처리하지 못해 오류가 난다.
일반 추론(`run.py`)과 `demo generate`는 기본 3단계에서 실행할 수 있다.
전체 시연을 실행할 때는 PowerShell에서 환경 변수를 먼저 설정한다:

```powershell
$env:TEXT_TO_SQL_STAGE='1'
python -m src.demo all
```

이 절의 시연이 끝나면 `Remove-Item Env:TEXT_TO_SQL_STAGE`로 기본 단계 3을 복원한다.

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

아래 수치 표는 1단계 재현 기준이다. `train`·`run`·BPE 명령은 `--stage 1`로 실행하거나,
PowerShell에서 먼저 `$env:TEXT_TO_SQL_STAGE='1'`을 설정한다. `demo`도 이 환경 변수를 따른다.
검증이 끝나면 `Remove-Item Env:TEXT_TO_SQL_STAGE`로 기본 단계 3을 복원한다.

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

- **커밋**: `src/`, `tests/`, `docs/`, `AGENTS.md`, `README.md`, `requirements.txt`, `data/stage<N>/`
  (DB, SQL, 학습 쌍, 평가셋, seed 토큰, 토크나이저), `data/common/wordnet/`, `data/integrity.json`,
  `licenses/`, `models/stage<N>/`(단계별 최종 배포 모델 하나).
- **커밋 안 함** (`.gitignore`): `.venv/`, `data_raw/`(LLM 라운드 원본·중간 산출물),
  `runs/`(학습·평가·추론 결과), `experiments/`(단계 전환 전 실험 코드·데이터·체크포인트).
