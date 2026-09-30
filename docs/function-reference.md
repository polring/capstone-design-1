# 함수 레퍼런스

`src/data/generator/`의 `db_gen.py`, `sql_gen.py`, `question_gen.py`, `question_gen_auto.py`와
`src/tokenizer/bpe.py`, `src/data/dataset.py` 6개 파일의 함수를 정리한 것. 모든 명령은 저장소 루트에서 실행한다. 파일마다 먼저 **실행 순서**(어떤 명령을 돌리면 함수들이 어떤
순서로 호출되는지)를 보여주고, 그다음 **함수별 입력/처리/출력**을 정리한다. "왜 이렇게 설계했는가"는
여기 없다 — 그건 [CLAUDE.md](CLAUDE.md)와
[text-to-sql-llm-project-plan.md](text-to-sql-llm-project-plan.md)에 있다.

## 전체 파이프라인 순서 (파일 간)

```
1. src/data/generator/db_gen.py        build   →  data/shop.db, data/holdout.json
2. src/data/generator/sql_gen.py       build   →  data/sql_train.json, data/sql_eval_indist.json
3. src/data/generator/question_gen.py  make-prompts → (사람이 LLM에 붙여넣음) → validate
                         ── 13+6라운드 반복 ──→ data/pilot*/, data/eval_indist*/ 의 pilot_train_pairs.json
   또는 src/data/generator/question_gen_auto.py  → 로컬 Ollama 모델을 직접 호출해 3을 사람 개입 없이 한 번에 끝냄
                         (question_gen.py의 샘플링/프롬프트/검증 함수를 그대로 재사용, 모델 호출과
                         JSON 복구만 새로 추가)
4. src/tokenizer/bpe.py  (python -m src.tokenizer.bpe)  →  bpe_merges.json
5. src/data/dataset.py    1~4의 산출물을 읽어 토큰화·배치 생성 (아직 직접 실행 스크립트가 아니라
                          train.py가 가져다 쓸 모듈)
```

---

## `src/data/generator/db_gen.py` — DB 생성기

### 실행 순서

| 명령 | 호출 순서 |
|---|---|
| `build` | `main` → `build`(내부에서 `Config.validate` → `build_customers` → `build_items` → `build_orders` → `build_holdout` → SQLite INSERT + `holdout.json` 저장) → `verify` → `canonical_hash` → `sql_space` |
| `verify` | `main` → `verify` |
| `test` | `main` → `build` → `verify` → `test_determinism`(내부에서 `build` 3회 + `canonical_hash` 3회) → `test_price_equality` → `test_holdout` |
| `summary` | `main` → `summary` |
| `space` | `main` → `sql_space` |

### 함수

- **`Config.item_name_capacity(self) -> dict`**
  - 입력: 없음 (`self`만)
  - 처리: `name_style`에 따라 카테고리별 상품명 풀(`CATEGORY_NOUNS`) 크기를 계산
  - 출력: `{카테고리: 만들 수 있는 상품명 개수}`

- **`Config.validate(self) -> None`**
  - 입력: `Config`의 필드값들(행 수, ID 대역, 최소 개수 등)
  - 처리: 서로 모순되는 설정(예: 카테고리 최소치 합 > 상품 수)이 있는지 검사
  - 출력: 없음 — 문제 있으면 `ValueError`

- **`balanced_column(rng, values, total, minimum, caps=None) -> list`**
  - 입력: 값 후보 리스트, 총 개수, 값별 최소/최대(cap) 횟수
  - 처리: 각 값을 최소 `minimum`번 이상 채운 뒤 남는 자리를 cap 이내로 무작위 배분하고 섞음
  - 출력: 길이 `total`의 셔플된 리스트 (city/membership/category/status/quantity가 굶지 않게 하는 핵심 함수)

- **`sample_ids(rng, lo, hi, n) -> list[int]`**
  - 입력: 범위 `[lo, hi]`, 개수 `n`
  - 처리: 그 범위에서 중복 없이 `n`개 추출 후 정렬
  - 출력: ID 리스트 (customer_id/item_id/order_id 생성에 사용)

- **`build_customers(rng, cfg) -> list[tuple]`**
  - 입력: `Config`
  - 처리: `sample_ids`로 ID, `GIVEN_NAMES`에서 이름, `balanced_column`으로 city/membership 생성
  - 출력: `(customer_id, name, city, membership)` 튜플 300개

- **`build_items(rng, cfg) -> list[tuple]`**
  - 입력: `Config`
  - 처리: 카테고리 먼저 배분 → 그 카테고리 명사 풀에서 상품명 추출 → price/stock 무작위 생성, 품절 상품 강제 삽입
  - 출력: `(item_id, item_name, category, price, stock)` 튜플 200개

- **`build_orders(rng, cfg, customer_ids, item_ids) -> list[tuple]`**
  - 입력: 고객/상품 ID 리스트
  - 처리: 모든 고객·상품이 최소 1건씩 걸리도록 채운 뒤 나머지를 무작위로 채움
  - 출력: `(order_id, customer_id, item_id, quantity, status)` 튜플 500개

- **`build_holdout(rng, cfg, customers, items, orders) -> dict`**
  - 입력: 생성된 세 테이블
  - 처리: 일부 행/ID를 "이름+ID 둘 다 미등장" / "ID만 미등장" / "주문 ID 미등장" 세 계층으로 무작위 분리
  - 출력: `holdout` 딕셔너리 (`data/holdout.json`으로 저장됨)

- **`load_bans(ho) -> dict[str, dict[str, set]]`**
  - 입력: `holdout.json`을 읽은 딕셔너리(`build_holdout`의 출력)
  - 처리: `{테이블: {컬럼: 금지 리터럴}}` 형태로 펼침
  - 출력: 금지 리터럴 맵 — `sql_gen.py`가 SQL 만들 때 이 값들을 걸러냄

- **`build(seed, out_dir, cfg=None) -> tuple[str, str]`**
  - 입력: 시드, 출력 폴더
  - 처리: 위 함수들을 순서대로 호출해 세 테이블을 만들고 SQLite에 INSERT, holdout 저장
  - 출력: `(db_path, holdout_path)`

- **`verify(db_path, cfg=None) -> list[str]`**
  - 입력: DB 파일 경로
  - 처리: 행 수·ID 대역·중복·값 분포·price/stock 범위·FK 무결성·소문자 규칙을 SQL로 검사
  - 출력: 위반 사항 문자열 리스트 (비어있으면 전부 통과)

- **`canonical_hash(db_path) -> str`**
  - 입력: DB 파일 경로
  - 처리: 세 테이블 행 내용을 정렬·직렬화 후 SHA-256 (파일 바이트가 아니라 행 내용 기준)
  - 출력: 해시 문자열 — 같은 시드면 항상 동일 (결정성 판정용)

- **`sql_space(db_path, ho_path, detail=True) -> tuple[int, int]`**
  - 입력: DB, holdout 경로
  - 처리: 1단계 문법으로 만들 수 있는 SQL을 (테이블×WHERE컬럼) 단위로 셈
  - 출력: `(전체 고유 SQL 수, 그중 ID 조건 수)`

- **`test_determinism(out_dir, cfg) -> list[str]`**
  - 입력: 출력 폴더, `Config`
  - 처리: 같은 시드로 2번, 다른 시드로 1번 `build` 호출 후 해시 비교
  - 출력: "같은 시드인데 다름"/"다른 시드인데 같음" 문제가 있으면 그 문자열 리스트

- **`test_price_equality(db_path) -> list[str]`**
  - 입력: DB 경로
  - 처리: 모든 상품에 대해 `WHERE item_id=X AND price={저장값}` 쿼리를 실제 실행
  - 출력: `round(x,2)` 값이 SQL `=` 비교와 안 맞는 건이 있으면 리스트로

- **`test_holdout(ho_path) -> list[str]`**
  - 입력: `holdout.json`
  - 처리: "이름+ID 미등장"과 "ID만 미등장" 집합이 겹치지 않는지, 비어있지 않은지 확인
  - 출력: 문제 있으면 문자열 리스트

- **`summary(db_path) -> None`**
  - 입력: DB 경로
  - 처리: 테이블별 행 수·분포·price/stock 범위·샘플 행을 콘솔에 출력
  - 출력: 없음 (출력 자체가 목적)

- **`main() -> int`**
  - 입력: 커맨드라인 인자 (`build`/`verify`/`test`/`summary`/`space`, `--preset`/`--seed`/`--out`/`--name-style`)
  - 처리: 해당하는 함수 호출
  - 출력: 종료 코드 (0=성공, 1=검증 실패)

---

## `src/data/generator/sql_gen.py` — SQL 생성기

### 실행 순서

| 명령 | 호출 순서 |
|---|---|
| `build` | `main` → `build`(내부에서 `db_gen.load_bans` → `generate`[`decide_quota_pools` → `apply_id_ratio_cap`(반복 호출: `_combo_counts`) → `_select_variants`/`build_sql`/`format_literal` per 리터럴] → JSON 3종 저장) → **`verify`**(db_gen.py의 `build`처럼, main이 저장 직후 바로 재검증까지 호출) |
| `verify` | `main` → `verify`(내부에서 `generate` 재호출 후 저장된 report와 비교) |
| `test` | `main` → `test`(내부에서 `generate` 2회 + `canonical_hash` 2회) |
| `summary` | `main` → `summary` |

### 함수

- **`available_literals(con, bans, table, col) -> list`**
  - 입력: DB 커넥션, 금지 리터럴, 테이블/컬럼명
  - 처리: 그 컬럼의 서로 다른 값 중 holdout에 없는 것만 필터링
  - 출력: 정렬된 사용 가능 리터럴 리스트

- **`existing_ids_by_domain(con) -> dict[str, set[int]]`**
  - 입력: DB 커넥션
  - 처리: 세 ID 컬럼 각각의 실존 값을 조회
  - 출력: `{"customer_id": {...}, "item_id": {...}, "order_id": {...}}`

- **`_select_variants(table) -> list[str]`**
  - 입력: 테이블명
  - 처리: 컬럼 목록에 `"*"` 추가
  - 출력: SELECT 절에 올 수 있는 선택지 리스트

- **`decide_quota_pools(con, bans, cfg) -> dict[tuple[str,str], dict]`**
  - 입력: DB, holdout, 설정
  - 처리: (테이블, WHERE컬럼)별로 "실존 값 풀"과 "존재하지 않는 ID 후보 풀"을 시드 고정 셔플로 생성
  - 출력: 조합별 풀 딕셔너리 (ID 컬럼만 nonexistent_pool이 채워짐)

- **`_combo_counts(pools, cfg, cap_id) -> dict[tuple[str,str], dict]`**
  - 입력: 풀 딕셔너리, ID 상한(`cap_id`)
  - 처리: 그 `cap_id`를 적용했을 때 조합별 실제 사용 리터럴/SQL 개수 계산
  - 출력: 조합별 카운트 (카테고리형은 전량, ID는 `cap_id` 제한)

- **`apply_id_ratio_cap(pools, cfg) -> tuple[dict, int]`**
  - 입력: 풀, 설정
  - 처리: `cap_id`를 1씩 줄이며 `_combo_counts`를 반복, ID 조건 비율이 `id_ratio_max`(30%) 이하가 되는 최소 `cap_id`를 탐색
  - 출력: `(최종 counts, 사용된 cap_id)`

- **`format_literal(value, coltype) -> str`**
  - 입력: 값, 타입("text"/"float"/기타)
  - 처리: 타입별 SQL 리터럴 형식으로 변환 (text→따옴표, float→소수점 2자리)
  - 출력: 리터럴 문자열

- **`build_sql(table, select_col, where_col, value) -> str`**
  - 입력: 테이블, SELECT컬럼, WHERE컬럼, 값
  - 처리: `format_literal` 호출 후 SQL 문자열 조립
  - 출력: `SELECT ... FROM ... WHERE ... = ...` 완성 SQL 한 줄

- **`generate(con, bans, cfg) -> tuple[list[dict], list[dict], list[dict]]`**
  - 입력: DB, holdout, 설정
  - 처리: `decide_quota_pools`/`apply_id_ratio_cap`로 정한 할당량만큼 `build_sql`로 SQL을 만들고 SQLite에서 실행해 결과 행 수까지 기록, train/eval로 분할
  - 출력: `(학습용 SQL 리스트, 평가용 SQL 리스트, 조합별 할당량 표)`

- **`_paths(out_dir) -> dict[str, str]`**
  - 입력: 출력 폴더
  - 처리: 산출 파일 5종의 경로를 조립
  - 출력: 경로 딕셔너리

- **`build(out_dir, cfg) -> dict`**
  - 입력: 출력 폴더, 설정
  - 처리: `generate()` 호출 후 `sql_train.json`/`sql_eval_indist.json`/`sql_gen_report.json` 저장
  - 출력: report 딕셔너리

- **`verify(out_dir) -> list[str]`**
  - 입력: 이미 만들어진 `sql_gen_report.json`
  - 처리: 같은 설정으로 `generate()` 재실행 → (1)전체 SQL 실행 성공+행수 일치 (2)할당량 표 재현 (3)ID 비율 30% 이하, 3종 재검사
  - 출력: 위반 사항 리스트

- **`canonical_hash(rows) -> str`**
  - 입력: SQL 딕셔너리 리스트
  - 처리: SQL 문자열만 정렬·해시
  - 출력: 해시 문자열 (결정성 테스트용)

- **`test(out_dir, cfg) -> list[tuple[str, list[str]]]`**
  - 입력: 출력 폴더, 설정
  - 처리: `generate()` 2회로 결정성을, holdout 미포함/train·eval 비중복/ID 비율 상한을 검사
  - 출력: `(검사명, 실패 목록)` 쌍의 리스트

- **`summary(out_dir) -> None`**
  - 입력: `sql_gen_report.json`
  - 처리: (테이블, WHERE컬럼)별 실존/미존재/변형/SQL/train/eval 개수를 표로 조립
  - 출력: 없음 (콘솔 출력)

- **`main() -> int`**
  - 입력: 커맨드라인 인자 (`build`/`verify`/`test`/`summary`, `--cap-nonid`/`--cap-id`/`--id-ratio-max` 등)
  - 처리: 해당 함수 호출
  - 출력: 종료 코드

---

## `src/data/generator/question_gen.py` — LLM 질문 생성 파이프라인

### 실행 순서

| 명령 | 호출 순서 |
|---|---|
| `make-prompts` | `main` → `make_prompts` → `_make_prompts_impl`(내부에서 `stratified_sample` → 배치마다 `build_batch_prompt`[내부에서 `format_literal_for_prompt`] → `prompts/*.txt`, `pilot_sql.json` 저장) |
| `make-prompts-small` (2026-09-23 추가, 약한/로컬 모델용) | `main` → `make_prompts_small_model` → `_make_prompts_impl`(`build_batch_prompt` 대신 `build_small_model_batch_prompt`[내부에서 `small_model_rules_blurb`] 사용, 배치 크기 기본값이 더 작음) |
| (사람이 LLM에 프롬프트를 붙여넣고 `responses/*.json` 저장 — 또는 `question_gen_auto.py`가 이 단계를 자동화) | — |
| `validate` | `main` → `validate`(내부에서 `_load_pilot_sql` → `_load_responses` → id별로 `find_intra_group_duplicates` → 질문마다 `check_question`[내부에서 `literal_in_question`→`_isolated_number_match`] → 통과분에 `soft_flags` → `pilot_train_pairs.json`, `validation_report.json` 저장) |
| `template-check` | `main` → `template_check`(내부에서 `_load_pilot_sql` → `_load_responses` → 질문마다 `skeletonize` → `template_check_report.json` 저장) |
| `corpus-check` (2026-09-23 추가) | `main` → `corpus_check`(내부에서 `load_all_merged_pairs` → `find_cross_sql_question_conflicts` → `corpus_check_report.json` 저장) — 개별 라운드가 아니라 `data/` 전체(모든 `pilot*`/`eval_indist*` 병합 파일)를 대상으로, 라운드를 넘나드는 라벨 충돌(같은 질문, 다른 SQL)을 찾음 |

### 함수

- **`stratified_sample(entries, n_total, seed) -> list[tuple[int, dict]]`**
  - 입력: SQL 엔트리 리스트, 뽑을 개수
  - 처리: `(table, where_col)` 조합 비율을 유지한 채 `n_total`개 추출
  - 출력: `(원본 인덱스, 엔트리)` 쌍 리스트

- **`build_batch_prompt(batch) -> str`**
  - 입력: SQL 배치(최대 25개)
  - 처리: 스키마 설명 + 규칙 9개 + `id=.. | sql: .. | ...` SQL 목록을 이어붙임 (내부에서 `format_literal_for_prompt` 사용)
  - 출력: LLM에 붙여넣을 완성 프롬프트 텍스트

- **`format_literal_for_prompt(where_val, coltype) -> str`**
  - 입력: 값, 타입
  - 처리: 프롬프트에 보여줄 형식으로 변환 (float→소수점 2자리)
  - 출력: 문자열

- **`_make_prompts_impl(out_dir, train_path, n, batch_size, seed, exclude_dirs, prompt_builder, log_prefix="") -> None`**
  - 입력: `make_prompts`/`make_prompts_small_model` 공통 인자 + 어느 프롬프트 빌더 함수를 쓸지(`prompt_builder`)
  - 처리: 이전 라운드 SQL 제외 → `stratified_sample` → `batch_size`개씩 나눠 `prompt_builder(batch)` 호출
  - 출력: 없음 — `prompts/batch_NN.txt`들과 `pilot_sql.json` 파일 저장이 결과물

- **`make_prompts(out_dir, train_path, n, batch_size, seed, exclude_dirs=None) -> None`**
  - 입력: `sql_train.json` 경로, 뽑을 개수·배치 크기·이전 라운드 폴더들
  - 처리: `_make_prompts_impl`을 `build_batch_prompt`로 호출 (강한 모델용, 사람이 붙여넣는 걸 전제)
  - 출력: 없음

- **`QUESTION_STYLE_SLOTS` (상수)** / **`EXAMPLE_QUESTIONS_BY_SLOT` (상수)**
  - 각각 5~8개 말투 슬롯 설명 문자열 리스트, 그 슬롯 순서에 맞춘 few-shot 예시 질문 리스트
  - `small_model_rules_blurb`가 `n_questions`만큼 앞에서부터 잘라 씀 (그래서 `n_questions`는 이 리스트
    길이, 현재 8을 넘을 수 없음)

- **`small_model_rules_blurb(n_questions=5) -> str`** (2026-09-23 추가)
  - 입력: SQL 하나당 요구할 질문 개수
  - 처리: `build_batch_prompt`의 규칙 9개보다 더 구체적으로 풀어 쓰고, `n_questions`개 말투 슬롯을
    번호 매겨 명시하고, `EXAMPLE_QUESTIONS_BY_SLOT` 기반 few-shot 예시 1개를 포함, JSON-only 출력
    지시를 반복 강조
  - 출력: 프롬프트에 들어갈 규칙 텍스트

- **`build_small_model_batch_prompt(batch, n_questions=5) -> str`** (2026-09-23 추가)
  - 입력: SQL 배치, 질문 개수
  - 처리: `SCHEMA_BLURB` + `small_model_rules_blurb(n_questions)` + SQL 목록을 이어붙임
  - 출력: 약한/로컬 모델용 프롬프트 텍스트

- **`make_prompts_small_model(out_dir, train_path, n, batch_size, seed, exclude_dirs=None, n_questions=5) -> None`** (2026-09-23 추가)
  - 입력: `make_prompts`와 동일 + `n_questions`
  - 처리: `_make_prompts_impl`을 `build_small_model_batch_prompt`(부분 적용, `n_questions` 고정)로 호출,
    배치 크기 기본값이 더 작음(호출부 기준 8)
  - 출력: 없음

- **`_isolated_number_match(q, lit) -> bool`** (2026-09-23 추가)
  - 입력: 질문 문자열, 숫자 리터럴 문자열
  - 처리: `lit`이 질문 안에 등장하되, 앞뒤로 숫자나 `.`이 이어 붙어 더 긴 숫자의 일부가 되는 경우는
    제외(예: 목표값 `97.56`이 `197.56` 안에 파묻히거나 `97.567`처럼 소수부가 더 이어지면 실패, 문장을
    끝내는 마침표는 허용)
  - 출력: `True`/`False`

- **`literal_in_question(q, where_val, coltype) -> bool`**
  - 입력: 질문 문자열, 정답 리터럴·타입
  - 처리: 텍스트는 단어 경계(자연스러운 복수형 `s`/`'s`/`es`는 허용) 매칭, 숫자(float/int)는
    `_isolated_number_match`로 매칭 — 원래는 단순 substring 검사였으나 2026-09-23에 경계 인식
    방식으로 강화(숫자가 더 긴 숫자에 파묻혀 오탐 통과하는 문제 방지)
  - 출력: `True`/`False`

- **`check_question(q, entry) -> list[str]`**
  - 입력: 질문 하나, SQL 메타데이터
  - 처리: 소문자 규칙·리터럴 포함·숫자 단어 사용·스네이크케이스 노출·동의어 치환 5종 검사
  - 출력: 위반 사유 리스트 (비어있으면 통과)

- **`soft_flags(q, entry) -> list[str]`**
  - 입력: 질문, 메타데이터
  - 처리: stock/quantity 혼동 의심 표현 확인 + (2026-09-23 추가) `select == where_col`인데 존재확인/
    검증 프레이밍("is there"/"exist"/"confirm"/"verify"/"check")이 아니면 규칙10 위반으로 표시
    (동어반복형 의심)
  - 출력: 경고 문자열 리스트 (하드 실패 아님)

- **`find_intra_group_duplicates(qs) -> set[int]`** (2026-09-23 추가)
  - 입력: 같은 SQL(id)에 달린 질문 리스트
  - 처리: 문장부호를 지우고 정규화한 뒤, 먼저 나온 문장과 사실상 동일한(완전 동일 포함) 재등장 문장의
    인덱스를 찾음 — 첫 등장은 정상으로 둠
  - 출력: 실패 처리해야 할 인덱스 집합 (규칙6: 5문장은 서로 달라야 함, `template_check`는 다른 id
    사이의 재사용만 보고 같은 id 안의 중복은 못 잡으므로 이걸로 보완)

- **`find_cross_sql_question_conflicts(pairs) -> list[dict]`** (2026-09-23 추가)
  - 입력: `{"question":..., "sql":...}` 쌍 리스트 (라운드 하나가 아니라 전체 코퍼스 가능)
  - 처리: 동일한 질문 문자열이 서로 다른 SQL에 매핑된 경우를 찾음 (완전히 같은 (질문,SQL) 중복은 제외)
  - 출력: `[{"question":..., "conflicting_sqls": [...]}]` — 진짜 라벨 충돌 목록

- **`_load_pilot_sql(out_dir) -> dict[int, dict]`**
  - 입력: 라운드 폴더
  - 처리: `pilot_sql.json` 읽기
  - 출력: `{id: SQL 메타데이터}`

- **`_load_responses(out_dir) -> tuple[dict[int, list[str]], list[str]]`**
  - 입력: 라운드 폴더
  - 처리: `responses/*.json` 전부 읽기. 2026-09-23부터 항목 단위 예외 처리 추가 — JSON 배열이 아닌
    응답, `id`/`questions` 필드 누락, 같은 배치 안 `id` 중복을 개별적으로 감지해 `missing_files`에
    보고하고 나머지 정상 항목은 계속 처리(예전엔 배치 하나의 형식 오류가 전체를 예외로 죽였음)
  - 출력: `({id: 질문 리스트}, 문제 상황 설명 문자열 리스트)`

- **`validate(out_dir, expected_questions=5) -> dict`**
  - 입력: 라운드 폴더, SQL 하나당 기대하는 질문 개수(2026-09-23 추가 — `make_prompts_small_model`을
    `n_questions`로 다르게 만들었으면 여기도 맞춰야 함)
  - 처리: `_load_pilot_sql`/`_load_responses`로 읽은 뒤, id별로 `find_intra_group_duplicates`를 먼저
    돌리고 모든 질문에 `check_question` 적용(중복으로 잡힌 인덱스는 추가 실패 사유로 병합), 통과분만
    수집 후 `soft_flags`
  - 출력: report 딕셔너리 — `pilot_train_pairs.json`(통과 쌍), `validation_report.json`(통계) 저장

- **`skeletonize(q, entry) -> str`**
  - 입력: 질문, 메타데이터
  - 처리: 질문 안 리터럴 값을 `<V>`로 치환
  - 출력: "틀" 문자열 (다른 id에 재사용됐는지 비교용)

- **`template_check(out_dir, threshold=0.3) -> dict`**
  - 입력: 라운드 폴더, 재사용 비율 임계치
  - 처리: `(select, where_col)` 조합별로 `skeletonize` 결과 재사용 비율 계산
  - 출력: report 딕셔너리 — `template_check_report.json` 저장

- **`load_all_merged_pairs(data_dir) -> list[dict]`** (2026-09-23 추가)
  - 입력: `data/` 폴더 경로
  - 처리: `pilot*/pilot_train_pairs.json` + `eval_indist*/pilot_train_pairs.json`(즉 지금까지 merge된
    모든 라운드)를 전부 합침 — `dataset.py`의 `load_train_pairs`/`load_eval_indist_pairs`와 같은 glob
  - 출력: `{"question":..., "sql":...}` 쌍 전체 리스트

- **`corpus_check(data_dir) -> dict`** (2026-09-23 추가)
  - 입력: `data/` 폴더 경로
  - 처리: `load_all_merged_pairs` → `find_cross_sql_question_conflicts` — 개별 라운드가 아니라 지금까지
    커밋된 전체 코퍼스 차원에서 라벨 충돌을 찾음(라운드를 넘나드는 충돌은 개별 라운드의 `validate`로는
    못 잡음)
  - 출력: report 딕셔너리 — `corpus_check_report.json` 저장

- **`main() -> int`**
  - 입력: 커맨드라인 인자 (`make-prompts`/`make-prompts-small`/`validate`/`template-check`/`corpus-check`)
  - 처리: 해당 함수 호출
  - 출력: 종료 코드

---

## `src/data/generator/question_gen_auto.py` — 로컬 LLM(Ollama) 질문 생성 파이프라인 (완전 자동화, 2026-09-23 추가)

`question_gen.py`의 샘플링·프롬프트 생성·검증 함수를 그대로 가져다 쓰고(`import question_gen as qg`),
"사람이 프롬프트를 LLM 세션에 붙여넣는다" 단계만 로컬 Ollama API 호출로 바꾼 파일. 검증 로직은 이
파일에 따로 없다 — 전부 `question_gen.py`를 재사용.

### 실행 순서

```
main → run_pipeline
  → qg.make_prompts_small_model  (샘플링 + 프롬프트 생성, question_gen.py 재사용)
  → 배치마다 반복:
      call_ollama  (내부: extract_json_array[ _try_parse → _repair_bracket_patterns → _regex_extract_items ])
      실패 시 남은 예산 안에서 재시도(--max-retries), 연속 실패가 임계치 넘으면 조기 중단
      (ollama 서버 다운으로 추정, --max-consecutive-failures)
  → qg.validate            (question_gen.py 재사용)
  → qg.template_check      (question_gen.py 재사용)
  → pipeline_summary.json 저장
```

### 함수

- **`_try_parse(text) -> list | None`**
  - 입력: JSON처럼 보이는 문자열
  - 처리: `json.JSONDecoder().raw_decode`로 앞부분만 파싱 시도(뒤에 쓰레기 문자가 있어도 허용)
  - 출력: 파싱된 리스트, 실패/리스트가 아니면 `None`

- **`_expected_item_count(text) -> int`**
  - 입력: 원본 텍스트
  - 처리: `"id":` 등장 횟수를 셈 — 괄호가 깨져도 이 개수는 보존된다는 전제
  - 출력: 원래 있어야 할 항목 개수 추정치

- **`_repair_bracket_patterns(text) -> list | None`**
  - 입력: 파싱 실패한 원본 텍스트
  - 처리: 실제로 관찰된 두 가지 패턴을 정규식으로 보정 — (1) `"questions"` 배열을 닫는 `]` 누락(`"}` →
    `"]}`), (2) 객체 사이 콤마 누락(`}{` → `},{`). 전역 치환으로 안 되면 `]` 삽입 위치를 하나씩 바꿔가며
    재시도. 복구 결과 항목 개수가 `_expected_item_count`와 다르면(일부만 조용히 잘려나간 거짓 성공)
    버림
  - 출력: 복구된 리스트, 실패 시 `None`

- **`_regex_extract_items(text) -> list | None`**
  - 입력: 위 두 복구도 실패한 원본 텍스트
  - 처리: 괄호 구조를 아예 무시하고 `"id":<숫자>,"questions":[<따옴표 문자열들>]` 패턴만 정규식으로
    직접 추출 — 배열/객체 닫는 순서가 뒤바뀌거나 `{`가 통째로 빠져도 살아남는 마지막 수단
  - 출력: 복구된 리스트, 항목 개수가 안 맞으면 `None`

- **`extract_json_array(raw) -> list | None`**
  - 입력: 모델의 원본 응답 텍스트
  - 처리: `<think>` 태그·코드펜스 제거 → `_try_parse` → 실패 시 `_repair_bracket_patterns` → 그래도
    실패 시 `_regex_extract_items` (3단계, 앞 단계가 성공하면 뒷 단계는 안 씀)
  - 출력: 복구된 JSON 배열, 완전히 실패하면 `None`

- **`call_ollama(model, prompt, timeout) -> tuple[str, list | None, float]`**
  - 입력: 모델 이름, 프롬프트, 타임아웃(초)
  - 처리: ollama REST API(`/api/generate`, `think=false`)를 직접 호출(CLI는 스트리밍 렌더링용 ANSI
    코드가 섞여 들어와서 안 씀), 응답을 `extract_json_array`로 파싱
  - 출력: `(원본 응답 텍스트, 파싱된 리스트 또는 None, 소요 시간)`

- **`run_pipeline(model, train_path, n, batch_size, n_questions, seed, out_dir, exclude_dirs, budget_seconds, per_batch_timeout, max_retries, max_consecutive_failures=5) -> dict`**
  - 입력: 모델·샘플링·시간 예산 관련 전체 인자
  - 처리: `qg.make_prompts_small_model`로 프롬프트 생성 → 배치마다 `call_ollama`(이미 완료된 배치는
    건너뜀 — 중단 후 재시작 안전장치) → 연속 실패 임계치 넘으면 조기 중단 → `qg.validate`/
    `qg.template_check` 호출
  - 출력: 통과율·반복비율·소요시간 등을 담은 summary 딕셔너리, `pipeline_summary.json`으로도 저장

- **`main() -> int`**
  - 입력: 커맨드라인 인자 (`--model`/`--n`/`--batch-size`/`--questions-per-sql`/`--out`/`--budget-seconds`/
    `--per-batch-timeout`/`--max-retries`/`--max-consecutive-failures`/`--exclude-dir` 등)
  - 처리: `run_pipeline` 호출 후 요약 출력
  - 출력: 종료 코드

---

## `src/tokenizer/bpe.py` — BPE 토크나이저

### 실행 순서

CLI 서브커맨드가 없는 단일 스크립트. `python -m src.tokenizer.bpe` 실행 시:

```
(모듈 로드 시) _validate_seeds
load_corpus
  → train  (내부: corpus_piece_freqs → 반복[get_pair_counts → merge_word]) — merges와 id_to_bytes를 둘 다 반환
  → save_merges
  → 전체 코퍼스 round-trip 검증  (내부: encode[pre_tokenize → apply_merges] → decode, train()이 반환한
    id_to_bytes를 그대로 재사용 — id_to_bytes_from_merges는 여기서 안 쓰임)
  → holdout 미등장 이름 평균 토큰 수 측정  (내부: encode)
```

`id_to_bytes_from_merges()`는 이 스크립트 자체 실행에서는 호출되지 않는다 — `train()`을 다시 돌리지 않고
저장된 `merges`만으로(예: `dataset.py`처럼 별도 프로세스에서) 디코딩 테이블을 복원해야 할 때 쓰는
함수다.

### 함수

- **`_validate_seeds(seeds) -> None`**
  - 입력: `SEED_TOKENS`
  - 처리: 중복·최소 길이·앞뒤 공백 규칙 검사 (모듈 로드 시 1회 자동 실행)
  - 출력: 없음 — 위반 시 `AssertionError`

- **`_build_pattern(seeds) -> re.Pattern`**
  - 입력: seed 토큰 리스트
  - 처리: seed는 통째로(길이 긴 것부터, 단어 경계), 나머지는 알파벳 묶음/숫자 1글자/공백/아무 문자 순으로 매칭되게 정규식 조립
  - 출력: 컴파일된 정규식 패턴

- **`pre_tokenize(text) -> list[Piece]`**
  - 입력: 임의의 문자열
  - 처리: 위 정규식으로 순서대로 조각냄, 빈틈 없는지 내부 assert
  - 출력: `(문자열, seed_id 또는 None)` 튜플 리스트

- **`pieces_to_text(pieces) -> str`**
  - 입력: `pre_tokenize`의 출력
  - 처리: 조각 문자열들을 이어붙임
  - 출력: 복원된 원문

- **`check_roundtrip(text) -> None`**
  - 입력: 문자열
  - 처리: `pre_tokenize` → `pieces_to_text` 결과가 원문과 같은지 확인
  - 출력: 없음 — 다르면 `AssertionError`

- **`show(text) -> None`**
  - 입력: 문자열
  - 처리: 조각 분해 결과를 사람이 읽기 좋게 정리 (디버깅용)
  - 출력: 없음 (콘솔 출력)

- **`load_corpus(data_dir="data") -> list[str]`**
  - 입력: 데이터 폴더
  - 처리: `pilot*/pilot_train_pairs.json`(학습 라운드만, `eval_indist*`는 이름이 달라 제외)을 전부 읽음
  - 출력: question·sql 문자열이 펼쳐진 리스트

- **`corpus_piece_freqs(corpus) -> Counter[str]`**
  - 입력: 문자열 리스트
  - 처리: 각각 `pre_tokenize` 후 seed가 아닌 조각의 등장 빈도 집계
  - 출력: 조각별 빈도 `Counter` (BPE 학습의 출발점)

- **`get_pair_counts(word_syms, freqs) -> Counter[tuple[int,int]]`**
  - 입력: `{단어: 현재 심볼 ID 리스트}`, 단어별 빈도
  - 처리: 인접 심볼 쌍이 전체 코퍼스에서 몇 번 나오는지(빈도 가중) 집계
  - 출력: 쌍별 빈도 `Counter` — 매 병합 스텝마다 재호출됨

- **`merge_word(syms, pair, new_id) -> list[int]`**
  - 입력: 심볼 ID 리스트, 병합할 쌍, 새 ID
  - 처리: 그 쌍이 연속으로 나오는 자리를 전부 `new_id`로 치환
  - 출력: 병합된 심볼 ID 리스트

- **`build_base_id_to_bytes() -> dict[int, bytes]`**
  - 입력: 없음
  - 처리: ID 0~255는 해당 바이트, seed 토큰 ID는 그 문자열의 UTF-8 바이트로 채움
  - 출력: `{ID: bytes}` (병합 전 기본 상태)

- **`train(corpus, merge_budget=MERGE_BUDGET) -> tuple[list[Merge], dict[int, bytes]]`**
  - 입력: 전체 코퍼스, 병합 예산(726)
  - 처리: 매 스텝 최빈 쌍(동점 시 사전순 최솟값)을 찾아 병합, 예산만큼 반복
  - 출력: `(병합 규칙 리스트, 최종 id→bytes 딕셔너리)`

- **`id_to_bytes_from_merges(merges) -> dict[int, bytes]`**
  - 입력: 저장된 병합 규칙
  - 처리: `build_base_id_to_bytes()`에서 시작해 병합을 순서대로 재적용
  - 출력: `{ID: bytes}` — 학습 없이 저장된 `merges`만으로 디코딩 테이블 복원

- **`apply_merges(syms, merges) -> list[int]`**
  - 입력: 바이트 단위 심볼 ID 리스트, 병합 규칙
  - 처리: 규칙을 저장 순서대로 전부 적용
  - 출력: 최종 심볼 ID 리스트 (인코딩의 핵심 루프)

- **`encode(text, merges) -> list[int]`**
  - 입력: 문자열, 병합 규칙
  - 처리: `pre_tokenize`로 조각 → seed는 그대로, 나머지는 `apply_merges`
  - 출력: 토큰 ID 리스트

- **`decode(ids, id_to_bytes) -> str`**
  - 입력: 토큰 ID 리스트, id→bytes 매핑
  - 처리: 특수 토큰은 이름 문자열로, 나머지는 bytes를 이어붙여 UTF-8 디코딩
  - 출력: 복원된 원문 문자열

- **`save_merges(merges, path) -> None`** / **`load_merges(path) -> list[Merge]`**
  - 입력: 병합 규칙 리스트 / JSON 파일 경로
  - 처리: JSON으로 저장 / 읽기
  - 출력: 없음 / 병합 규칙 리스트

- **`if __name__ == "__main__":` 블록**
  - 함수는 아니지만, 위 "실행 순서" 참고. 전체 코퍼스 로드부터 `bpe_merges.json` 저장, round-trip 검증, holdout 토큰 수 측정까지 한 번에 실행되며 결과를 콘솔에 출력

---

## `src/data/dataset.py` — PyTorch Dataset

### 실행 순서

CLI 서브커맨드 없음. `python -m src.data.dataset` 실행 시 (자체 테스트, 저장 파일 없음):

```
load_merges
  → load_train_pairs / load_eval_indist_pairs  (내부: load_pairs)
  → tokenize_pairs  (내부: 쌍마다 encode_pair → bpe.encode)
  → 길이 통계 출력
  → 샘플 round-trip 확인  (내부: bpe.decode)
  → collate_fn 단독 검증
  → DataLoader(TextToSQLDataset, collate_fn=collate_fn) 로 실제 배치 하나 확인
```

`train.py`가 이 모듈을 가져다 쓸 때는 보통 `load_merges` → `tokenize_pairs` → `TextToSQLDataset` →
`DataLoader(..., collate_fn=collate_fn)` 순서로, 학습 루프 시작 시 1회만 호출한다.

### 함수

- **`load_pairs(data_dir, glob) -> list[Pair]`**
  - 입력: 데이터 폴더, glob 패턴
  - 처리: 매칭되는 모든 `pilot_train_pairs.json`의 항목을 합침
  - 출력: `{"question":..., "sql":...}` 리스트

- **`load_train_pairs(data_dir="data") -> list[Pair]`** / **`load_eval_indist_pairs(...) -> list[Pair]`**
  - 입력: 데이터 폴더 경로
  - 처리: `load_pairs`를 각각 `"pilot*/..."`(학습, `pilot_merged`+`pilot_qwen3_merged` 합쳐 22,862쌍) /
    `"eval_indist*/..."`(평가, 5,202쌍) 패턴으로 호출 — 2026-09-23 데이터 정리·Qwen3 추가 반영
  - 출력: (question, sql) 쌍 리스트

- **`encode_pair(question, sql, merges) -> list[int]`**
  - 입력: question 문자열, sql 문자열, BPE 병합 규칙
  - 처리: 각각 `bpe.encode()` 후 `<bos> question <sep> sql <eos>` 순서로 이어붙임
  - 출력: 토큰 ID 리스트

- **`tokenize_pairs(pairs, merges) -> list[Example]`**
  - 입력: (question, sql) 쌍 리스트, 병합 규칙
  - 처리: 각각 `encode_pair` 호출, SQL 시작 인덱스(`sql_start`)도 기록
  - 출력: `{"ids":..., "sql_start":...}` 리스트

- **`TextToSQLDataset(examples)`** — `__len__` / `__getitem__`
  - 입력: 토큰화된 예시 리스트
  - 처리: 감싸기만 함 (패딩은 안 함)
  - 출력: `__getitem__(idx)`는 인덱스를 받아 그 예시를 그대로 반환

- **`collate_fn(batch) -> dict[str, torch.Tensor]`**
  - 입력: 예시 리스트 (한 배치분)
  - 처리: 배치 최대 길이로 `<pad>` 오른쪽 패딩 → `input_ids`/`target_ids` 한 칸 시프트 → SQL+`<eos>` 구간만 `True`인 `loss_mask` 생성
  - 출력: `{"input_ids":..., "target_ids":..., "loss_mask":...}` 텐서 딕셔너리

- **`if __name__ == "__main__":` 블록**
  - 함수는 아니지만, 위 "실행 순서" 참고. 저장하는 파일 없이 콘솔에 통계·검증 결과만 출력
