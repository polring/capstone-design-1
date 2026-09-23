# 환경 재현 가이드

이 저장소의 파이썬 스크립트로 로컬 환경을 다시 만드는 방법이다. DB·SQL(1·2단계)과 토크나이저·Dataset
(4·5단계)은 코드+시드로 바이트 단위까지 재현되지만, 질문 생성(3단계)만은 LLM 세션이 끼어 있어 완전히
같은 결과가 아니라 같은 절차·같은 검증 기준을 통과하는 동등한 데이터셋으로 재현된다(자세한 이유는
3-4절, 데이터를 커밋하지 않기로 한 결정은 맨 아래 "커밋해야 하는 것" 참고). 각 스크립트의 세부 옵션과
설계 배경은 [CLAUDE.md](CLAUDE.md)와 [text-to-sql-llm-project-plan.md](text-to-sql-llm-project-plan.md)에
있고, 함수 단위 입출력은 [function-reference.md](function-reference.md)에 있다.

## 폴더 구조

이 저장소(`capstone-design-1/`)의 실제 git 루트는 팀원 4명이 공유하는 상위 폴더다. 팀원1이 다루는
코드·데이터·문서는 다른 팀원 폴더와 겹치지 않게 전부 `capstone-design-1/datasets/` 밑에 몰아뒀고,
이 문서를 포함한 `.md` 문서 5개는 그 안에서 한 단계 더 들어간 `capstone-design-1/datasets/docs/`에
있다. 아래 안내에서 "저장소 루트"라고 쓴 곳은 실제 git 루트가 아니라 **`datasets/`**를 가리킨다.
전체 파일 구조는 [README.md](../README.md)에 있다.

## 요구 사항

- Python 3.10 이상 (개발에는 3.13.14 사용)
- `db/db_gen.py`, `db/sql_gen.py`, `db/question_gen.py`, `bpe_tokenizer.py`는 외부 패키지 없이 표준
  라이브러리(`sqlite3`, `json`, `re`, `argparse`, `collections`, `pathlib` 등)만 사용한다
- Dataset/train.py 단계부터는 PyTorch·NumPy·TensorBoard가 필요하다 — 아래 0절 참고

## 0. 가상환경 설정 및 PyTorch 설치

`.venv`는 `datasets/` 안에 있다(`requirements.txt`와 같은 위치) — 아래 명령은 **`datasets/`에서** 실행한다.

```
cd datasets
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt`가 `--extra-index-url`로 PyTorch CUDA 빌드 인덱스를 가리키므로 GPU 빌드가 그대로 깔린다.
Windows용 CUDA 빌드는 일반 PyPI에 없다 — `pip install torch`만 하면 CPU 전용이 깔리므로 반드시
`requirements.txt`를 거칠 것.

- **CUDA 빌드 버전 고르는 법**: `nvidia-smi` 출력의 "CUDA Version"이 드라이버가 지원하는 **최대치**다(설치된
  CUDA 툴킷 버전이 아님). 이 값 이하의 가장 최신 PyTorch 빌드 태그를 고른다
  (https://download.pytorch.org/whl/ 에서 `cuNNN` 목록 확인). 2026-09-21 기준 이 환경은 드라이버
  591.86(CUDA 13.1 지원) + RTX 4070 Super로, `cu130`(CUDA 13.0) 빌드를 선택했다. 드라이버를 업그레이드하면
  더 최신 빌드로 올릴 수 있다.
- **설치 확인**:
  ```
  .venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  `2.14.0+cu130 True NVIDIA GeForce RTX 4070 SUPER`처럼 나와야 한다. `False`가 나오면 드라이버 버전과
  빌드 태그가 안 맞는 경우가 가장 흔하다.
- 이후 모든 `python` 명령은 `.venv\Scripts\python`(또는 venv를 activate한 셸)으로 실행한다고 가정한다.
  `.venv`가 `datasets/` 안에 있으므로, `datasets/`(또는 그 하위 `db/`)에서는 상대경로 그대로
  (`db/`에서는 `..\.venv\Scripts\python`) 쓰면 되고, 미리 `.venv\Scripts\activate`로 셸을
  activate해두면 경로 신경 안 써도 된다.

## 실행 위치 주의

스크립트마다 상대 경로 기준이 다르다. 잘못된 위치에서 실행하면 `data/` 폴더가 엉뚱한 곳에 생긴다.
아래 "저장소 루트"는 위 폴더 구조대로 **`datasets/`**를 뜻한다 (실제 git 루트인 `capstone-design-1/`가 아님).

| 스크립트 | 실행 위치 | 기본 출력 경로 |
| --- | --- | --- |
| `db/db_gen.py`, `db/sql_gen.py`, `db/question_gen.py` | `datasets/db/` 폴더 안 | `db/data/...` (`--out` 기본값이 `./data`) |
| `bpe_tokenizer.py`, `dataset.py` | `datasets/` | `db/data/...`, `bpe_merges.json`을 상대경로로 읽음 (`dataset.py`는 저장 파일 없이 콘솔 출력만) |

## 1. DB 생성

```
cd datasets/db          # capstone-design-1/ 루트에서 (datasets/db/ — 위 폴더 구조 참고)
python db_gen.py build            # db/data/shop.db, db/data/holdout.json 생성 (seed=0, preset=large 기본값)
python db_gen.py test             # 분포/결정성/price 등호/홀드아웃 분리 4종 검증
```

## 2. SQL 생성

```
python sql_gen.py build --cap-nonid 9999 --cap-id 9999
# db/data/sql_train.json (학습용 4,609개), db/data/sql_eval_indist.json (분포 내 평가용 1,154개) 생성
python sql_gen.py verify          # 완료 조건 재확인
python sql_gen.py test            # 결정성/홀드아웃 리터럴 미포함/학습-평가 분리 검증
```

## 3. 질문 생성 (LLM 세션 필요 — 완전 자동화 불가)

`db/data/pilot`~`pilot13`(학습용, 13라운드)과 `db/data/eval_indist1`~`eval_indist6`(분포 내 평가용,
6라운드)는 각 라운드마다 `prompts/`·`responses/`·`validation_report.json`·`template_check_report.json`
등 중간 산출물을 포함해 폴더 전체가 약 8.1MB인데, 이 중 실제로 필요한 건 검증 통과한 (질문,SQL) 쌍뿐이다.
그래서 (2026-09-21) 13라운드 전부를 `db/data/pilot_merged/pilot_train_pairs.json`(23,045쌍, 약 2.5MB)으로,
6라운드 전부를 `db/data/eval_indist_merged/pilot_train_pairs.json`(5,770쌍, 약 0.65MB)으로 합쳐서
**이 합친 파일 2개만 GitHub에 커밋**하기로 했다. 원본 19개 라운드 폴더는 `db/data_raw/`로 옮겨 로컬에는
그대로 두되 커밋은 안 한다(`.gitignore` 참고) — DB/SQL(1·2단계)과 달리 이 데이터는 LLM 세션 19라운드를
다시 거쳐야만 재현되므로(아래 3-4절), 합친 파일을 커밋해두면 새로 클론한 사람이 그 과정을 반복할 필요가
없다. `load_corpus()`/`load_train_pairs()`/`load_eval_indist_pairs()`는 `pilot*/`, `eval_indist*/`를
글롭으로 찾으므로 `db/data_raw/`(이름이 다름)는 안 잡히고 병합 폴더만 잡힌다 — 코드 변경 없음.

아래 3-1~3-3은 그 19라운드가 처음에 어떻게 만들어졌는지의 절차이고, 지금 다시 밟을 필요는 없다(이미
병합해서 커밋해뒀으므로). 이후 라운드를 새로 추가하게 되면 이 절차를 그대로 따르면 된다.

### 3-1. pilot 폴더가 만들어지는 절차

1. **`make-prompts`**: `stratified_sample()`이 학습 SQL(`sql_train.json`)에서 `(table, where_col)` 조합
   비율을 유지한 채 `--n`개를 뽑고, `--batch-size`(기본 25)개씩 나눠 `prompts/batch_NN.txt`를 만든다.
   `--exclude-dir`에 이전 라운드들을 나열하면 그 라운드들의 `pilot_sql.json`에 있던 SQL을 후보에서
   빼서 라운드 간 중복을 막는다. 각 배치의 정답 SQL과 메타데이터(select/where_col/literal/exists 여부)는
   `pilot_sql.json`에 같이 저장된다 — 이 파일은 검증에만 쓰고 LLM에게는 보여주지 않는다.
2. **사람이 `batch_NN.txt`를 독립된 LLM 세션에 하나씩 붙여넣는다.** 배치 하나당 반드시 별도 세션이어야
   한다 — 한 세션이 여러 배치(또는 라운드 전체)를 처리하면 같은 `(SELECT, WHERE)` 조합이 10여 개를
   넘는 순간 표현이 소수의 "무난한 문장"으로 수렴하는 현상이 실측으로 확인됐다(CLAUDE.md 참고). 응답
   JSON을 `responses/batch_NN.json`으로 저장한다.
3. **`validate`**: 모든 응답을 모아 규칙대로 검사해 통과한 쌍만 `pilot_train_pairs.json`에, 통과율과
   실패 사유는 `validation_report.json`에 저장한다.
4. **`template-check`**: 표현이 몇 가지 틀로만 획일화되지 않았는지 진단(참고용, 통과/실패 게이트 아님).

```
python question_gen.py make-prompts --out data/pilotN --n 200 --batch-size 25 --seed N \
    --exclude-dir data_raw/pilot,data_raw/pilot2,...,data_raw/pilot13,data_raw/eval_indist1,...,data_raw/eval_indist6
# (원본 19라운드가 data_raw/로 옮겨졌으니 exclude-dir도 그쪽을 가리켜야 함 — data/pilot_merged,
#  data/eval_indist_merged는 pilot_sql.json이 없어서 exclude-dir로 못 씀. 위 "3. 질문 생성" 도입부 참고)
# prompts/batch_NN.txt 각각을 독립된 LLM 세션에 전달 → 응답을 responses/batch_NN.json으로 저장
python question_gen.py validate --out data/pilotN        # pilot_train_pairs.json 생성 + 자동 검증
python question_gen.py template-check --out data/pilotN  # 표현 다양성 진단
```

### 3-2. 프롬프트 구성 (`build_batch_prompt()`가 만드는 실제 텍스트, 3부분)

1. **스키마 설명**: customers/items/orders 3테이블의 컬럼을 자연어로 설명하고, "stock(재고 남은 수량)과
   quantity(주문한 수량)는 헷갈리기 쉬우니 구분하라"는 주의를 덧붙인다.
2. **규칙 9개** — 모두 지켜야 그 질문이 통과된다:
   1. 전부 소문자, 과한 문장부호 스타일링 금지
   2. WHERE 리터럴을 철자/숫자 그대로 반드시 포함(의역 금지)
   3. 상품/카테고리 단어를 동의어·상위어로 바꾸지 않기(예: laptop→computer, sneakers→shoes,
      `stock = 0`→sold out 금지)
   4. 숫자를 단어로 쓰지 않기("twenty" 대신 항상 숫자)
   5. 다른 테이블 정보가 필요한 질문(JOIN 필요) 금지 — 그 SQL 하나로 알 수 있는 사실만
   6. 같은 SQL에 대한 질문 5개는 말투(격식체/캐주얼/키워드식/명령문/의문문 등)가 서로 확실히 다를 것
   7. SELECT가 `stock`이면 재고를, `quantity`면 주문 수량을 묻기 — 절대 섞지 않기
   8. SELECT가 `*`이면 특정 컬럼을 지목하지 않는 일반적인 "알려줘/보여줘" 식 질문
   9. `item_id`/`order_id`처럼 스네이크케이스 컬럼명을 질문에 그대로 쓰지 않고 자연어로 풀어쓰기
   그리고 "JSON 배열만, 코드펜스 없이" 라는 출력 형식 지정이 뒤따른다.
3. **SQL 목록**: 그 배치에 속한 SQL들을 `id=.. | sql: .. | select=.. | where_col=.. | where_value=.. |
   exists=True/False` 형식으로 한 줄씩 나열.

### 3-3. 검증 로직 (`question_gen.py`의 `check_question()`)

하나라도 걸리면 그 질문은 탈락하는 하드 실패 검사:

1. **소문자 규칙 위반** — `q != q.lower()`
2. **리터럴 미포함** — 문자열 값은 소문자로, float 값은 소수점 둘째 자리 포맷으로, 그 외는 그대로
   질문 텍스트 안에 정확히 있는지
3. **숫자를 단어로 씀** — `zero`~`hundred` 집합(`NUMBER_WORDS`)과 겹치는 토큰이 있는지. `"one"`은
   "is this one of ours" 같은 일상 표현에 흔해 오탐이 많아 이 집합에서 제외했다
4. **SQL 컬럼명을 스네이크케이스 그대로 씀** — `item_id`, `order_id` 등을 정규식으로 탐지
5. **동의어/상위어 치환 의심** — `KNOWN_SYNONYM_SWAPS`(computer↔laptop, shoes↔sneakers,
   sold out↔`stock = 0`, sofa↔couch) 중 금지어가 단어 경계로 등장하면 실패. 단 그 금지어 자체가
   원래 정답 리터럴인 경우는 제외한다(예: 상품명이 실제로 "couch"인 SQL)

하드 실패는 아니지만 사람이 확인하도록 표시만 하는 `soft_flags()`도 있다 — stock 질문인데 order
표현만 쓰였거나 그 반대인 경우(재고/주문 수량 혼동 의심). SQL 하나당 질문이 정확히 5개가 아니거나
응답 자체가 없으면 그 SQL은 통째로 실패 처리된다.

### 3-4. "재현"의 의미가 다르다는 것

LLM 세션 응답은 결정적이지 않으므로, 이 단계는 같은 명령을 다시 돌려도 **완전히 같은 문장이 나오지
않는다.** DB·SQL·토크나이저(2·3·4단계)는 코드+시드로 바이트 단위까지 재현되지만, 질문 데이터는
"같은 규칙과 같은 자동 검증을 통과하는, 동등한 품질의 데이터셋"이 나온다는 의미로만 재현된다.

## 4. BPE 토크나이저 학습

```
cd ..            # datasets/ 로 이동 (실제 git 루트는 아님 — 위 폴더 구조 참고)
python bpe_tokenizer.py
```

`db/data/pilot*/pilot_train_pairs.json` 전체(46,090개 질문+SQL 문자열)를 코퍼스로 병합 726개를 학습하고,
결과를 `datasets/`의 `bpe_merges.json`에 저장한다. 이어서 전체 코퍼스 round-trip 검증과
`db/data/holdout.json` 미등장 이름 평균 분할 토큰 수 측정까지 같은 실행에서 끝낸다. 코퍼스가 고정되어
있고 동점 처리가 사전순 최솟값으로 결정적이므로, 이 단계는 실행할 때마다 완전히 동일한 `bpe_merges.json`이
나온다.

## 5. Dataset 자체 테스트

```
python dataset.py
```

`bpe_merges.json`으로 학습용(23,045쌍)·분포 내 평가용(5,770쌍) 전체를 토큰화해 길이 통계를 내고,
`collate_fn`/`DataLoader`로 배치 하나를 실제로 만들어 shape과 loss 마스크가 맞는지 확인한다. 파일을
저장하지 않고 콘솔 출력만 낸다 — `train.py`가 이 모듈을 가져다 쓸 때 실제로 호출하는 함수들
(`load_train_pairs`/`tokenize_pairs`/`TextToSQLDataset`/`collate_fn`)이 제대로 맞물리는지 미리 확인하는
용도다.

## 재현 결과 확인

경로는 모두 `datasets/` 기준이다.

| 파일 | 무엇을 확인하나 |
| --- | --- |
| `db/data/sql_gen_report.json` | SQL 구조별 할당량이 표(계획문서 5-3)와 일치하는지 |
| `bpe_merges.json` | 병합 726개, `MERGE_BASE=298`부터 순서대로 |
| `python bpe_tokenizer.py` 출력 | round-trip 실패 0건, 미등장 이름 평균 분할 토큰 수 ≈ 3.49 |
| `python dataset.py` 출력 | 시퀀스 길이 min=23 max=64 mean=38.2, 컨텍스트 128 초과 0건, `loss_mask` 검증 통과 |

## 커밋해야 하는 것 / 재생성 가능한 것

아래 경로는 실제 git 루트(`capstone-design-1/`) 기준이다 — `git add` 등 git 명령을 쓸 때는 이 경로 그대로 쓰면 된다.

- **커밋**: 파이썬 코드(`datasets/db/db_gen.py`, `datasets/db/sql_gen.py`, `datasets/db/question_gen.py`,
  `datasets/bpe_tokenizer.py`, `datasets/dataset.py`), `datasets/requirements.txt`, 계획 문서, 진행
  보고서, 이 가이드, [function-reference.md](function-reference.md) (전부 `datasets/docs/` 아래),
  그리고 데이터 산출물 `datasets/db/data/shop.db`, `sql_train.json`, `sql_eval_indist.json`,
  `holdout.json`, `sql_gen_report.json`, `pilot_merged/pilot_train_pairs.json`(23,045쌍),
  `eval_indist_merged/pilot_train_pairs.json`(5,770쌍) — 전부 1.5MB(DB/SQL) + 3.3MB(질문 쌍) 수준으로
  가볍고, DB/SQL은 시드 고정으로 바이트 단위까지, 질문 쌍은 이미 검증 통과한 최종 결과물이라 커밋해두면
  다른 사람이 4단계(토크나이저)까지 스크립트 실행만으로 바로 재현할 수 있다
- **커밋 안 함**: `.venv/`(로컬 가상환경, `requirements.txt`로 재현), `datasets/db/data_raw/`(3절 참고 —
  19라운드 원본, prompts/responses/report 포함 약 8.1MB — 이미 검증 통과분만 병합해서 커밋했으므로
  다시 볼 일이 없는 중간 산출물), `bpe_merges.json`(코드+시드로 재현되는 학습 산출물). `.gitignore`가
  이 경로들을 이미 제외하도록 돼 있다

### `bpe_merges.json`만 로컬에서 한 번 더 돌려야 한다

3-1~3-3의 원본 LLM 세션 19라운드(prompts/responses)는 커밋하지 않지만, 그 결과물(검증 통과한 질문 쌍)은
`pilot_merged`/`eval_indist_merged`로 커밋해뒀으므로, 이 저장소를 새로 클론한 사람은 1~3단계까지는 바로
재현된 상태로 받는다. 남은 건 `bpe_merges.json`(4절)뿐인데, 이건 `python bpe_tokenizer.py` 한 번이면
코드+고정 코퍼스+결정적 동점 처리로 항상 바이트 단위까지 똑같이 재현되므로, LLM 세션을 다시 거칠 필요
없이 스크립트 실행 한 번으로 끝난다.
