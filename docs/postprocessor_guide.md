# Python SQL 후처리기(Post-processor) 가이드

본 문서는 Issue #29를 통해 구현된 **Python 기반 SQL 후처리기**의 동작 원리, 알고리즘, 그리고 구성 파일의 역할을 설명합니다. 
최근 업데이트를 통해 하드코딩된 사전(`DEFAULT_DICTIONARY`)이 제거되고, 오직 동적 스키마 로딩(`shop.db`)과 순수 `numpy` 및 `GloVe` 기반의 의미론적 교정(Semantic Correction) 기능이 적용되도록 개선되었습니다.

---

## 1. 추가된 파일 및 모듈 역할

이번 구현을 통해 `src/postprocessor/python/` 및 `tests/` 폴더에 다음 파일들이 구성되었습니다.

* **`__init__.py`**
  * 모듈 초기화 파일로, 외부에서 `process_sql` 및 `load_dictionary_from_sqlite` 함수를 쉽게 임포트할 수 있게 해줍니다.
* **`core.py`**
  * 파싱, 교체, 재조립 모듈들을 연결하여 최종적인 SQL 교정 작업을 수행하는 핵심 비즈니스 로직(`process_sql` 함수)이 담긴 모듈입니다.
* **`main.py`**
  * CLI(Command Line Interface) 파라미터 파싱 및 단독 실행을 위한 진입점(Entry point)입니다. 
  * `-i`, `-s`, `-b`, `-d`, `-t` 파라미터를 받아 `core.py`의 파이프라인을 구동합니다. `--db`가 없을 경우 기본적으로 프로젝트 내 `data/shop.db`를 로드합니다.
* **`schema.py`**
  * 지정된 SQLite 데이터베이스(기본값: `data/shop.db`)에서 테이블명과 컬럼명을 동적으로 추출하여 교정 사전을 구축하는 모듈입니다.
* **`parser.py`**
  * **입력 SQL 파싱 및 식별자 추출**을 담당합니다.
  * `sqlglot` 라이브러리를 사용하며, 예약어 오타로 인한 파싱 에러를 막기 위해 AST 생성 전 텍스트 레벨에서 예약어를 1차 교정합니다.
* **`replacement.py`**
  * **단어 교체 알고리즘(Typo, N-gram, Semantic)**과 **전략 파이프라인(Chaining)**을 수행합니다.
  * 추출된 식별자들을 `schema.py`에서 얻어온 도메인 사전(Dictionary)과 비교하여 적절하게 교정합니다. 각 기능별(Typo, N-gram, Semantic)로 상단에 상세한 블록 주석이 포함되어 있습니다.
* **`build_semantic_dict.py` (오프라인 사전 구축기)**
  * 무거운 GloVe(6B) 원본 모델을 런타임에 매번 로드하는 것을 방지하기 위해, `shop.db`의 식별자들과 코사인 유사도가 높은 핵심 단어들의 임베딩만 추려내어 가벼운 `.pkl` 캐시 파일로 압축 생성해 주는 독립 실행 스크립트입니다.
* **`reconstruction.py`**
  * 교정이 완료된 AST 객체를 다시 **유효한 SQL 문자열로 재생성**하여 반환하는 역할을 합니다.
* **`tests/test_postprocessor_python.py`**
  * `pytest`를 활용하여 각 파이프라인(예약어 교정, 식별자 교정, Chaining 등)이 정상 작동하는지 검증하는 단위 테스트 파일입니다.

---

## 2. 후처리기 각 단계 및 알고리즘 설명

### 알고리즘 종류
후처리기는 상황에 맞춰 단어를 교정할 수 있도록 3가지 알고리즘 전략을 지원하며, 콤마(`,`)를 이용해 여러 개를 중복 적용(Chaining)할 수 있습니다.

1. **Typo (편집 거리 기반 교정)**
   * 파이썬 내장 모듈(`difflib`)을 활용하여 문자열 간의 삽입, 삭제, 변경 횟수를 기반으로 매칭합니다.
   * `usr_name` $\rightarrow$ `user_name` 처럼 1~2글자의 형태적 오타를 매우 빠르고 직관적으로 잡아냅니다.
2. **N-gram (부분 문자열 Jaccard 유사도)**
   * 단어를 N개(기본 2문자)로 쪼갠 집합을 만들어 두 집합 간의 **Jaccard 유사도(교집합/합집합)**를 계산합니다.
   * `departmnt` $\rightarrow$ `department` 등 길이가 길거나 편집 거리가 애매한 경우 패턴을 통해 교정합니다.
3. **Semantic (의미론적 유사도 기반 교정)**
   * 순수 `numpy` 행렬 연산과 스탠포드 **GloVe** 워드 임베딩을 사용하여, 문자 형태가 전혀 달라도 의미상 가장 유사한 단어(예: `products` $\rightarrow$ `items`)로 교정합니다. 상세 원리는 아래 **3. 핵심 기술 원리** 항목에서 설명합니다.
4. **전략 Chaining (다중 적용 및 Fallback)**
   * `--strategy typo,ngram,semantic` 처럼 넘겨주면 가장 가벼운 Typo부터 시도하고, 교정에 실패하면 다음 알고리즘으로 넘어가는(Fallback) 파이프라인 구조로 작동합니다.

---

## 3. 핵심 기술 원리: GloVe와 평균 풀링(Mean Pooling)

### 3.1. GloVe (Global Vectors for Word Representation) 모델
스탠포드 대학에서 개발한 GloVe는 단어들의 동시 등장 확률(Co-occurrence)을 학습하여 단어의 의미를 다차원 벡터 공간에 매핑하는 알고리즘입니다. 이 모델을 사용하면 `products`와 `items`처럼 스펠링(형태)이 전혀 다르더라도 서로 비슷한 문맥에서 자주 사용된다면 임베딩 공간 상에서 두 벡터의 거리가 매우 가깝게(코사인 유사도가 높게) 배치됩니다. 본 후처리기에서는 이 점을 이용해 의미론적 교정을 수행합니다.

### 3.2. 단어 분해(Split)와 평균 풀링(Mean Pooling)
데이터베이스 컬럼명이나 테이블명은 대개 `customer_id`, `order_status`와 같은 합성어(스네이크 케이스)로 이루어져 있습니다. 이러한 단어는 일반 자연어로 학습된 GloVe 기본 단어장(Vocabulary)에 통째로 존재하지 않기 때문에 일반적인 방식으로는 의미를 추출할 수 없습니다.

이를 해결하기 위해 본 후처리기는 다음과 같은 **평균 풀링(Mean Pooling)** 방식을 사용합니다:
1. **분해(Split):** `customer_id`라는 단어를 언더스코어(`_`) 기준으로 쪼개어 `customer`와 `id`로 분리합니다.
2. **벡터 추출 및 합산:** GloVe 단어장에서 `customer` 벡터와 `id` 벡터를 각각 추출하여 더합니다.
3. **평균(Mean):** 더해진 벡터를 구성 단어의 개수(2개)로 나누어 평균값을 구합니다.

이 과정을 거치면 두 단어의 의미가 고루 섞인 하나의 결합된 "대표 벡터"가 생성되며, 합성어 전체의 전반적인 의미를 임베딩 공간에 훌륭하게 표현할 수 있습니다. 이후 이 대표 벡터 간의 코사인 유사도를 계산하여 가장 의미가 가까운 스키마를 찾아냅니다.

---

## 4. CLI 옵션 및 실행 예시

### 4.1. 오프라인 의미론적 사전 캐시 생성 (`build_semantic_dict.py`)
GloVe 데이터 연산을 가볍게 처리하기 위해, 사전에 `.pkl` 캐시를 생성해야 합니다.

**사용 가능한 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `-d`, `--dim` | int | `300` | 생성할 GloVe 임베딩 차원 (`50`, `100`, `200`, `300` 중 택 1) |
| `-b`, `--db` | str | `data/shop.db` | 스키마를 추출할 SQLite 데이터베이스 경로 |
| `-D`, `--data_dir` | str | `data/` | GloVe 압축 파일 및 `.pkl` 캐시가 저장될 디렉토리 |
| `-t`, `--threshold` | float | **필수** | 코사인 유사도 기준(예: `0.5`). 이 점수 이상의 모든 유의어를 캐싱 |

**CMD 실행 예시:**
```cmd
:: 100차원을 사용하고, 유사도 0.5 이상인 단어를 캐싱
python -m src.postprocessor.python.build_semantic_dict -d 100 -t 0.5

:: 300차원을 사용하고, 0.6 이상으로 좀 더 엄격하게 필터링
python -m src.postprocessor.python.build_semantic_dict -d 300 -t 0.6

:: 별도의 데이터베이스 경로 지정
python -m src.postprocessor.python.build_semantic_dict -b "C:\path\to\custom.db" -d 50 -t 0.45
```

### 4.2. 런타임 SQL 후처리기 실행 (`main.py`)
오프라인 캐시(선택적)를 기반으로 실제 SQL 쿼리의 오타 및 의미적 불일치를 교정합니다.

**사용 가능한 옵션:**

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `-i`, `--input` | str | **필수** | 교정할 원본 SQL 문자열 |
| `-s`, `--strategy` | str | `typo,ngram,semantic` | 적용할 교정 전략 (콤마 `,`로 구분하여 우선순위 체이닝) |
| `-b`, `--db` | str | `data/shop.db` | 스키마를 추출할 SQLite 데이터베이스 경로 |
| `-d`, `--dim` | int | **필수** | Semantic 전략 시 사용할 차원 (`50`, `100`, `200`, `300` 중 택 1). 전략에 semantic이 포함되면 반드시 명시해야 합니다. |
| `-t`, `--threshold` | float | `0.0` | Semantic 전략에 적용할 코사인 유사도 커트라인(예: `0.5`). 미지정 시 캐시에 있는 모든 단어를 대상으로 검사 |

**CMD 실행 예시:**
```cmd
:: 1. 모든 전략(기본값) 사용 시 dim 필수 지정
python -m src.postprocessor.python.main -i "SELETC products FRMO ordes" -d 300

:: 2. 런타임 Threshold를 적용하여 보수적인(유사도 0.8 이상) 교정만 허용
python -m src.postprocessor.python.main -i "SELETC products FRMO ordes" -d 100 -t 0.8

:: 3. 단일 전략(Typo)만 사용하도록 변경
python -m src.postprocessor.python.main -i "SELECT usr_name FROM usrs" -s "typo"

:: 4. 50차원 사전 및 다중 전략 지정
python -m src.postprocessor.python.main -i "SELETC products FRMO ordes" -s "typo,semantic" -d 50

:: 5. 특정 DB 파일을 명시적으로 지정
python -m src.postprocessor.python.main -i "SELECT id FROM product" -b "data/shop.db" -d 300
```

### 4.3. 내부 처리 흐름 예시 분석
명령어: `python -m src.postprocessor.python.main -i "SELETC products FRMO ordes" -d 300`

1. **예약어 1차 교정**: 텍스트 레벨에서 `SELETC` $\rightarrow$ `SELECT`, `FRMO` $\rightarrow$ `FROM` 교정
2. **파싱 및 식별자 추출**: `products`, `ordes` 2개의 식별자 감지
3. **Typo 적용**: `ordes`는 편집 거리 내에서 `orders` 테이블명과 유사하므로 바로 치환됨. `products`는 실패.
4. **N-gram 적용**: `products`에 대해 N-gram 시도 실패.
5. **Semantic 적용**: 300차원 압축 사전(`semantic_dict_300d.pkl`) 로드. `products`를 GloVe 벡터로 변환 후 코사인 유사도 검사 $\rightarrow$ `items` 등 의미가 가장 가까운 타겟으로 치환됨.
6. **최종 재조립 반환**: `SELECT items FROM orders`

---

## 5. 실제 실행 결과 (300차원, Threshold 0.5)

다음은 Issue #29 개발 과정에서 `shop.db`를 바탕으로 실제로 **300차원 GloVe 임베딩**과 **0.5 Threshold**를 적용하여 테스트한 일련의 명령어와 출력 결과입니다.

### 5.1. 사전 빌드 결과 (`build_semantic_dict.py`)
```cmd
$ python -m src.postprocessor.python.build_semantic_dict -d 300 -t 0.5
Target dictionary size: 15
Downloading GloVe embeddings (this may take a while)...
Download complete.
Extracting glove.6B.300d.txt...
Deleted downloaded .zip file to save disk space.
Loading GloVe into memory (this may take a minute)...
Computing target vectors...
Filtering GloVe vocabulary using cosine similarity...
Reduced vocabulary size (Threshold >= 0.5): 139
Successfully saved reduced dictionary to C:\Users\issil\Desktop\capstone\capstone-design-1\data\semantic_dict_300d.pkl
Deleted extracted .txt file to save disk space.
```
* **결과 분석**: 데이터베이스 스키마에서 추출된 15개의 타겟 단어(`shop.db`)를 바탕으로 전체 GloVe 단어장 중 유사도 0.5 이상인 단어만 필터링한 결과, 단 **139개**의 단어만 포함된 매우 가벼운 `.pkl` 캐시 파일이 생성되었습니다. 작업 완료 후 `.zip`과 `.txt`는 디스크 공간 확보를 위해 자동 삭제되었습니다.

### 5.2. 후처리기 테스트 결과 (`main.py`)

**1. Typo (편집 거리 기반) 전략 테스트**
* 입력된 SQL 내의 형태적 오타를 `difflib`을 이용해 빠르게 찾아 교정합니다.
```cmd
$ python -m src.postprocessor.python.main -i "SELECT cuetomer_id FROM ordrs" -s typo
SELECT customer_id FROM orders

$ python -m src.postprocessor.python.main -i "SELECT usr_name, emil FROM usrs" -s typo
SELECT user_name, email FROM users
```

**2. N-gram (부분 문자열 패턴) 전략 테스트**
* 단어 길이 차이가 크거나 편집 거리 계산이 모호한 경우, 부분 문자열 집합의 Jaccard 유사도로 교정합니다.
```cmd
$ python -m src.postprocessor.python.main -i "SELECT ctegory FROM itms" -s ngram
SELECT category FROM items

$ python -m src.postprocessor.python.main -i "SELECT manger_id FROM departmnt" -s ngram
SELECT manager_id FROM department
```

**3. Semantic (의미론적 유사도) 전략 테스트**
* 단어의 형태는 완전히 다르지만, 의미론적으로 유사한 단어(GloVe 임베딩 기반)를 찾아 교정합니다.
```cmd
$ python -m src.postprocessor.python.main -i "SELECT products" -s semantic -d 300
SELECT items

$ python -m src.postprocessor.python.main -i "SELECT buyer_id FROM purchases" -s semantic -d 300
SELECT customer_id FROM orders
```

**4. 종합 체이닝 (Typo + N-gram + Semantic) 테스트**
* 위 3가지 전략을 순차적으로 적용하여, Typo로 못 잡은 오류를 N-gram이 잡고, 그래도 못 잡으면 Semantic이 처리하도록 합니다.
```cmd
$ python -m src.postprocessor.python.main -i "SELECT products FROM ordrs" -d 300
SELECT items FROM orders
```

**5. Threshold(임계값)에 의한 교정 실패 예외 케이스**
```cmd
$ python -m src.postprocessor.python.main -i "SELECT client FROM orders" -d 300
SELECT client FROM orders
```

### 5.3. 단위 테스트 결과 (`pytest`)

**테스트 목적 및 항목:**
`tests/test_postprocessor_python.py` 파일 내에는 후처리기 모듈들이 개별적으로 잘 작동하는지 검증하기 위한 8개의 단위 테스트가 작성되어 있습니다.
1. `test_pre_correct_keywords`: 파서 진입 전 SQL 예약어 오타(`SELETC`, `FRMO`) 사전 교정 기능 테스트
2. `test_typo_correction_keywords`: Typo 교정기를 통한 예약어 교정 및 파이프라인 통과 여부 검증
3. `test_typo_correction_identifiers`: `usr_name` $\rightarrow$ `user_name` 등 단순 오타 교정 검증
4. `test_typo_correction_complex`: 여러 개의 식별자 오타가 한 쿼리에 있을 때 모두 교정되는지 검증
5. `test_get_ngrams`: N-gram 분리 함수(예: 2글자씩 쪼개기)가 집합을 정확히 반환하는지 테스트
6. `test_ngram_correction`: `departmnt` $\rightarrow$ `department` 등 N-gram 유사도 교정 검증
7. `test_strategy_chaining`: `typo,ngram` 다중 전략이 주어졌을 때 앞선 전략 실패 시 Fallback이 이루어지는지 테스트
8. `test_semantic_correction_pipeline`: 의미론적 모델을 로드하여 반환하거나, 모델이 없어도 에러 없이 원본을 반환하는 안전성 검증

**테스트 실행 및 결과:**
```cmd
$ .\.venv\Scripts\pytest tests/test_postprocessor_python.py -v
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\issil\Desktop\capstone\capstone-design-1\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\issil\Desktop\capstone\capstone-design-1
configfile: pytest.ini
plugins: anyio-4.15.1, platformdirs-4.12.4
collecting ... collected 8 items

tests/test_postprocessor_python.py::test_pre_correct_keywords PASSED     [ 12%]
tests/test_postprocessor_python.py::test_typo_correction_keywords PASSED [ 25%]
tests/test_postprocessor_python.py::test_typo_correction_identifiers PASSED [ 37%]
tests/test_postprocessor_python.py::test_typo_correction_complex PASSED  [ 50%]
tests/test_postprocessor_python.py::test_get_ngrams PASSED               [ 62%]
tests/test_postprocessor_python.py::test_ngram_correction PASSED         [ 75%]
tests/test_postprocessor_python.py::test_strategy_chaining PASSED        [ 87%]
tests/test_postprocessor_python.py::test_semantic_correction_pipeline PASSED [100%]

============================== 8 passed in 0.17s ==============================
```
