# Python SQL 후처리기(Post-processor) 가이드

본 문서는 Issue #29를 통해 구현된 **Python 기반 SQL 후처리기**의 동작 원리, 알고리즘, 그리고 구성 파일의 역할을 설명합니다. 
최근 업데이트를 통해 동적 스키마 로딩(`shop.db`)과 순수 `numpy` 및 `GloVe` 기반의 의미론적 교정(Semantic Correction) 기능이 추가되었습니다.

---

## 1. 추가된 파일 및 모듈 역할

이번 구현을 통해 `src/postprocessor/python/` 및 `tests/` 폴더에 다음 파일들이 구성되었습니다.

* **`__init__.py`**
  * 모듈 초기화 파일로, 외부에서 `process_sql` 및 `load_dictionary_from_sqlite` 함수를 쉽게 임포트할 수 있게 해줍니다.
* **`main.py`**
  * CLI(Command Line Interface) 실행을 위한 진입점(Entry point)입니다. 
  * `--input`, `--strategy`, `--db`, `--dim` 파라미터를 받아 후처리 파이프라인을 실행합니다.
* **`schema.py`**
  * 지정된 SQLite 데이터베이스(기본값: `data/shop.db`)에서 테이블명과 컬럼명을 동적으로 추출하여 교정 사전을 구축하는 모듈입니다.
* **`parser.py`**
  * **입력 SQL 파싱 및 식별자 추출**을 담당합니다.
  * `sqlglot` 라이브러리를 사용하며, 예약어 오타로 인한 파싱 에러를 막기 위해 AST 생성 전 텍스트 레벨에서 예약어를 1차 교정합니다.
* **`replacement.py`**
  * **단어 교체 알고리즘(Typo, N-gram, Semantic)**과 **전략 파이프라인(Chaining)**을 수행합니다.
  * 추출된 식별자들을 `schema.py`에서 얻어온 도메인 사전(Dictionary)과 비교하여 적절하게 교정합니다.
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
   * 순수 `numpy` 행렬 연산과 스탠포드 `GloVe` 워드 임베딩을 사용하여, 문자 형태가 전혀 달라도 **의미상 가장 유사한 단어**(예: `client` $\rightarrow$ `customers`)로 교정합니다.
   * **스네이크 케이스 및 평균 풀링(Mean Pooling)**: DB의 `customer_id` 같은 컬럼명을 `['customer', 'id']`로 쪼갠 뒤 각각의 GloVe 벡터를 합산하고 평균을 내어 기준 벡터를 만듭니다. 입력된 단어 역시 쪼개서 평균 풀링 후 코사인 유사도를 계산합니다.
4. **전략 Chaining (다중 적용 및 Fallback)**
   * `--strategy typo,ngram,semantic` 처럼 넘겨주면 가장 가벼운 Typo부터 시도하고, 교정에 실패하면 다음 알고리즘으로 넘어가는(Fallback) 파이프라인 구조로 작동합니다.

---

## 3. 실행 방법 및 예시

### 3.1. (최초 1회) 의미론적 사전 캐시 생성
GloVe 데이터 연산을 가볍게 하기 위해 오프라인 상태에서 미리 사전을 구축합니다. 
`--dim` 옵션으로 사용할 임베딩 차원(50, 100, 200, 300)을 지정할 수 있습니다.
```bash
python -m src.postprocessor.python.build_semantic_dict --dim 100
```
> 완료되면 `data/semantic_dict_100d.pkl` 파일이 생성됩니다.

### 3.2. 런타임 SQL 후처리기 실행
`--db` 인자를 생략하면 자동으로 `data/shop.db`를 읽어 스키마를 구성합니다. 
`--dim`은 `semantic` 전략이 켜져있을 때만 사용되며, 사전 구축 시 만들었던 특정 차원의 `.pkl` 캐시를 로딩합니다.

> **Input SQL**: `SELETC client FRMO ordes`

```bash
python -m src.postprocessor.python.main --input "SELETC client FRMO ordes" --strategy "typo,ngram,semantic" --dim 100
```

**내부 처리 과정:**
1. **예약어 1차 교정**: 텍스트 레벨에서 `SELETC` $\rightarrow$ `SELECT`, `FRMO` $\rightarrow$ `FROM` 교정
2. **파싱 및 식별자 추출**: `client`, `ordes` 2개의 식별자 감지
3. **Typo 적용**: `ordes`는 편집 거리 내에서 `orders` 테이블명과 유사하므로 바로 치환됨. `client`는 실패.
4. **N-gram 적용**: `client`에 대해 N-gram 시도 실패.
5. **Semantic 적용**: 100차원 압축 사전(`semantic_dict_100d.pkl`) 로드. `client`를 GloVe 벡터로 변환 후 코사인 유사도 검사 $\rightarrow$ `customers` 또는 `customer_id` 등 의미가 가장 가까운 타겟으로 치환됨.
6. **최종 재조립 반환**: `SELECT customer_id FROM orders`
