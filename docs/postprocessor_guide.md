# Python SQL 후처리기(Post-processor) 가이드

본 문서는 Issue #29를 통해 구현된 **Python 기반 SQL 후처리기**의 동작 원리, 알고리즘, 그리고 구성 파일의 역할을 설명합니다.

---

## 1. 추가된 파일 및 모듈 역할

이번 구현을 통해 `src/postprocessor/python/` 및 `tests/` 폴더에 다음 파일들이 추가되었습니다.

* **`src/postprocessor/python/__init__.py`**
  * 모듈 초기화 파일로, 외부에서 `process_sql` 함수를 쉽게 임포트할 수 있게 해줍니다.
* **`src/postprocessor/python/main.py`**
  * CLI(Command Line Interface) 실행을 위한 진입점(Entry point)입니다. 
  * `--input`과 `--strategy` 파라미터를 받아 후처리 파이프라인을 실행하고 결과를 터미널에 출력합니다.
* **`src/postprocessor/python/parser.py`**
  * **입력 SQL 파싱 및 식별자 추출**을 담당합니다.
  * `sqlglot` 라이브러리를 사용하며, 예약어 오타로 인한 파싱 에러를 막기 위해 AST 생성 전 `pre_correct_keywords` 함수로 예약어(Keywords)를 1차 교정합니다.
* **`src/postprocessor/python/replacement.py`**
  * **단어 교체 알고리즘(Typo, N-gram, Semantic)**과 **전략 파이프라인(Chaining)**을 포함하고 있습니다.
  * 파싱된 AST에서 추출한 식별자들(테이블명, 컬럼명)을 사전에 정의된 도메인 단어(Dictionary)와 비교해 적절하게 교정합니다.
* **`src/postprocessor/python/reconstruction.py`**
  * 교정이 완료된 AST 객체를 다시 **유효한 SQL 문자열로 재생성**하여 반환하는 역할을 합니다.
* **`tests/test_postprocessor_python.py`**
  * `pytest`를 활용하여 각 파이프라인(예약어 교정, 식별자 교정, Chaining 등)이 정상 작동하는지 검증하는 단위 테스트 파일입니다.

---

## 2. 후처리기 각 단계 및 알고리즘 설명

### 알고리즘 종류
후처리기는 상황에 맞춰 단어를 교정할 수 있도록 3가지 알고리즘 전략을 지원하며, 콤마(`,`)를 이용해 여러 개를 중복 적용(Chaining)할 수 있습니다.

1. **Typo (편집 거리 기반 교정)**
   * 파이썬 내장 모듈(`difflib`)을 활용하여 문자열 간의 삽입, 삭제, 변경 최소 횟수를 계산합니다.
   * `usr_name` $\rightarrow$ `user_name` 처럼 1~2글자의 오타를 매우 빠르고 직관적으로 잡아냅니다.
2. **N-gram (부분 문자열 Jaccard 유사도)**
   * 단어를 N개(기본 2~3개)의 문자로 쪼갠 집합을 만들어 두 집합 간의 **Jaccard 유사도(교집합/합집합)**를 계산합니다.
   * `departmnt` $\rightarrow$ `department` 등 길이가 길거나 편집 거리가 애매한 경우 형태적 유사성을 통해 비교적 정확하게 교정합니다.
3. **Semantic (의미론적 유사도) - *현재 Placeholder 상태***
   * 벡터 임베딩을 이용해 형태가 완전히 달라도 의미가 같은 단어(예: `employee` $\rightarrow$ `user`)를 찾아내는 기법입니다.
   * *(참고: 현재는 무거운 로컬 모델 다운로드를 방지하기 위해 빈 뼈대로만 구성되어 있으며, 실제 교정 연산을 수행하지는 않습니다.)*
4. **전략 Chaining (다중 적용)**
   * `--strategy typo,ngram`을 넘겨주면, 식별자에 대해 **Typo 로직을 먼저 시도**하고, 만약 교정되지 않으면 **N-gram 로직을 이어서 시도(Fallback)**하는 방식으로 상호 보완적으로 동작합니다.

---

## 3. 잘못된 SQL 교정 과정 예시 (Step-by-Step)

사용자가 다음과 같이 심하게 훼손된 SQL을 입력했다고 가정해 봅시다.

> **Input SQL**: `SELETC manger_id FRMO departmnt`

### Step 1: 예약어 1차 교정 (Pre-correction)
* **모듈**: `parser.py` (`pre_correct_keywords`)
* `sqlglot`은 유효하지 않은 SQL 키워드를 만나면 파싱 에러를 뱉습니다. 
* 따라서 문자열을 공백 단위로 쪼개어 SQL 예약어(Keywords) 리스트와 편집 거리를 비교합니다.
* `SELETC` $\rightarrow$ `SELECT`, `FRMO` $\rightarrow$ `FROM` 으로 사전 교정됩니다.
* **중간 결과**: `SELECT manger_id FROM departmnt`

### Step 2: AST 파싱 및 식별자 추출 (Parsing)
* **모듈**: `parser.py` (`extract_identifiers`)
* 1차 교정된 텍스트를 `sqlglot`에 넣어 추상 구문 트리(AST)로 파싱합니다.
* 트리를 순회하며 교정이 필요한 사용자 정의 식별자(Identifier) 노드들만 추출합니다.
* **추출된 식별자 리스트**: `[manger_id, departmnt]`

### Step 3: 단어 치환 및 교정 (Replacement)
* **모듈**: `replacement.py` (`correct_identifiers`)
* 준비된 도메인 사전(Dictionary = `['manager_id', 'department', ...]`)과 입력된 전략(예: `typo,ngram`)을 기반으로 교정을 시도합니다.
* `manger_id`: Typo 전략을 적용하여 `manager_id`와 매우 유사함을 파악하고 치환합니다.
* `departmnt`: Typo 전략에서 실패할 경우(허용 오차 밖), 이어서 N-gram 전략을 타게 되며 문자열 유사도를 바탕으로 `department`로 치환됩니다.
* 노드의 속성값이 교정된 단어로 덮어씌워집니다.

### Step 4: SQL 재조립 (Reconstruction)
* **모듈**: `reconstruction.py` (`reconstruct_sql`)
* 단어 교체가 끝난 AST 객체를 다시 SQL 문자열로 변환(Transpile)합니다.
* 이 과정에서 `sqlglot`에 의해 기본적인 SQL 포맷팅(대소문자 정리 등)이 적용될 수 있습니다.

### 최종 반환 결과
> **Output SQL**: `SELECT manager_id FROM department`
