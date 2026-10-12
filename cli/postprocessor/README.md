# SQL Post-processor C 구현 가이드

학습 저장소의 Python 구현체(`src/postprocessor/python`)와 같은 알고리즘·동작을 하도록 C 언어로 작성된 모듈(`cli/postprocessor/`)에 대한 설계 및 사용 설명서입니다.

C 프로젝트에서 헤더를 인클루드하여 정적/동적 라이브러리로 링크해 사용할 수 있으며, CLI 바이너리로 빌드하여 단독 실행할 수도 있습니다.

---

## 1. 아키텍처 및 모듈 구성

모듈은 `cli/postprocessor/` 디렉터리에 위치하며 다음 파일들로 구성되어 있습니다:

| 파일명 | 역할 설명 |
| :--- | :--- |
| [`postprocessor.h`](postprocessor.h) | 외부 공개용 C 라이브러리 인터페이스 헤더 |
| [`core.h`](core.h) / [`core.c`](core.c) | 전체 교정 파이프라인(`process_sql`) 제어 및 오케스트레이션 |
| [`parser.h`](parser.h) / [`parser.c`](parser.c) | SQL 예약어 사전 보정(`pre_correct_keywords`), 토크나이저 및 SQL 재구성기 |
| [`replacement.h`](replacement.h) / [`replacement.c`](replacement.c) | Typo(difflib 비율), N-gram(Jaccard 유사도), Semantic(임베딩 코사인 유사도) 교정기 및 전략 체이닝 |
| [`schema.h`](schema.h) / [`schema.c`](schema.c) | SQLite 데이터베이스로부터 테이블 및 컬럼 스키마 사전 로더 |
| [`main.c`](main.c) | CLI 단독 실행용 진입점 (명령줄 인수 파싱 및 실행) |
| [`test_postprocessor.c`](test_postprocessor.c) | `tests/test_postprocessor_python.py`의 모든 테스트 케이스를 1:1 검증하는 C 단위 테스트 |
| [`Makefile`](Makefile) | 정적 라이브러리(`libpostprocessor.a`), CLI 바이너리, 테스트 바이너리 자동 빌드 스크립트 |

---

## 2. 알고리즘 1:1 대응 원리

1. **SQL 예약어 사전 보정 (`pre_correct_keywords`)**:
   - `SELECT`, `FROM`, `WHERE` 등 표준 SQL 예약어 리스트(`PRIMARY_KEYWORDS`)를 대상으로 `difflib.SequenceMatcher`와 동일한 Gestalt Pattern Matching 알고리즘을 수행하여 유사도 $\ge 0.7$일 때 오타를 사전 교정합니다 (예: `SELETC` $\rightarrow$ `SELECT`, `FRMO` $\rightarrow$ `FROM`).
   - 원래 토큰의 대소문자(전체 소문자 여부)를 유지합니다.

2. **식별자 토큰 추출 및 교정**:
   - 쿼리를 단어, 문자열 리터럴, 기호, 공백 등으로 분해하여 키워드가 아닌 단어를 식별자(테이블/컬럼명)로 인식합니다.
   - 원본 대소문자 속성(`is_all_upper`)을 기록하여 교정 후 원래 대소문자에 맞게 복원합니다.

3. **교정 알고리즘 (전략 및 체이닝)**:
   - **Typo**: Python `difflib.get_close_matches(word, dictionary, cutoff=0.7)`와 동일한 문자열 일치 비율($2M / (|A| + |B|)$)로 가장 유사한 사전 단어를 탐색합니다.
   - **N-gram**: 2-gram 부분 문자열 집합을 추출하고 Python의 `set.intersection` 및 `set.union`과 동일하게 Jaccard 유사도($|A \cap B| / |A \cup B|$)를 계산(cutoff $\ge 0.3$)합니다. 단어 길이가 $N$ 미만이면 Typo로 Fallback합니다.
   - **Semantic**: Snake_case로 분리된 단어들의 임베딩 벡터를 Mean-pooling한 뒤 코사인 유사도로 타겟 사전을 검색합니다. 캐시가 없으면 파이프라인 오류 없이 안전하게 원래 단어를 반환합니다.
   - **전략 체이닝**: `typo,ngram,semantic`처럼 쉼표로 연결된 전략 순서대로 교정을 시도하며, 먼저 교정에 성공한 전략이 나오면 즉시 중단합니다.

---

## 3. 빌드 방법

### 라이브러리 및 CLI 빌드
```bash
cd cli/postprocessor   # 또는 저장소 루트에서 make -C cli postprocessor
make
```

빌드 산출물:
* `libpostprocessor.a`: 다른 C 애플리케이션에서 링크 가능한 정적 라이브러리
* `postprocessor.exe` (Linux 환경: `postprocessor`): 단독 실행 가능한 CLI 바이너리

### C 단위 테스트 실행
```bash
make test
```
`test_postprocessor.c`가 컴파일되어 실행되며, 파이썬 pytest와 동일한 7개 시나리오(예약어 보정, 식별자 보정, 복합 오타, n-gram, 체이닝, semantic fallback)를 모두 검증합니다.

---

## 4. 라이브러리로 사용하기 (`#include "postprocessor.h"`)

다른 C 프로그램에서 라이브러리로 임포트하여 사용할 때는 `postprocessor.h`를 인클루드하고 `libpostprocessor.a`를 링크합니다.

### C 코드 예제:
```c
#include "postprocessor.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* 1. 스키마 사전 정의 (또는 load_dictionary_from_sqlite 사용) */
    const char *dictionary[] = {
        "users", "user_name", "email", "id", "name", "department", "manager_id"
    };
    size_t dict_size = sizeof(dictionary) / sizeof(dictionary[0]);

    /* 2. 교정할 SQL 입력 (예약어 및 컬럼 오타 포함) */
    const char *input_sql = "SELETC usr_name, emil FRMO users WHERE manger_id = 1";

    /* 3. process_sql 호출 (전략: typo,ngram) */
    char *fixed_sql = process_sql(input_sql, "typo,ngram", dictionary, dict_size, 0, 0.0);

    printf("교정 결과: %s\n", fixed_sql);
    /* 출력: SELECT user_name, email FROM users WHERE manager_id = 1 */

    /* 4. 메모리 해제 */
    free(fixed_sql);
    return 0;
}
```

### 컴파일 명령:
```bash
gcc -std=c11 -O2 my_code.c -Icli/postprocessor cli/postprocessor/libpostprocessor.a -lm -o my_code
```

---

## 5. CLI 단독 실행 도구로 사용하기

빌드된 `postprocessor` 바이너리는 Python의 `main.py`와 동일한 옵션을 제공합니다.

### 명령어 옵션
```text
Usage:
  postprocessor -i <sql_string> [options]

Options:
  -i, --input <sql>          교정할 입력 SQL 쿼리 문자열 (필수)
  -s, --strategy <strats>    교정 전략 (기본값: typo,ngram,semantic)
  -b, --db <path>            스키마 사전을 추출할 SQLite 데이터베이스 경로
  -d, --dim <50|100|200|300> Semantic 교정에 사용할 GloVe 차원
  -t, --threshold <float>    Semantic 교정 코사인 유사도 임계값 (기본값: 0.0)
  -h, --help                 도움말 출력
```

### 실행 예시
```bash
# 1. 기본 실행 (-b 를 생략하면 실행 위치의 data/shop.db 를 찾는다. semantic_dict_<dim>d.bin 도 실행 위치의 data/ 에서 찾는다)
./postprocessor -i "SELETC usr_name, emil FRMO users" -b "data/shop.db"

# 2. 특정 전략만 지정하여 실행 (N-gram)
./postprocessor -i "SELECT id FROM departmnt" -s "ngram" -b "data/shop.db"

# 3. GloVe 차원 지정하여 Semantic 전략 포함 실행
./postprocessor -i "SELECT usr FROM users" -s "typo,ngram,semantic" -d 300 -t 0.5 -b "data/shop.db"
```

---

## 6. 벡터 임베딩(Semantic) 데이터 연동 방법

C 언어는 Python 전용 객체 직렬화 규격인 `.pkl` 파일을 직접 언패킹하지 못하므로, 기존 `data/semantic_dict_<dim>d.pkl`을 C 고속 바이너리 규격(`.bin`)으로 1회 변환하여 연동합니다:

### 1단계: pkl 데이터를 C 호환 바이너리로 내보내기 (1회 실행)
```powershell
python -m src.postprocessor.python.export_embeddings -d 300
```
* `data/semantic_dict_300d.pkl`의 GloVe 임베딩 벡터들이 고속 바이너리 포맷인 `data/semantic_dict_300d.bin`으로 추출됩니다.

### 2단계: C postprocessor에서 의미론적(Semantic) 유사도 교정 실행
```powershell
.\postprocessor.exe -i "SELECT client FROM customers" -b "data\shop.db" -s "semantic" -d 300 -t 0.5
```
* `shop.db`의 `customers` 테이블 스키마에서 컬럼/테이블명을 읽어온 후, `client` 단어와 `customers` 간 코사인 유사도를 계산하여 교정합니다.

