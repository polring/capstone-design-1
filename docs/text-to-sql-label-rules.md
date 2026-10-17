# Text-to-SQL 라벨 규칙

같은 표현에 서로 다른 SQL이 정답으로 붙어 있으면 모델은 다수 쪽을 따라가고, 소수 쪽 문항은 원리상 맞힐 수 없다(1단계 학습
데이터에서 "which item ~"의 정답이 `item_name` 197 / `*` 24로 갈려 있었다). 그래서 **질문만 보고 정답 SQL이 하나로 정해지도록** 단계마다
라벨 규칙을 정한다. 이 문서는 모든 단계의 규칙을 모은 것이다. 원칙은 [계획서](text-to-sql-llm-project-plan.md) 4-5에 있다.

- 앞 단계 규칙은 다음 단계에도 그대로 적용한다. 단계가 늘면 이 문서에 절을 더한다.
- 질문 생성·검증, 학습 데이터 정리, 질문 조합기, 평가 질문의 정답 작성(외부 말투, 사람 작성)에 모두 같은 규칙을 쓴다.
- 학습 데이터는 규칙과 다른 라벨의 쌍을 지운다. 평가셋은 실행끼리 비교하려고 정제하지 않고, 규칙과 다른 문항을 뺀 **규칙 일치 EM**을
  함께 낸다.
- 규칙과 단서 목록은 학습 데이터를 읽고서만 고친다. 평가 오답을 보고 고치지 않는다(규칙 일치 EM이 바뀌므로). 고치면 학습 데이터를 다시
  정제한다.
- 규칙으로 정답을 하나로 정할 수 없는 질문은 **만들지 않음**: 학습 데이터에 넣지 않고, 평가에 나오면 따로 집계해 정확도에서 뺀다.

용어: **식별 컬럼**은 행을 가리키는 이름 컬럼이다 — customers `name`, items `item_name`, orders `order_id`(주문은 이름이 없다).
**자기 ID**는 그 테이블의 기본키다 — customers `customer_id`, items `item_id`.

| 규칙 | 단계 | 다루는 것 | 코드 판정 |
| --- | --- | --- | --- |
| L1 | 1 | 존재 확인 | 1단계 `soft_flags`, 2단계 `drop_reason` |
| L2 | 1 | `*` / 식별 컬럼 | `clean_ambiguous` (1·2단계) |
| L3 | 1 | 자기 ID / 이름 | `clean_ambiguous` (1·2단계) |
| L4 | 1 | 테이블 (orders / customers·items) | `clean_ambiguous` (1·2단계) |
| L5 | 1 | 고객 / 상품 구분 단어 없음 | 판정 없음, 평가에서 따로 집계 |
| L6 | 2 | 비교 표현 → 연산자 | 조합기 표현 사전 |
| L7 | 2 | BETWEEN | 조합기 |
| L8 | 2 | 조건 순서, IN 값 순서 | 조합기, 채점 정규형 |
| L9 | 2 | AND / OR / IN | 조합기 |
| L10 | 2 | 부정 | 조합기 |
| L11 | 2 | 정렬 방향·컬럼 | 조합기 |
| L12 | 2 | 최상급과 LIMIT | 조합기 |
| L13 | 2 | SELECT (속성, 역할 단서) | 조합기 (속성 SELECT는 판정하지 않음) |
| L14 | 2 | 조건이 여러 개인 존재 확인 | 조합기, 2단계 `drop_reason` |
| L15 | 2 | 테이블 | 조합기 (L4 판정) |
| L16 | 2 | 컬럼 단서와 값 표기 | 조합기, 질문 검증의 값 검사 |
| L17 | 2 | 이름 LIKE | 조합기 |

## 1. 1단계 규칙 (L1~L5)

1단계 SQL은 `SELECT` 컬럼 하나 또는 `*`, `WHERE` 등호 조건 하나다.

| 규칙 | 내용 | 예 |
| --- | --- | --- |
| L1 존재 확인 | SELECT 컬럼과 WHERE 컬럼이 같은 SQL은 존재 확인 질문에만 쓴다. 답을 질문에 노출하는 동어반복형("what is the price of the item priced 97.56?")은 쓰지 않는다 | "is there an item priced 97.56?" → `SELECT price … WHERE price = 97.56` |
| L2 `*` / 식별 컬럼 | 질문에 행 전체 단서도 식별 컬럼 단서도 없으면 아래 순서에서 먼저 맞는 것으로 정한다 | |
| | ① WHERE가 이미 식별 컬럼이면 `*` | "henry?" → `SELECT * … WHERE name = 'henry'` |
| | ② which / who 의문문이면 식별 컬럼 | "which item has 29 in stock?" → `item_name` |
| | ③ 복수형 목록 질문이면 `*` | "gold membership customers" → `*` |
| | ④ 조회 동사가 있으면 `*` | "show me the item priced 77.42" → `*` |
| | ⑤ 그 외는 식별 컬럼 | "what item costs 91.52?" → `item_name` |
| L3 ID / 이름 | 자기 ID는 질문이 ID를 요구할 때만 SELECT 한다. WHERE 조건을 가리키는 "item id 3300"은 단서가 아니다 | "which item has 94 left?" → `item_name` (`item_id` 아님) |
| L4 테이블 | orders를 customer_id / item_id로 조회하면서 SELECT가 `*` 또는 WHERE 컬럼이라 customers / items로도 같은 SQL이 성립하면, 질문에 주문 단서가 있어야 orders다. SELECT 컬럼이 orders에만 있으면(`quantity`, `status`) 단서가 없어도 orders다 | "customer 1400 details" → customers, "orders for customer 1400" → orders |
| L5 고객 / 상품 구분 | 질문에 customer / item 같은 단어가 없어도 값 자체로 구분되면 허용한다("chaise - all details"). 실제 사용자가 쓰는 표현이므로 지우지 않고, 평가에서는 따로 집계한다 | |

### 1-1. 단서 목록

코드가 판정에 쓰는 단서다. 단어 경계로 찾고 대소문자는 무시한다. 실제 학습 문장을 읽고 정했으므로 완전하지 않다.

| 단서 | 목록 | 쓰는 규칙 |
| --- | --- | --- |
| 존재 확인 | is there, exist, confirm, verify, check | L1 |
| 행 전체 | all, everything, every, full, whole, complete, entire, detail(s), info, information, record(s), entry, entries, row(s), data, profile, lowdown, rundown, about, overview, summary, anything, column(s), field(s), on file, scoop, deal with, going on | L2 (단서 있으면 `*`) |
| 식별 컬럼 | customers·items: name(s), named, called / orders: order id(s), order number(s), id(s), number(s) | L2 (단서 있으면 식별 컬럼) |
| 의문사 (L2-②) | which, who | L2 |
| 복수 명사 (L2-③) | items, products, orders, customers | L2 |
| 조회 동사 (L2-④) | show, pull up, look up, look like, have on, what's up, what's in, what is in, story, tell me about | L2 |
| ID 요구 | id(s), identifier(s), number(s), # | L3 |
| 주문 | order(s), ordered, ordering, purchase(s), purchased, bought, buy(s), transaction(s), shipment(s), shipped, deliver…, pending, cancel… | L4 |

- WHERE 조건을 말하는 표현은 SELECT 단서로 보지 않는다. "named henry"의 named, orders 질문의 "item id 3155"가 그렇다.
- 행 전체·식별 컬럼 단서 목록은 다중 정답 EM의 `sql_rules.select_ambiguous`에도 쓰인다. 이쪽은 지표를 고정하려고 따로 두고 바꾸지 않는다.

## 2. 2단계 규칙 (L6~L17)

2단계 SQL은 WHERE 조건 0~3개(AND, 서로 다른 컬럼의 OR), 같은 컬럼 값 나열 `IN`, 비교 연산자·`BETWEEN`·이름 `LIKE`, ORDER BY, LIMIT이다.

### L6. 비교 표현과 연산자

비교 표현 하나에 연산자 하나를 정한다. 일상 영어에서 경계가 모호한 표현(under 80이 80을 포함하는가)도 하나로 고정한다.

| 연산자 | 표현 |
| --- | --- |
| `=` | 비교 표현 없음(기본), exactly, priced at {v}, for {v}, at {v}, with {v} in stock, with {v} left |
| `<` | under, below, less than, fewer than, cheaper than, sub |
| `<=` | at most, or less, or fewer, or cheaper, up to, max, maximum, and under, and below, no more than, not over, tops |
| `>` | over, above, more than, greater than, higher than, more expensive than, pricier than, upwards of, north of |
| `>=` | at least, or more, and up, and above, {v}+, min, minimum, starting at, from {v} up, no less than, no fewer than, no cheaper than, not under |
| `!=` | L10 |

- 기호 입력(`price < 20`, `stock >= 5`)은 기호 그대로의 연산자다.
- 컬럼에 묶인 표현은 그 컬럼에만 쓴다. cheaper / more expensive / pricier는 price, fewer / left / in stock은 stock이다.
- 예: "items under $20" → `price < 20.00`, "items $20 or less" → `price <= 20.00`, "items $20+" → `price >= 20.00`.

### L7. BETWEEN

- between a and b, from a to b, a to b, a-b, in the a-b range, anywhere from a to b → `BETWEEN a AND b`. 양 끝을 포함한다.
- SQL은 항상 작은 값을 먼저 쓴다. 질문이 "between 80 and 50"이면 `BETWEEN 50 AND 80`이다.
- 양 끝을 포함하지 않는 범위("over 50 but under 80")는 같은 컬럼 AND가 되는데 2단계 문법은 같은 컬럼 범위를 BETWEEN으로만 쓴다. 만들지 않음.

### L8. 조건 순서

- AND 조건은 질문에 나온 순서와 **무관하게** 스키마 컬럼 순서다. customers `customer_id, name, city, membership`,
  items `item_id, item_name, category, price, stock`, orders `order_id, customer_id, item_id, quantity, status`.
  예: "items under $80 in kitchen"과 "kitchen items under $80"은 모두 `WHERE category = 'kitchen' AND price < 80.00`이다.
- 서로 다른 컬럼의 OR 조건도 스키마 컬럼 순서다. "items under $10 or in kitchen" → `category = 'kitchen' OR price < 10.00`.
- IN의 값은 **질문에 나온 순서**다. "vest or radio" → `item_name IN ('vest', 'radio')`. 순서만 다른 IN은 결과가 같으므로 채점은 IN 값
  순서를 무시한다.
- SQL 생성기 레코드의 `sql`은 IN 값도 값 순서(숫자 오름차순, 텍스트 알파벳순)로 정렬한 정규형이다. 분할·겹침 검사 등 SQL의 정체로만
  쓰고, 학습 쌍의 SQL은 질문에 맞춰 IN 값을 다시 늘어놓는다(`compose.question_order_sql`).
- 이유: 값을 값 순서로 쓰게 하면 모델이 처음 보는 이름을 알파벳순으로 정렬해야 한다. 15M·26M 모델 모두 이를 배우지 못했다.

### L9. AND / OR / IN

- 서로 다른 컬럼의 조건을 나열하면 AND다. 연결어가 없어도("boston gold customers"), and여도, 쉼표여도 같다.
- **같은 컬럼의 값을 나열하면 or로 쓰든 and로 쓰든 IN이다.** "kitchen or sports items", "items in kitchen and sports" 모두
  `category IN ('kitchen', 'sports')`다. 한 행이 두 값을 가질 수 없으므로 and가 목록을 뜻한다. 쉼표·슬래시만으로 잇거나
  ("denver/boston"), 숫자를 공백으로 늘어놓아도("orders 1620 1988") 같다.
  값 나열에 다른 조건이 붙어도 IN + AND다: "gold or silver members in chicago" → `membership IN ('gold', 'silver') AND city = 'chicago'`.
- **서로 다른 컬럼의 조건을 or로 이으면 OR다**: "items in kitchen or under $10" → `category = 'kitchen' OR price < 10.00`.
  값 나열이 한 항이면 IN이 OR 안에 들어간다: "kitchen or sports items, or anything under $10" →
  `category IN ('kitchen', 'sports') OR price < 10.00`.
- IN은 `=` 값 목록에만 쓴다(최대 3개). 같은 컬럼 범위의 OR("under 10 or over 100"), price 값 나열("items priced 10 or 20")은 만들지 않음.
- AND와 OR가 함께 필요한 질문("kitchen items under $10 or over $100", "gold members in boston or anyone in chicago")은 2단계 문법 밖이다. 만들지 않음.

### L10. 부정

- 범주형 값 하나의 부정은 `!=`다. not, not in, not from, other than, except, excluding, besides, outside (of), non-{v},
  that isn't / aren't in, anything but. 예: "non-kitchen items", "items except kitchen" → `category != 'kitchen'`.
- 여러 값의 부정("customers outside boston and denver")은 같은 컬럼 AND가 되므로 만들지 않음.
- 이름·ID·숫자의 부정("items not named kayak", "not priced at 20", "stock is not 0")은 2단계 문법에 없다. 만들지 않음.
- status는 문자 그대로의 부정만 쓴다. "orders whose status is not shipped" → `status != 'shipped'`. 상태에 순서가 있어
  "orders that haven't shipped yet"은 실제로 pending을 뜻하므로 만들지 않음.

### L11. 정렬

| 컬럼 | ASC | DESC |
| --- | --- | --- |
| price | cheapest first, low to high, ascending, from cheapest to most expensive | most expensive first, high to low, descending, priciest to cheapest |
| stock | lowest stock first, fewest left first, low to high | highest stock first, most left first, high to low |
| quantity | smallest orders first | largest orders first |

- 방향 없는 "sorted by price", "order by stock"은 ASC다. top, highest, most, largest는 DESC다.
- 정렬 컬럼은 숫자 컬럼(price, stock, quantity)만이다. 이름·알파벳 정렬은 만들지 않음.
- 정렬 컬럼은 SELECT 컬럼과 달라도 된다: "names of kitchen items, cheapest first" → `SELECT item_name … ORDER BY price ASC`.
- 정렬 컬럼 단서가 없는 질문("the 3 best items", 컬럼이 여럿인 문장의 "order descending")은 만들지 않음.

### L12. 최상급과 LIMIT

- 단수 최상급은 `LIMIT 1`이다: "the cheapest kitchen item", "which item costs the most" → `ORDER BY … LIMIT 1`.
- 숫자가 붙은 최상급과 top n은 `LIMIT n`이다: "the 3 cheapest items", "top 5 items by price" → `LIMIT 3`, `LIMIT 5`. n은 2~10.
- 정렬 표현 뒤에 붙은 개수도 LIMIT이다: "sports items, lowest stock first, just 10", "…, cheapest first, only 3" → `ORDER BY … LIMIT n`.
- LIMIT 숫자와 조건 숫자가 같아도 쓴다("the 10 cheapest items over $10" → `price > 10.00 … LIMIT 10`).
- 숫자 없는 복수 최상급("the cheapest items")은 LIMIT이 정해지지 않는다. 만들지 않음.
- ORDER BY 없는 LIMIT("show me 5 kitchen items")은 2단계 문법에 없다. 만들지 않음.
- 동점이 있으면 `LIMIT 1`은 그중 하나만 돌려준다. 동점 전부는 서브쿼리가 필요한 3단계 범위다.

### L13. SELECT

- SELECT는 컬럼 하나 또는 `*`다. 여러 컬럼("names and prices of …")은 만들지 않음.
- `*` / 식별 컬럼 / 자기 ID 선택은 조건이 여러 개여도 L2·L3를 그대로 쓴다. 조합기 틀의 역할이 이 단서를 정한다.

  | 역할 | 단서 | SQL |
  | --- | --- | --- |
  | star | show, list, find, pull up, look up, details, everything about, full records | `*` |
  | ident | which, name(s), called | 식별 컬럼 |
  | own_id | id(s), id number | 자기 ID |
  | 속성 | the prices of, how much … cost, stock levels of, what category … in | 그 컬럼 |

- find는 star 단서다("find the item that costs 76.40" → `*`).
- 복수형 속성 명사("the prices of", "the quantities of")는 그 컬럼의 SELECT다.
- 속성 명사가 있으면 L2 단서보다 우선한다. "which category does … belong to"는 `category`, "show the prices of …"는 `price`다.
  which / show는 `*`와 식별 컬럼 사이를 고를 때만 단서다.
- 개수("how many items cost under 80")는 `COUNT`가 필요한 3단계 범위다. "how many … in stock"(stock), "how many units"(quantity)처럼
  컬럼 값을 묻는 표현은 그 컬럼의 SELECT다. 단, 행이 하나로 정해질 때(단수)만 쓴다. 여러 주문에 대한 "how many units were ordered in
  the orders of customer 1452"는 합계(`SUM`)로 읽히므로 만들지 않음.

### L14. 존재 확인

- L1 그대로: SELECT가 `=` 조건 컬럼인 SQL은 조건 1개, 정렬 없음일 때만 쓰고 존재 확인 질문으로만 만든다.
- 존재 확인 단서: is there, are there any, do we have, do we carry, does … exist, any … at all (1단계 단서에 더함).
- L1을 쓸 수 없는 존재 확인, 곧 조건이 여러 개이거나 조건 하나가 `=`가 아닌 존재 확인은 **식별 컬럼을 SELECT** 한다. 정렬·LIMIT은 붙이지 않는다.
  - "do we have any furniture under $40?" → `SELECT item_name FROM items WHERE category = 'furniture' AND price < 40.00`
  - "are there any items starting with ka?" → `SELECT item_name FROM items WHERE item_name LIKE 'ka%'`
  - "do we have any kitchen items?"(= 조건 하나)는 그대로 L1 → `SELECT category FROM items WHERE category = 'kitchen'`
- 그래서 같은 "do we have any …" 문형이라도 `=` 조건 하나면 조건 컬럼, 그 밖이면 식별 컬럼이다. L1을 바꾸면 1단계 학습·평가 라벨
  수백 개를 다시 정제해야 해서 L1은 그대로 두었다.
- 존재 확인 질문으로 속성 SQL(예: `SELECT membership …`)을 묻지 않는다.

### L15. 테이블

- L4 그대로: orders는 주문 단서가 있거나 SELECT·조건에 orders 전용 컬럼(quantity, status)이 있을 때만이다.
  "is there customer id 1317"은 customers, "is there an order of customer 1317"은 orders다.
- 조건이 여러 개면 고객 단서와 주문 단서가 함께 나올 수 있다("pending orders of customer 1452"). 주문 단서가 하나라도 있으면 orders다.
- stock(items)과 quantity(orders) 구분: left, in stock, available, on hand는 stock, ordered, units ordered, quantity는 quantity다.
- JOIN이 필요한 질문("orders of customers from boston")은 4단계 범위다.

### L16. 컬럼 단서와 값 표기

- 생성 데이터의 숫자 비교에는 항상 컬럼 단서를 넣는다. price는 $, dollars, bucks, priced, cost, cheaper, expensive이고, stock은 in stock,
  left, available, on hand, stock이다. 단서 없는 "items under 80"은 만들지 않음. 평가에 나오면 price로 정답을 달고 따로 집계한다.
- 값 표기:
  - price: $20, 20, 20.00, 20 dollars, 20 bucks는 모두 `20.00`이다. SQL은 항상 소수 둘째 자리다. 소수 한 자리("20.0")는 쓰지 않는다.
  - 정수 컬럼은 숫자 그대로다.
  - 텍스트 값은 따옴표 유무와 관계없이 소문자 값 그대로다("kayak", 'kayak', kayak → `'kayak'`).
  - 숫자를 영어 단어로 쓴 것("twenty")은 범위 밖이다.
- ID 컬럼은 `=`·IN만 쓴다. 비교 연산("customer id over 1500")은 만들지 않음.
- 고정값 표현:

  | 표현 | SQL |
  | --- | --- |
  | out of stock, sold out, with none left, with no stock | `stock = 0` |
  | in stock, available, still in stock, that aren't sold out (조건으로 쓸 때) | `stock > 0` |
  | low stock, running low (기준 숫자 없음) | 만들지 않음 |

  `stock >= 1`은 `stock > 0`과 결과가 같지만 질문 표현으로는 숫자로만("with at least 1 in stock") 쓴다.

### L17. 이름 LIKE

| 패턴 | 표현 |
| --- | --- |
| `'x%'` 접두 | starts with, starting with, begins with, beginning with, name starting with |
| `'%x'` 접미 | ends with, ending in, ending with, name ending in |
| `'%x%'` 포함 | contains, containing, with x in the name, with names that include x |

- 이름 컬럼(customers `name`, items `item_name`)에만 쓴다. 조각은 질문의 따옴표를 벗긴 소문자 그대로다.
- 와일드카드 기호로 방향을 표시한 것("\*isc\*", "%tto%", "\*vet")도 그 방향의 LIKE다.
- 방향이 없는 "items like ka", "ka-something items"는 만들지 않음.
- 조각이 이름 전체와 같으면 `=`가 되므로 생성기는 그런 조각을 쓰지 않는다.

## 3. 판정 코드와 적용 위치

### 3-1. 1단계 판정

| 규칙 | 코드 | 적용 대상 |
| --- | --- | --- |
| L1 | 질문 생성 프롬프트 규칙 10, `src/data/stage1/question_gen.soft_flags()` | LLM 질문 생성 라운드 검수 |
| L2~L4 | `src/data/stage1/clean_ambiguous.py` (`drop_reason`) | 학습 데이터: 규칙과 다른 라벨의 쌍을 지운다 (`--apply`) |
| L2~L4 | `src/data/stage1/sql_rules.label_mismatch` (같은 판정) | 평가: 규칙과 다른 문항을 뺀 규칙 일치 EM |
| L5 | — | 평가에서 따로 집계 |

### 3-2. 2단계 판정

`src/data/stage2/clean_ambiguous.py`의 `drop_reason(pair)`: 질문과 정답 SQL만 보고 정답이 라벨 규칙과 어긋나는지 정한다. LLM 질문 검증
(`question_gen validate`), 1단계 학습 쌍 사본(`compose build`), 조합기 검사(`compose check`), 2단계 규칙 일치 EM(`sql_rules.label_mismatch`)에 쓴다.

| 사유 | 규칙 | 판정 |
| --- | --- | --- |
| L1 | 존재 확인 | SELECT가 `=` 조건 컬럼(조건 하나)인데 존재 확인 단서가 없다 |
| L2 | `*` / 식별 컬럼 | 둘 다 단서가 없을 때 1단계 순서(①~⑤)와 다르다 |
| L3 | 자기 ID | item_id(items)·customer_id(customers)를 SELECT하는데 ID 단서가 없다 |
| L4 | 테이블 | customers / items로도 성립하는 orders SQL인데 주문 단서가 없다 |
| L14 | 존재 확인 | 질문 첫머리가 존재 확인인데(행 전체 단서 없이) 정답이 `*`다 |

- 단서는 1단계 목록(1-1)에 L13 star 단서(list, find), 조합기의 유사어(display, dig up, retrieve, enumerate, send me, fill me in …)·복수 명사
  (things, clients, purchases …)·존재 확인 틀(check whether there are, do any … exist, got any)을 더했다. 문장 첫머리 감탄사 뒤의 의문사
  ("hey, who're the silver members?", "um, which …")와 부사가 낀 존재 확인("do we even / really have")도 단서다.
- WHERE 조건을 말하는 표현("customer 1102", "named amy", ID 조건 값 앞의 id)은 단서로 보지 않는다. which / who는 의문문일 때만 단서다
  ("shoppers who live in …"의 관계대명사는 아니다).
- **속성 SELECT(L13)는 판정하지 않는다.** 속성을 컬럼 이름 없이 묻는 말("where does order 5017 stand", "what's a stopwatch going for")이
  학습 문장에도 많아, 단서 목록으로 판정하면 맞는 문항을 버린다(버리면 점수가 부푼다). 그래서 존재 확인으로 속성 SQL을 묻는 문장(L14)이나
  단서 없이 속성 SQL에 붙은 문장은 판정기를 통과한다. 사람 작성 문항은 직접 확인한다.
- 조합기 문장은 분할마다 0~6개(0.005% 이하)만 걸리고, `compose check`는 0.1%를 넘으면 실패한다.
- 1단계 학습 쌍 중 2단계 규칙과 어긋난 123개는 2단계 학습 사본에서 뺀다(목록 `data_raw/stage2/stage1_pairs_removed.json`):
  "find the item that costs 76.40" → `item_name` 122개(L13에서 find를 star로 정함)와 관계대명사 who를 의문사로 본 1개.
