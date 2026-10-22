# Text-to-SQL 라벨 규칙

현재 3단계 구현 범위와 주요 정책은 [3단계 보고서](text-to-sql-stage3-report.md#rule-review)에 요약했다.
의미 규칙·샘플링 설정·현재 AST 구현 한계를 분리했으며, 아래 1·2단계 제외 규칙을 3단계에 자동 전파하지 않는다.

같은 표현에 서로 다른 SQL이 정답으로 붙어 있으면 모델은 다수 쪽을 따라가고, 소수 쪽 문항은 원리상 맞힐 수 없다(1단계 학습
데이터에서 "which item ~"의 정답이 `item_name` 197 / `*` 24로 갈려 있었다). 그래서 **질문만 보고 정답 SQL이 하나로 정해지도록** 단계마다
라벨 규칙을 정한다. 이 문서는 모든 단계의 규칙을 모은 것이다. 원칙은 [계획서](text-to-sql-llm-project-plan.md) 4-5에 있다.

- 앞 단계 규칙은 다음 단계에도 적용하되, 새 단계에서 명시적으로 확장·변경한 항목은 해당 단계 규칙을 우선한다. 과거 단계 데이터와 평가 라벨은 소급 변경하지 않는다.
- 질문 생성·검증, 학습 데이터 정리, 질문 조합기, 평가 질문의 정답 작성(외부 말투, 사람 작성)에 모두 같은 규칙을 쓴다.
- 학습 데이터는 규칙과 다른 라벨의 쌍을 지운다. 평가셋은 실행끼리 비교하려고 정제하지 않고, 규칙과 다른 문항을 뺀 **규칙 일치 EM**을
  함께 낸다.
- 규칙과 단서 목록은 학습 데이터를 읽고서만 고친다. 평가 오답을 보고 고치지 않는다(규칙 일치 EM이 바뀌므로). 고치면 학습 데이터를 다시
  정제한다.
- 여러 해석이 가능한 표현은 **단계별 기본 해석을 하나 정해** 학습·평가에 동일하게 적용한다. 질문에 명시적 단서가 있으면 기본값보다 우선한다.
  기본값은 생략된 의미를 보완하기 위한 것이며, 명시된 조건을 지우거나 지원 범위 밖 요청을 다른 의미로 바꾸는 데 쓰지 않는다.
  기준 날짜·비교 대상 등 필수 정보가 없고 기본값도 정해져 있지 않으면 미정의로 분류한다. 실제 지원 범위를 넘는 요청은 범위 밖으로 분류한다.

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
| L18 | 3 | 규칙 우선순위·지원 범위 | 정책 확정, 코드 반영 대조 필요 |
| L19 | 3 | 행 개수 / 수량 합계 / 평균 / 최솟값·최댓값 | 신규 의미 검증 필요 |
| L20 | 3 | 집계 컬럼·단위·기본값 | 신규 의미 검증 필요 |
| L21 | 3 | 그룹 키·집계 출력 | 일부 AST 구현, 기본 해석 검증 필요 |
| L22 | 3 | WHERE / HAVING 조건의 범위 | 일부 AST 구현, 의미 검증 필요 |
| L23 | 3 | 서브쿼리 기준 집단 | 일부 AST 구현, 의미 검증 필요 |
| L24 | 3 | 최상급·정렬·LIMIT·동점 | 기본 해석·보조 정렬 일부 코드 변경 필요 |
| L25 | 3 | 날짜 비교·기간·상대 날짜 | 날짜 SQL 구현, 표현 정책 대조 필요 |
| L26 | 3 | SQL 정규형·숫자·빈 결과 | 일부 구현, 생성 분포 대조 필요 |
| L27 | 3 | 이름·LIKE·기존 모호 표현 | 두 단어 이름 구현, 일부 정책 변경 필요 |
| L28 | 3 | LLM 질문 작성·채택 | 배치 생성 구현, 의미 검증 미완료 |
| L29 | 3 | 분할·holdout·평가·정책 변경 | 일부 구현, 적용 기록 필요 |

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

## 3. 3단계 규칙 (L18~L29)

2026-10-14 정책. 아래는 **3단계 데이터의 정답을 정하는 기준**이다. 현재 코드가 모두 준수한다는 뜻은 아니다.
새 정책과 기존 생성물의 차이는 3-13에 기록한다. 질문 생성, 조합기, 검증기, 사람 라벨링은 같은 기준을 사용한다.

### L18. 우선순위와 지원 범위

해석 순서는 **명시적 요청 → 3단계 기본값 → 기존 L1~L17**이다. 모호한 표현을 일괄 제외하지 않고 아래 기본값으로 정답을 단다.
단, 기본값을 이유로 명시적인 요청을 무시하지 않는다. 예를 들어 “동점 모두”를 LIMIT 1로 바꾸면 안 된다.

| 영역 | 3단계 범위 |
| --- | --- |
| 일반 행 조회 | 2단계 문법 + 날짜, 두 단어 이름·상품명 |
| 전체 집계 | 단일 테이블의 집계식 1개: COUNT(*) 또는 SUM/AVG/MIN/MAX |
| 그룹 집계 | 그룹 키 1개 + 집계식 1개. SELECT는 그룹 키, 집계식 순서 |
| HAVING | 그룹 집계의 SELECT와 동일한 집계식에 숫자 조건 1개 |
| 그룹 순위 | SELECT한 집계식으로 정렬, 필요한 경우 LIMIT 1~10 |
| 스칼라 서브쿼리 | 같은 테이블·같은 숫자 컬럼의 AVG/MIN/MAX와 비교. 깊이 1, 비상관 |
| 새 문법의 행 필터 | 바깥·안쪽 각각 최대 2개, AND 연결. 이름·ID 등호·IN·LIKE·비교·BETWEEN 허용. 서브쿼리 비교는 별도 1개 |
| 현재 AST의 미구현 범위 | 고급 OR, 복수 집계식·복수 그룹 키, 서브쿼리 결과의 그룹화·정렬·LIMIT. 의미상 금지가 아닌 구현 한계 |
| 공통 제외 범위 | JOIN, DISTINCT, 상관/중첩 서브쿼리, 산술식·날짜 함수·월별 집계, OFFSET, UNION, 윈도 함수 |

- 기존 조회에서는 AND/OR/IN/LIKE를 지원한다. 이름·ID 조건을 집계/서브쿼리와 결합하는 것은 허용한다.
  일반 OR에도 이름·ID·범주 부정·정렬·LIMIT을 결합할 수 있다. 정렬과 LIMIT은 전체 WHERE 결과에 적용하며 한 OR 항에만 적용하지 않는다.
  유일 키 조건과 다른 필터·정렬, WHERE 없는 전체 조회도 유효하다.
- 존재 확인은 L1·L14를 유지한다. “주문이 있나?”를 임의로 COUNT 질문으로 바꾸지 않는다.
- L1의 SELECT=WHERE 존재 확인 제약은 **일반 행 조회**에 적용한다. `AVG(price) WHERE price > 20.00`처럼 집계 입력을 필터링하는 질문에는 적용하지 않는다.
- 집계 출력은 L2의 show/which/summary 단서보다 우선한다. “show the average price”는 AVG(price)이며 `*`가 아니다.
- “고객별 주문 수”는 orders의 customer_id로 그룹화한다. 고객 이름까지 출력하라는 명시적 요청은 JOIN이 필요하므로 범위 밖이다.

### L19. 집계 함수의 기본 해석

| 표현 | 정답 해석 | 구분 단서 |
| --- | --- | --- |
| how many orders, number of orders, total orders, 총 주문 | COUNT(*) FROM orders | 주문 **건수**. quantity를 합하지 않음 |
| how many customers/items, number of customers/items | 해당 테이블 COUNT(*) | 고유 고객 수를 주문에서 세라는 요청은 DISTINCT가 필요하므로 범위 밖 |
| total units ordered, total quantity ordered, 주문 수량 합계 | SUM(quantity) FROM orders | 주문 **수량** |
| total stock, total units in stock, 재고 합계 | SUM(stock) FROM items | 상품 **종류 수**와 구분 |
| total price, sum of item prices | SUM(price) FROM items | 매출·주문금액·재고자산가치를 뜻하지 않음 |
| average/mean price, stock, quantity | AVG(price/stock/quantity) | 행별 값의 단순 산술평균. 가중평균 아님 |
| minimum/maximum price, lowest/highest price | MIN(price) / MAX(price) | 가격 **값**을 반환 |
| earliest/latest order date | MIN(order_date) / MAX(order_date) | 날짜 **값**을 반환 |

- “how many units”는 한 행이 특정되면 해당 stock/quantity의 일반 조회, 여러 행 또는 전체 집합을 가리키면 SUM이다.
  예: “quantity of order 5001” → `SELECT quantity ... WHERE order_id = 5001`;
  “units ordered by customer 1452” → `SELECT SUM(quantity) FROM orders WHERE customer_id = 1452`.
- “average order size”, “largest order size”에서 size의 기본값은 quantity다. 주문 금액을 뜻하는 명시적 단서가 있으면 이 기본값을 쓰지 않는다.
- COUNT는 `COUNT(*)`로 통일한다. COUNT(id), COUNT(column), COUNT(DISTINCT ...)로 대체하지 않는다.
- “평균 주문량”은 주문 행의 AVG(quantity)다. 고객별 합계의 평균처럼 두 번 집계하는 뜻을 임의로 추가하지 않는다.

### L20. 컬럼·대상 집합의 기본값

| 생략되거나 모호한 표현 | 기본 해석 |
| --- | --- |
| items under 80, items above 20 | price 비교. L16의 모호 표현도 3단계에서는 이 기본값으로 생성·라벨링 가능 |
| available, left, in stock | stock. ordered, units ordered, order size는 quantity |
| average stock, average price | 조건이 없으면 items 전체 |
| total orders, average order size | 조건이 없으면 orders 전체 |
| customer별 / item별 주문 집계 | orders의 customer_id / item_id |
| category별 집계 | items의 category. orders에는 category가 없으므로 주문의 카테고리별 집계는 JOIN 범위 |

- 명시적 단위·컬럼 단서가 기본값보다 우선한다. “stock below 80”은 price가 아니다.
- SUM/AVG의 대상은 price·stock·quantity로 한정한다. MIN/MAX도 같은 숫자 컬럼 또는 order_date에 사용한다.
  ID의 합·평균·최솟값·최댓값을 업무 집계로 사용하지 않는다. ID는 조회·조건·허용된 그룹 키로 사용한다.
- 숫자 기준 없는 “low stock”은 `stock <= 10`으로 정의하며 품절도 포함한다. “low but in stock”은 `stock BETWEEN 1 AND 10`이다.
  “out of stock”은 `stock = 0`, “in stock/available”은 기존대로 `stock > 0`이다.
  이는 업무상 보편 정의가 아니라 이 프로젝트의 기본값이다. 숫자가 명시되면 그 숫자와 연산자를 우선한다.
- 단독 “평균”, “합계”처럼 테이블·컬럼을 문장이나 값으로도 결정할 수 없으면 미정의다. 임의의 테이블을 추측하지 않는다.

### L21. 그룹과 출력

| 질문 | 정답 SQL의 핵심 |
| --- | --- |
| average item price | `SELECT AVG(price) FROM items` |
| average item price by category | `SELECT category, AVG(price) FROM items GROUP BY category` |
| orders per customer, 고객별 주문 | `SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id` |
| units ordered per customer | `SELECT customer_id, SUM(quantity) FROM orders GROUP BY customer_id` |
| stock by category | `SELECT category, SUM(stock) FROM items GROUP BY category` |
| price by category, 집계 함수 없는 카테고리별 가격 | `SELECT category, AVG(price) FROM items GROUP BY category` |
| orders by date, 일자별 주문 | `SELECT order_date, COUNT(*) FROM orders GROUP BY order_date` |

- by, per, for each, grouped by 및 “~별”은 그룹 단서다. 단, “orders by customer 1452”는 특정 고객 조건이지 그룹화가 아니다.
- 그룹을 요구하면 그룹 키도 기본으로 출력한다. “값만, 그룹 키는 제외”라는 명시적 요청은 현재 출력 문법 밖이다. 무시하고 키를 붙이지 않는다.
- 허용 그룹 키: customers는 city/membership, items는 category, orders는 customer_id/item_id/status/order_date.
- 없는 그룹을 만들지 않는다. 주문 0건인 고객까지 포함하는 집계, 모든 날짜의 빈 구간 채우기는 현재 범위 밖이다.

### L22. WHERE와 HAVING

개별 행의 속성 조건은 WHERE, 집계한 값의 조건은 HAVING이다. 조건의 위치는 문장 순서가 아니라 의미로 정한다.

| 질문 의미 | 정답 SQL의 핵심 |
| --- | --- |
| quantity가 5 이상인 주문을 고객별로 세기 | `WHERE quantity >= 5 GROUP BY customer_id` + COUNT(*) |
| 주문이 5건 이상인 고객 | `GROUP BY customer_id HAVING COUNT(*) >= 5` |
| 평균 가격이 50 이상인 카테고리 | `GROUP BY category HAVING AVG(price) >= 50` |
| 가격 50 이상인 상품들의 카테고리별 평균 가격 | `WHERE price >= 50.00 GROUP BY category` + AVG(price) |
| 고객별 주문 수량 합계가 20 초과 | `GROUP BY customer_id HAVING SUM(quantity) > 20` |

- “customers with at least 5 orders”는 customers 테이블의 행 조회가 아니라 orders의 customer_id와 COUNT(*)를 출력하는 그룹 집계다.
- 그룹을 요구하면서 “category별 price over 50”처럼 집계 함수가 생략되면 L21의 기본 집계에 대한 HAVING으로 해석한다.
  “items priced over 50, by category”처럼 **상품 행**을 먼저 제한하는 단서는 WHERE로 해석한다.
- L6의 숫자 경계와 부정 규칙을 집계 결과에도 적용한다. at least → >=, over → >, not equal → !=.
- 그룹 HAVING에는 SELECT와 같은 집계식을 사용한다. “평균 가격을 보여주되 상품 수가 5개 이상인 카테고리”는 복수 집계가 필요하므로 범위 밖이다.

### L23. 서브쿼리와 기준 집단

비교값이 숫자 리터럴이면 일반 조건, 비교값이 집계 결과이면 스칼라 서브쿼리다.

| 질문 | 기본 해석 |
| --- | --- |
| items more expensive than average | `price > (SELECT AVG(price) FROM items)` |
| kitchen items more expensive than average | 바깥 `category = 'kitchen'`, 안쪽은 **전체 items 평균** |
| items above the average price of kitchen items | 바깥 카테고리 제한 없음, 안쪽 `WHERE category = 'kitchen'` |
| kitchen items above the average kitchen-item price | 바깥·안쪽 모두 `category = 'kitchen'` |
| orders larger than average | quantity가 전체 orders의 AVG(quantity) 초과 |
| items at the highest price, all most expensive items | price가 전체 items의 MAX(price)와 같음. 동점 모두 |

- **기준 집단을 생략하면 같은 테이블 전체**다. 바깥 WHERE를 안쪽으로 자동 복사하지 않는다.
- “그중 평균보다”, “above their average”, “average of those items”처럼 앞서 지정한 집합을 명시적으로 다시 가리키면 해당 필터를 안쪽에도 적용한다.
- “각 상품의 같은 카테고리 평균”은 행마다 기준 집단이 바뀌는 상관 서브쿼리이므로 범위 밖이다. 고정된 kitchen 평균은 비상관이므로 가능하다.
- 비교 대상 컬럼과 안쪽 집계 컬럼은 동일하게 한다. price와 AVG(stock)의 비교는 범위 밖이다.
- 출력은 L2·L3·L13을 따른다. “which items ...”는 item_name, “all details ...”는 `*`, “item ids ...”는 item_id다.
  서브쿼리에서 SELECT와 바깥 WHERE의 컬럼이 같다는 이유로 존재 확인을 강제하지 않는다.

### L24. 최상급·순위·정렬·동점

| 표현 | 기본 해석 |
| --- | --- |
| highest price, maximum price | MAX(price): 값 1개 |
| the most expensive item / the cheapest item | price DESC/ASC + LIMIT 1: 행 1개 |
| the most expensive items / the cheapest items, 개수 없는 복수형 | price DESC/ASC + LIMIT 1: 복수형만으로 동점 전체를 뜻하지 않음 |
| all most expensive items, every item at the maximum price, tied for highest price | price = (SELECT MAX(price) FROM items): 동점 모두 |
| the 3 most expensive items, top 3 by price | price DESC + LIMIT 3 |
| categories with the highest average price | AVG(price) DESC + LIMIT 1: 그룹 1개 |
| top categories by average price, 개수 없는 top | AVG(price) DESC + LIMIT 1 |
| categories sorted by average price | AVG(price) ASC, LIMIT 없음 |
| show 5 kitchen items, 정렬 없는 개수 요청 | 자기 기본키 ASC + LIMIT 5 |

- 단수/복수 형태만으로 동점 포함 여부를 바꾸지 않는다. **all/every/tied** 등 동점 전체 단서가 기본 LIMIT 1보다 우선한다.
- 일반 “all items”의 all은 행 전체 또는 전체 목록 단서다. 최상급과 결합한 “all cheapest items”의 all은 동점 모두 단서다.
- 정렬 방향이 생략되면 ASC. top/highest/most/largest는 DESC, bottom/lowest/least/smallest는 ASC.
- 숫자 없는 first/last도 LIMIT 1. last는 명시된 정렬 키의 DESC이며, 키가 없으면 기본키 DESC다.
- 일반 조회의 동점 보조 키는 자기 기본키 **ASC**로 통일한다. 예: `ORDER BY price DESC, item_id ASC LIMIT 3`.
  기본키 자체를 정렬하면 중복해서 붙이지 않는다. 날짜 정렬도 `ORDER BY order_date DESC, order_id ASC`다.
- 그룹 집계는 `ORDER BY 집계식 방향, 그룹키 ASC`. 동점 보조 정렬은 시스템 기본값이므로 질문마다 설명하도록 강제하지 않는다.
  질문이 동점 기준을 명시하면 그 기준을 우선하되, 현재 문법으로 표현할 수 없으면 범위 밖으로 분류한다.
- “상위 3개”는 최대 3행이며 3등 동점을 추가로 포함하지 않는다. “3등 동점까지 전부”는 현재 범위 밖이다.
- 그룹 집계에서 “최고 평균인 모든 카테고리”는 그룹 결과에 대한 추가 집계가 필요하므로 현재 범위 밖이다.

### L25. 날짜

| 표현 | order_date 조건 |
| --- | --- |
| on, dated, 주문일이 d | `= 'd'` |
| before d, earlier than d | `< 'd'` |
| on or before d, through d, until d | `<= 'd'` |
| after d, later than d | `> 'd'` |
| on or after d, from d, since d | `>= 'd'` |
| between a and b, from a to b, a부터 b까지 | `BETWEEN 'a' AND 'b'`, 양 끝 포함 |
| latest/newest orders | order_date DESC. 개수 없으면 LIMIT 1 |
| earliest/oldest orders | order_date ASC. 개수 없으면 LIMIT 1 |

- 날짜 리터럴은 유효한 `YYYY-MM-DD`로 고정한다. 범위 양 끝은 빠른 날짜부터 쓴다.
- “날짜별”은 정확한 order_date별 그룹이다. “월별/연도별”로 확대 해석하지 않는다.
- 날짜 조건은 집계 전 주문 행에 대한 WHERE다. 기준 집단의 날짜를 명시하면 서브쿼리 안쪽 WHERE에 적용한다.
- MIN/MAX(order_date)는 날짜 값 조회, latest/earliest **orders**는 주문 행 조회다.
- today/yesterday/last month/최근처럼 기준일이나 기간이 필요한 표현에는 시스템 시각을 몰래 넣지 않는다.
  기준일과 기간을 입력에 제공하는 상대 날짜 규약은 현재 정의하지 않았으므로 미정의로 분류한다.
  “최근”만 있으면 현재 프로젝트에서는 **최신 주문 1개**로 해석하고, “최근 7일”은 기준일이 필요하므로 미정의다.

### L26. 정답 SQL 정규형·숫자·빈 결과

- 키워드 대문자, 텍스트 소문자, 문자열 작은따옴표. 별칭·AS·끝 세미콜론은 사용하지 않는다.
- SELECT 순서: 그룹 키 → 집계식. GROUP BY 없는 전체 집계는 집계식만 출력한다.
- WHERE는 스키마 컬럼 순서, orders에서는 order_date를 status 뒤에 추가한다. 안쪽 WHERE도 같은 규칙을 사용한다.
  서브쿼리 비교식은 바깥 일반 조건 뒤에 둔다. 기존 IN 값은 L8대로 질문에 나온 순서다.
- COUNT(*)로 통일. HAVING/ORDER BY는 SELECT한 집계식을 그대로 반복하고 별칭·컬럼 번호로 대체하지 않는다.
- price 리터럴은 기존처럼 소수 둘째 자리. COUNT/SUM(stock)/SUM(quantity) 등의 정수 임계값은 정수 표기.
  집계 비교 임계값의 불필요한 소수 0은 제거한다. 모델이 AVG의 실제 계산값을 질문에 복사하거나 반올림해서 SQL에 넣지 않는다.
- ROUND/COALESCE/IFNULL은 추가하지 않는다. “소수 둘째 자리로 반올림”을 명시적으로 요구하면 현재 범위 밖이다.
- 결과가 없을 수 있다는 이유만으로 정답 SQL을 바꾸지 않는다. 빈 입력의 COUNT(*)는 0, SUM/AVG/MIN/MAX는 NULL이다.
  GROUP BY할 행이 없으면 그룹 결과는 0행이며, 값 0인 그룹을 만들어 넣지 않는다.
- 실행 결과가 우연히 같아도 서로 다른 의미의 SQL은 같은 라벨로 취급하지 않는다. COUNT와 SUM의 의미 일치를 실행 결과만으로 판단하지 않는다.
- 빈 결과·NULL·COUNT=0은 유효한 예다. 실제 생성 비율은 데이터 생성 설정과 보고서에 별도 기록한다.

### L27. 이름·LIKE 및 기존 모호 표현의 확장

- 두 단어 고객 이름·상품명은 전체가 하나의 값이다. “james smith”를 이름 2개로 나누지 않는다.
  전체 이름만 주면 `=`, 접두·접미·포함 단서가 있으면 LIKE다. 이름 안의 공백은 보존한다.
- 방향 없는 “items like ka”, “ka-something items”의 기본값은 **포함 검색**: `item_name LIKE '%ka%'`.
  starts/begins with는 접두, ends with는 접미로 우선한다. 명시적 와일드카드는 L17을 따른다.
- 일반 조회의 SELECT 기본값은 기존 L2·L3·L13을 유지한다. 새 집계 의미가 없는 “show customers”를 COUNT로 바꾸지 않는다.
- L12의 개수 없는 복수 최상급·ORDER BY 없는 LIMIT, L16의 컬럼 없는 상품 숫자 비교·low stock,
  L17의 방향 없는 LIKE는 3단계에서 위 기본값으로 허용한다. 1·2단계의 제외 규칙은 과거 데이터에 유지한다.
- AND/OR 혼합, 여러 값의 부정처럼 SQL 구조 자체가 지원 범위를 넘는 표현은 기본값으로 다른 뜻으로 바꾸지 않는다.

### L28. LLM 질문 생성과 채택

- SQL은 코드가 만든 학습 SQL에서 선택한다. LLM은 SQL을 작성·수정하지 않고 질문만 작성한다.
- LLM 질문은 배치별 **새 독립 세션**으로 작성한다. 같은 세션에서 다음 배치를 이어 만들지 않는다.
  SQL 25개·SQL당 5질문은 현재 배치 스크립트의 운영값이며 의미·채택 규칙이 아니다.
- 직설형·대화형·명령형·상황 설명형·업무 메모형을 기본 말투로 사용한다. 각 문장은 실제로 작성하고 템플릿 생성 코드를 쓰지 않는다.
- 자연어 축약·기호·컬럼명 사용은 허용한다. SQL 용어가 있다는 이유만으로 탈락시키지 않는다.
  정답 SQL 전체를 질문에 복사한 예는 질문 생성 결과로 채택하지 않는다. SQL 문법에 가까운 문장이 많으면 말투 편향으로 보고한다.
- 기본 해석으로 충분한 내용은 모두 풀어 설명하도록 강제하지 않는다. 예: total orders를 반드시 “order row count”로 바꾸거나,
  기본 동점 처리를 매 질문에 반복하면 실제 사용자 말투와 멀어진다.
- 문자열·날짜·숫자 값을 임의로 바꾸지 않는다. 여러 표기가 같은 값인 경우는 L16·L26의 정규화 기준을 따른다.
- 검증은 ① 응답 형식·SQL 보존·개수 ② 값 보존 ③ 집계/그룹/필터 범위/비교/출력/순위의 의미 ④ 질문별 정답 충돌로 나눈다.
  리터럴이 모두 있다는 사실만으로 의미 검증 통과로 처리하지 않는다. 자동 의미 검사 미지원 항목은 검토 미완료로 기록한다.
- 정답이 충돌하면 기본값·명시적 단서로 라벨을 판정한다. 정책으로 옳고 그름을 결정할 수 있으면 SQL 원본은 보존하고 잘못 생성된 질문을 재작성·재검수한다.
  판정이 안 되면 격리하고 정책 결정을 기록한다. 중복 질문을 그대로 여러 번 넣어 가중치를 높이지 않는다.
- 기존 2,500쌍은 후보 상태다. 이 정책으로 재검수하기 전 자동으로 학습 폴더에 편입하지 않는다.

### L29. 데이터 분할·평가와 정책 버전

- 조합기 본 데이터 목표: 고유 SQL 8만 × 질문 3개 = 24만 쌍. 기존 유형 2만 SQL/6만 쌍,
  새 유형 6만 SQL/18만 쌍. LLM 질문은 SQL당 5개이며 별도 출처로 집계한다. 조합기와의 합계·중복 제거 후 수를 구분한다.
- SQL 그룹 단위로 학습/검증/평가를 분리한다. 같은 SQL의 다른 질문을 서로 다른 분할에 배치하지 않는다.
- 토크나이저는 채택된 학습 데이터만 사용한다. 평가·holdout·미검수 후보 질문은 BPE 학습에 넣지 않는다.
- 날짜·개체·ID holdout은 바깥 조건과 서브쿼리 안쪽 조건, BETWEEN 양 끝에서 모두 검사한다.
  구조 holdout은 값만이 아니라 지정한 SQL 구조 조합을 학습에서 제외한다.
- EM과 실행 정확도를 함께 보고하고, 정렬이 있으면 행 순서도 검사한다. 정책 위반 항목과 판정 미지원 항목을 구분해 보고한다.
- 정책은 학습 질문과 스키마·지원 범위를 근거로 정한다. 평가 오답에 맞춰 기본값을 바꾸지 않는다.
  변경일·변경 규칙·영향 데이터·재검수 여부를 보고서에 기록하고, 봉인 평가 라벨을 조용히 덮어쓰지 않는다.

### 3-13. 구현과의 차이 — 2026-10-14 작성 시점

이 절은 최초 정책 작성 당시의 상태다. 당시 변경은 문서만이었다. 이후 반영 상태는 다음 절을 따른다.

| 항목 | 2026-10-14 당시 상태와 필요한 반영 |
| --- | --- |
| ID 집계 금지 | advanced AST가 숫자형 여부로 검사하므로 정책의 허용 컬럼 명시 검사와 대조 필요 |
| 모호 표현 기본값 | 조합기·GPT 프롬프트·clean_ambiguous의 기본값 및 예외를 L18~L27과 대조하고 반영해야 함 |
| 그룹 출력 | 그룹 키 + 집계식, 집계식 순위·그룹키 ASC는 이미 구현 |
| 최상급 동점 | 일반 숫자 정렬의 기본키 보조 정렬은 미구현. 기존 날짜 정렬은 order_id도 주 정렬과 같은 방향이므로 ASC 정책으로 변경 필요 |
| 정렬 없는 LIMIT | 기존 일반 SQL 생성·검증은 이를 직접 허용하지 않음. 기본키 정렬로 정규화하는 처리가 필요 |
| 날짜 MIN/MAX 서브쿼리 | 날짜 최솟값·최댓값 집계는 있지만 날짜 비교 서브쿼리는 미지원. “최신 날짜의 주문 모두”는 현재 범위 밖 |
| 새 문법 질문 검증 | 형식·SQL 보존·문자열 검사만으로 집계 의미 검증은 완료되지 않음. 기존 검사기는 advanced 질문 일부 판정을 건너뜀 |
| 빈 집계 결과 | 현재 advanced 생성기가 빈 결과·NULL·COUNT=0을 제외함. 유효 예로 포함하는 생성 설정 보강 필요 |
| 기존 생성물 | 조합기 18만 쌍·GPT 후보 2,500쌍은 새 정책으로 전수 재검수하지 않음. 정책 적용 완료나 24만 쌍 생성 완료로 보고하지 않음 |

### 3-14. 틀 단위 반영 — 2026-10-15

- 검수 단위는 개별 대량 문장이 아니라 SQL 구조, 문장 틀, 표현 사전과 슬롯 결합 규칙이다.
  조합기 데이터는 틀별 대표·경계·반례 테스트와 전체 자동 검사로 검증한다. 자유 생성 LLM 질문은 틀의 정확성만으로 의미를 보장하지 못하므로
  배치 자동 검사와 구조·출처별 표본 감사를 별도로 적용한다. 2,500쌍 파일럿은 틀 문제를 발견하는 데 사용한다.
- 새 일반 Query는 모든 정렬에 자기 기본키 ASC 보조 키를 적용한다. 태그 SQL과 IN 값 재배열, 이름 교체도 같은 정렬을 보존한다.
  과거 SQL을 읽을 때는 누락된 보조 키를 몰래 추가하거나 명시적 DESC를 ASC로 바꾸지 않는다. 새 데이터는 새 규칙으로 재생성해야 한다.
- COUNT 표현에 number of customers/items/orders 및 total customers/items/orders를 추가했다.
- 집계 함수 생략 그룹 표현의 연결을 명시했다: 고객/상품/주문 → COUNT(*), 상품 price → AVG(price), stock → SUM(stock), units ordered → SUM(quantity).
- 서브쿼리의 안쪽 조건이 없을 때만 전체 기준 집단 표현을 생략할 수 있다. 안쪽 필터가 있으면 기준 집단 슬롯이 있는 틀만 사용한다.
- low stock/running low on stock은 stock <= 10에만, 방향 없는 name like 표현은 LIKE 포함 패턴에만 연결했다.
- 밑줄 컬럼명 단서 오탐과 checking 배경 설명의 L1 미탐을 보완했다. 집계 질문 전체 의미를 판정하는 범용 자연어 검사기는 여전히 없다.
- `tests/data/stage3/test_template_contracts.py`가 각 확장 틀의 필수 슬롯, 기준 집단 분리, WHERE/HAVING, 집계 표현,
  기본 정렬, 동점 전부와 LIMIT 1의 차이, 기존 SQL 보존을 검사한다. 정책의 모든 범위를 이 테스트만으로 보장한다고 해석하지 않는다.
- 정렬 없는 개수 요청의 기본키 정렬 질문 틀, 빈/NULL 집계 예의 생성 분포, 날짜 스칼라 서브쿼리 등 미지원 범위는 별도 보강 대상이다.
  대량 데이터 재생성·24만 쌍 구성·토크나이저 및 모델 재학습은 이 변경에 포함하지 않는다.

## 4. 판정 코드와 적용 위치

2026-10-15 재검토: 고급 이름·ID·IN·LIKE 필터, 안쪽 2조건, 유효한 빈/NULL 결과를 허용하고
일반 OR·유일 키 조합·전체 조회의 불필요한 제외를 제거했다. [현재 구현 범위](text-to-sql-stage3-report.md#rule-review)를 참고한다.
기존 코퍼스·토크나이저·모델은 이 변경으로 재생성/재학습하지 않았다.

3단계 틀 확장(2026-10-15): 문장 틀은 총 600개(학습 528·평가 전용 72)다.
기존/날짜 문형 309개와 확장 문형 291개이며, HAVING 덧붙임 3개·집계 표현 15개는 문장 틀 수에 합산하지 않는다.
확장 문형 중 학습 문형 97개는 학습 SQL로 만든 2,500문장 파일럿에서 구조를 추출했고,
커밋하지 않는 `data_raw/stage3/composer/template_provenance.json`에 원문 질문·SQL·원문 ID를 보존했다. 평가 문형은 원문 추출 대상으로 쓰지 않았다.
학습용 문형의 선택 확률은 직접/짧은형 70%, 대화형 20%, 상황형 10%를 기본으로 하며,
해당 SQL 역할에 없는 말투는 존재하는 말투 사이에서 재정규화한다. 이 비율은 실제 사용자 로그 통계가 아닌 설계값이다.
문형 개수가 많은 말투가 자동으로 더 많이 뽑히지 않도록 말투 그룹을 먼저 선택한다. 평가 전용 문형에는 이 학습 가중치를 적용하지 않는다.

### 4-1. 1단계 판정

| 규칙 | 코드 | 적용 대상 |
| --- | --- | --- |
| L1 | 질문 생성 프롬프트 규칙 10, `src/data/stage1/question_gen.soft_flags()` | LLM 질문 생성 라운드 검수 |
| L2~L4 | `src/data/stage1/clean_ambiguous.py` (`drop_reason`) | 학습 데이터: 규칙과 다른 라벨의 쌍을 지운다 (`--apply`) |
| L2~L4 | `src/data/stage1/sql_rules.label_mismatch` (같은 판정) | 평가: 규칙과 다른 문항을 뺀 규칙 일치 EM |
| L5 | — | 평가에서 따로 집계 |

### 4-2. 2단계 판정

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

### 4-3. 3단계 판정

`src/data/stage3/clean_ambiguous.py`의 `drop_reason(pair)`. 일반 조회는 2단계 판정(L1~L4·L14)에 L19 수량 합계를 더하고,
집계·그룹·서브쿼리(고급) 쌍은 아래로 판정한다. 조합기 검사(`compose check`), LLM 배치 검증(`llm_batches`), 매 epoch 유사어 교체,
3단계 규칙 일치 EM(`sql_rules.label_mismatch`)이 같은 판정을 쓴다.
`judge(pair)`는 `status`(violation / review / pass), `reason`, `evidence`, `unchecked`를 반환한다.
실제로 연결하지 못한 비교 방향·조건 범위만 review로 남긴다. `drop_reason`은 violation의 사유만 반환하므로 기존 호출 형식을 유지한다.
LLM audit은 위반과 review 모두 실패로 끝내며, review는 별도 `review_flags.json`에 저장한다. 확인 전 학습 편입은 금지한다.

| 사유 | 규칙 | 판정 |
| --- | --- | --- |
| L19 | 집계 함수 | 정답 함수의 단서가 없다(그룹 집계는 3-14의 함수 생략 기본 표현 인정). 집계 표현("average price")이 다른 함수와만 이어진다. COUNT에 "how many units"만 있다. orders COUNT에 고객·상품 수 단서만, customers·items COUNT에 주문 건수 단서가 있다. 일반 조회 stock·quantity가 여러 행인데 how many units / how much stock이다 |
| L20 | 집계 컬럼 | 집계·서브쿼리 비교 컬럼의 단서가 없거나, 집계 표현이 다른 컬럼과만 이어진다 |
| L21 | 그룹 | GROUP BY인데 그룹 키 단서가 없다. by / per / each 등의 목적어 및 명시된 그룹 속성이 SQL 키와 다르다. sorted by, 고객 ID가 뒤따르는 조건, 테이블에서 그룹 키가 될 수 없는 단가·단위 표현은 제외한다 |
| L22 | 조건 범위 | 질문의 범주 값·숫자·날짜가 SQL에 없다. 값·컬럼에 붙은 비교 방향이 SQL과 반대다. 행 속성의 WHERE와 집계 결과의 HAVING이 뒤바뀌었다. 같은 숫자도 행 조건·집계 조건 구간으로 나눠 대조한다 |
| L23 | 기준 집단 | 집계 기준 구간과 조회 요청이 다시 시작되는 지점을 구분해 안쪽·바깥 조건 값을 대조한다. their average 등 명시적 되가리킴은 바깥 집합을 안쪽에도 연결한다. 기준값에 붙은 비교 연산자도 대조한다. 출력 컬럼은 L2·L3·L14(L1 제외) |
| L24 | 정렬·LIMIT | 그룹 정렬을 요구하지 않았거나 방향 단서가 반대다(동점 설명 "ascending for ties"는 빼고 본다). 정렬이 없는데 요구한다. LIMIT n의 개수가 없다. LIMIT이 없는데 top 3 / "which category has the highest"로 개수를 정한다 |

- 범주 값은 clothes, non-shipped처럼 다른 표기로도 써서 L23 값 세기에서 뺀다. 이름·LIKE 조각은 단서 판정 전에 가린다("steel partition"의 partition).
- A2~A6 측정은 조합기 **학습** SQL과 GPT **학습** 쌍만 사용한다. 결과·남은 한계·review 비율은
  [검사기 요약](text-to-sql-stage3-report.md#checker-report)을 참고한다. 평가 문장·평가 오답으로 단서를 조정하지 않는다.
- 일반 조회 L19: "how much stock is there for …", "…, how many units are left?" 틀은 결과가 한 행일 때(LIMIT 1, 유일 키 = 조건)만 쓴다.
  여러 행이면 같은 표현을 `advanced.json`의 `aggregate_default`(SUM(stock)·SUM(quantity), 유일 키 = 조건이 없을 때만)로 만든다.
- 3단계 평가셋(`eval_build`)은 만들 때 이 판정에 걸린 문항을 버린다. 평가 전용 표현(phrase 세트)에서 판정이 어긋나도
  검사기를 평가 문장에 맞추지 않는다. 그래서 `compose check`는 phrase 세트의 불일치를 건수만 보고, 나머지 분할은 0.1%를 넘으면 실패한다.
