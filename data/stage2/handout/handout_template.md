# 2단계 질문 생성 안내

이 폴더는 Text-to-SQL 모델의 2단계 질문 데이터를 만드는 작업 묶음입니다. 여러분이 쓰는 LLM이 **질문 생성 프롬프트를
스스로 쓰고**, 그 프롬프트로 SQL <<N_SQL>>개에 대한 질문을 만듭니다. 지금 데이터는 다른 LLM이 쓴 문장 위주라, 이번
데이터는 여러분의 LLM이 고른 말투여야 의미가 있습니다. 프롬프트를 사람이나 다른 LLM이 대신 쓰지 않습니다.

```
README.md                          이 문서
sql/pilot.txt                      프롬프트 시험용 SQL <<PILOT_SIZE>>개
sql/batch_01.txt ~ batch_<<LAST_BATCH>>.txt   본 생성용 SQL (파일 하나에 <<BATCH_SIZE>>개)
```

## 지켜 주실 것

- **처음부터 끝까지 같은 모델, 같은 모드**를 씁니다(가장 성능이 좋은 모델 권장). 모델 이름과 모드(예: thinking 여부)를 기록해 주세요.
- **채팅마다 기억이 이어지지 않게** 합니다. 임시 채팅이 있으면 그것을 쓰고, 없으면 메모리·맞춤 지침·개인화를 꺼 주세요.
- 코드 실행, 캔버스, 파일 첨부 기능은 쓰지 않습니다. 텍스트를 그대로 붙여 넣습니다.
- **프롬프트와 응답을 손으로 고치지 않습니다.** 힌트나 예시 문장을 덧붙이지 않고, 기존 데이터나 다른 팀원의 결과를 보여 주지 않습니다.
- 모델이 되물으면 아래 BRIEF에 있는 내용으로만 답하고, BRIEF에 없는 것은 "decide it yourself"라고 답합니다.

## 진행 순서

### 1. 프롬프트 작성 (채팅 A)

새 채팅을 열고 아래 **BRIEF** 블록(``` 안쪽 전체)을 붙여 넣습니다. 모델이 쓴 프롬프트를 `prompt_v1.txt`로 저장합니다.

### 2. 파일럿 (새 채팅)

새 채팅을 열고 `prompt_v1.txt` 내용 바로 뒤에 `sql/pilot.txt` 내용을 붙여 넣습니다. 응답을 `pilot_v1.json`으로 저장합니다.

### 3. 점검·수정 (채팅 A로 돌아가기)

채팅 A에 아래 **REVIEW** 블록을 붙여 넣고, 그 아래에 `sql/pilot.txt` 내용과 `pilot_v1.json` 내용을 차례로 붙여 넣습니다.

- 모델이 `NO CHANGE`라고 답하면 `prompt_v1.txt`가 최종본입니다.
- 수정된 프롬프트를 주면 `prompt_v2.txt`로 저장하고 2~3단계를 한 번 더 합니다(`pilot_v2.json`).
- **수정은 최대 2번**(v3까지)입니다. 마지막 버전을 `prompt_final.txt`로 복사합니다.

### 4. 본 생성 (배치마다 새 채팅)

`sql/batch_01.txt`부터 `batch_<<LAST_BATCH>>.txt`까지, **배치 하나마다 새 채팅**을 엽니다. `prompt_final.txt` 내용 바로 뒤에 배치 파일
내용을 붙여 넣고, 응답을 `responses/batch_NN.json`으로 저장합니다(번호는 SQL 파일과 같게).

- 응답은 수정하지 않고 그대로 저장합니다.
- JSON이 깨졌거나 SQL 일부가 빠졌으면 그 배치를 **새 채팅에서 다시** 돌려 덮어쓰고, 다시 돌린 배치 번호를 기록합니다.
- 한 채팅에서 여러 배치를 이어서 돌리지 않습니다(같은 채팅이 계속 쓰면 문장이 몇 가지 틀로 굳습니다).

### 5. 돌려줄 것

아래를 폴더 하나(예: `gpt/`)에 담아 보내 주세요.

```
model.txt               모델 이름·모드, 작업 날짜, 다시 돌린 배치 번호
writer_chat.txt         채팅 A 대화 전체 (복사하거나 공유 링크)
prompt_v1.txt, prompt_v2.txt ...   중간 프롬프트
pilot_v1.json ...       파일럿 응답
prompt_final.txt
responses/batch_01.json ~ batch_<<LAST_BATCH>>.json
```

---

## BRIEF

```
Write a prompt for the task below. Do not write any of the questions yourself.

# Task
We are building question data for a small model that translates English questions into SQLite SQL
for one fixed shop database. For each SQL query we need 5 English questions that real users of this
shop's system could type, and whose correct translation is exactly that SQL.

Your prompt will be used like this:
- Each run is a new chat session with no memory of other runs. It receives your prompt, followed by
  <<BATCH_SIZE>> lines of the form `id=<number> | <sql>`.
- The session writes the questions itself. It must not write or run code to produce them.
- It returns only the JSON described under "Output format".

How the questions are phrased is your decision. The 5 questions for one SQL should read like 5 different
real users: vary length, tone, word choice and sentence structure (short notes to long requests, casual
to formal, questions and instructions). The model will later meet users we cannot predict, so variety
matters more than polish -- as long as every requirement below holds.

# Database
customers (customer_id, name, city, membership)
items     (item_id, item_name, category, price, stock)
orders    (order_id, customer_id, item_id, quantity, status)

- customer_id: 1000-1999. item_id: 3000-3999. order_id: 5000-5999.
- name, item_name: one lowercase word.
- customer_id, item_id, order_id, name and item_name are unique in their own table. In orders,
  customer_id and item_id can match several orders.
- city: new york, chicago, houston, seattle, boston, denver
- membership: bronze, silver, gold, platinum
- category: electronics, office supplies, kitchen, furniture, clothing, sports
- price: 1.00-199.99 dollars. stock: 0-100, units left in inventory.
- quantity: 1-10, units in one order. status: pending, shipped, delivered, cancelled
- Some values in the SQL list match no rows. Questions are written the same way either way.

# SQL that appears
One table, no joins. SELECT one column or *. WHERE has 0-3 conditions joined by AND, or by OR (never
both); OR and AND always join conditions on different columns. Conditions: = , != , < , > , <= , >= ,
BETWEEN, IN (a list of 2-3 values of one column), LIKE on name / item_name ('ab%' starts with,
'%ab' ends with, '%ab%' contains). Optionally ORDER BY price, stock or quantity, ASC or DESC, and
LIMIT n (only after ORDER BY).

# Requirements for every question
1. Ask for exactly what the SQL returns: the same table, the same selected column (or *), EVERY
   condition, the sort column and direction, and the number of rows when there is a LIMIT. Add nothing,
   leave nothing out. Long SQL must not lose a condition.
2. Contain every value exactly as written in the SQL: same spelling, numbers as digits, text values
   with or without quotes. A price may be written 20, 20.00 or $20 (not 20.0). A LIMIT of 2 or more
   appears as its digit. For LIKE, contain the letters between the % signs exactly, and make clear
   whether the name starts with, ends with, or contains them (wildcards like ka* or *ka* are fine).
   Never replace a value with a synonym, a broader word, or a description. Exception: stock = 0 may be
   said as out of stock / sold out, and stock > 0 as in stock / available.
3. Numbers are never spelled out as words.
4. No SQL keywords and no column identifiers with underscores (say "item id", not item_id).
5. Uses only what this one SQL needs. No information from another table.
6. The 5 questions for one SQL differ from each other.
7. Typos and abbreviations are allowed in ordinary words, never in values.

# The question must determine the SQL
Someone who knows the schema and sees only the question must arrive at exactly this SQL. These
readings are fixed:

| SQL part | The question must |
|---|---|
| SELECT * | ask for the records themselves (show / list / find them, full details), not one attribute |
| SELECT name / item_name / order_id | ask which ones (which, who, names of) |
| SELECT customer_id FROM customers, SELECT item_id FROM items | explicitly ask for the id |
| SELECT <other column> | ask for that attribute (for several rows: that attribute of each) |
| SELECT column = column of the only condition, which is = | ask whether such a row exists (and nothing else) |
| FROM orders | be about orders; FROM customers / items must not be about orders |
| stock vs quantity | stock = units left in inventory; quantity = units in one order. never mix them |
| = | no comparison word, or "exactly" |
| < and > | strict: "under / below / less than", "over / above / more than" exclude the value itself |
| <= and >= | inclusive: "at most / or less / up to", "at least / or more / and up" include the value |
| comparisons on price / stock | say which column (dollars, cost, price / in stock, left, available) |
| BETWEEN a AND b | an inclusive range from a to b, smaller value first |
| != | "not / except / other than" one category, city, membership or status value. for status, the literal status: "not shipped" is fine, "hasn't shipped yet" is not |
| IN | a list of values of the same column (joined by or, and, commas or slashes, in any order) |
| AND | all conditions hold |
| OR | either condition holds, clearly an alternative between different columns |
| ORDER BY without LIMIT | a sorted list, naming the sort column; DESC must be stated (high to low, most first), no direction means ASC |
| ORDER BY ... LIMIT 1 | a singular superlative (the cheapest item, the one with most stock) |
| ORDER BY ... LIMIT n | n with the superlative or top n (the 3 cheapest, top 5 by price) |

Never write questions that ask for a count ("how many items ..."), several columns, a plural superlative
without a number ("the cheapest items"), or vague amounts ("low stock").

# Output format (fixed)
The session must return only a JSON array, with no code fences and no text before or after it:
[
  {"id": <id from the input line>, "questions": ["...", "...", "...", "...", "..."]},
  ...
]
One object per input line, the same ids as the input, exactly 5 questions each.

# What to return
Return only the prompt text, ready to paste.
```

## REVIEW

```
Below are the pilot SQL list and the output of a new session that received your prompt followed by
that list. Check the output against the brief. If any question breaks a requirement, could be read as
a different SQL, drops part of the SQL, or is not what you intended, revise the prompt and return the
full revised prompt only. If no change is needed, reply exactly: NO CHANGE
```
