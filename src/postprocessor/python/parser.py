import re
import difflib
import sqlglot
import sqlglot.expressions as exp

SQL_KEYWORDS = [
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "JOIN",
    "ON",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "AS",
    "INNER",
    "LEFT",
    "RIGHT",
    "OUTER",
]


def pre_correct_keywords(sql_string):
    """
    sqlglot이 정상적으로 파싱할 수 있도록 SQL 예약어의 오타를 사전에 교정합니다.
    """
    tokens = re.split(r"(\W+)", sql_string)
    fixed = []
    for t in tokens:
        if t.isalpha():
            upper_t = t.upper()
            if upper_t in SQL_KEYWORDS:
                fixed.append(t)
            else:
                matches = difflib.get_close_matches(
                    upper_t, SQL_KEYWORDS, n=1, cutoff=0.7
                )
                if matches:
                    match = matches[0]
                    if t.islower():
                        match = match.lower()
                    fixed.append(match)
                else:
                    fixed.append(t)
        else:
            fixed.append(t)
    return "".join(fixed)


def extract_identifiers(sql_string):
    """
    SQL 문자열을 파싱하여 AST(추상 구문 트리)와 식별자 목록을 추출합니다.
    """
    corrected_sql = pre_correct_keywords(sql_string)
    ast = sqlglot.parse_one(corrected_sql)

    identifiers = []
    for node in ast.find_all(exp.Identifier):
        identifiers.append(node)

    return ast, identifiers
