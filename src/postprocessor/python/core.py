# 역할: SQL 파싱, 식별자 교정과 SQL 재구성을 하나의 처리 흐름으로 묶는다.
# 사용처: 후처리 CLI와 테스트에서 process_sql()을 호출해 SQL을 교정할 때 사용한다.

from .parser import extract_identifiers
from .replacement import correct_identifiers
from .reconstruction import reconstruct_sql


def process_sql(
    sql_query, strategy="typo,ngram,semantic", dictionary=None, dim=None, threshold=0.0
):
    """
    파싱된 SQL 구문의 식별자들을 주어진 사전(dictionary)과 전략을 바탕으로 교정하고
    다시 SQL 문자열로 변환하여 반환합니다.
    사전이 비어있거나 주어지지 않으면 에러 메시지를 반환합니다.
    """
    if dictionary is None or len(dictionary) == 0:
        return "Error: No valid dictionary provided for correction. Please check the DB path."

    if "semantic" in strategy and dim is None:
        return "Error: dim is required when 'semantic' strategy is used."

    try:
        ast, identifiers = extract_identifiers(sql_query)
        correct_identifiers(identifiers, dictionary, strategy, dim, threshold)
        return reconstruct_sql(ast)
    except Exception as e:
        return f"Error processing SQL: {e}"
