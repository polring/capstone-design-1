# 역할: 식별자를 교정한 구문 트리를 SQL 문자열로 다시 변환한다.
# 사용처: 후처리 core.py에서 최종 교정 SQL을 반환할 때 사용한다.

def reconstruct_sql(ast):
    """
    수정된 AST 객체를 다시 유효한 SQL 문자열로 변환하여 반환합니다.
    """
    return ast.sql()
