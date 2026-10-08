def reconstruct_sql(ast):
    """
    수정된 AST 객체를 다시 유효한 SQL 문자열로 변환하여 반환합니다.
    """
    return ast.sql()
