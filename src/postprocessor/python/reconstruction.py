def reconstruct_sql(ast):
    """
    Converts the modified AST back into a valid SQL string.
    """
    return ast.sql()
