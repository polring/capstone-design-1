import argparse
import os
from .parser import extract_identifiers
from .replacement import correct_identifiers
from .reconstruction import reconstruct_sql
from .schema import load_dictionary_from_sqlite


def process_sql(sql_query, strategy="typo", dictionary=None, dim=100):
    """
    파싱된 SQL 구문의 식별자들을 주어진 사전(dictionary)과 전략을 바탕으로 교정하고
    다시 SQL 문자열로 변환하여 반환합니다.
    사전이 비어있거나 주어지지 않으면 에러 메시지를 반환합니다.
    """
    if dictionary is None or len(dictionary) == 0:
        return "Error: No valid dictionary provided for correction. Please check the DB path."

    try:
        ast, identifiers = extract_identifiers(sql_query)
        correct_identifiers(identifiers, dictionary, strategy, dim)
        return reconstruct_sql(ast)
    except Exception as e:
        return f"Error processing SQL: {e}"


def main():
    """
    CLI 파라미터를 파싱하고 데이터베이스에서 사전을 로드하여 후처리기를 구동합니다.
    """
    parser = argparse.ArgumentParser(description="SQL Post-processor")
    parser.add_argument(
        "--input", type=str, required=True, help="Input SQL string to correct"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="typo",
        help="Correction strategy (comma-separated for chaining: typo,ngram,semantic)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database to extract schema dictionary from",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[50, 100, 200, 300],
        default=100,
        help="GloVe dimension to use for semantic correction",
    )

    args = parser.parse_args()

    dictionary = None
    if args.db:
        dictionary = load_dictionary_from_sqlite(args.db)
    else:
        # 리포지토리 내의 shop.db를 기본값으로 사용합니다.
        default_db_path = os.path.join(os.getcwd(), "data", "shop.db")
        if os.path.exists(default_db_path):
            dictionary = load_dictionary_from_sqlite(default_db_path)

    fixed_sql = process_sql(args.input, args.strategy, dictionary, args.dim)
    print(fixed_sql)


if __name__ == "__main__":
    main()
