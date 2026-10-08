import argparse
import os
from .parser import extract_identifiers
from .replacement import correct_identifiers
from .reconstruction import reconstruct_sql
from .schema import load_dictionary_from_sqlite

# Default dictionary just in case db is not provided or fails to load
DEFAULT_DICTIONARY = [
    "users",
    "user_name",
    "email",
    "id",
    "name",
    "department",
    "manager_id",
]


def process_sql(sql_query, strategy="typo", dictionary=None):
    if dictionary is None or len(dictionary) == 0:
        dictionary = DEFAULT_DICTIONARY

    try:
        ast, identifiers = extract_identifiers(sql_query)
        correct_identifiers(identifiers, dictionary, strategy)
        return reconstruct_sql(ast)
    except Exception as e:
        return f"Error processing SQL: {e}"


def main():
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

    args = parser.parse_args()

    dictionary = None
    if args.db:
        dictionary = load_dictionary_from_sqlite(args.db)
    else:
        # Default fallback to shop.db in the repo
        default_db_path = os.path.join(os.getcwd(), "data", "shop.db")
        if os.path.exists(default_db_path):
            dictionary = load_dictionary_from_sqlite(default_db_path)

    fixed_sql = process_sql(args.input, args.strategy, dictionary)
    print(fixed_sql)


if __name__ == "__main__":
    main()
