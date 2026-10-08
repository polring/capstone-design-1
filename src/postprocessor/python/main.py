import argparse
from .parser import extract_identifiers
from .replacement import correct_identifiers
from .reconstruction import reconstruct_sql

# Target dictionary mapping domain specific terms and column names
DICTIONARY = ["users", "user_name", "email", "id", "name", "department", "manager_id"]


def process_sql(sql_query, strategy="typo", dictionary=None):
    if dictionary is None:
        dictionary = DICTIONARY

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
        choices=["typo", "ngram", "semantic"],
        help="Correction strategy",
    )

    args = parser.parse_args()

    fixed_sql = process_sql(args.input, args.strategy)
    print(fixed_sql)


if __name__ == "__main__":
    main()
