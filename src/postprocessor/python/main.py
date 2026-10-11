import argparse
import os
from .core import process_sql
from .schema import load_dictionary_from_sqlite


def main():
    """
    CLI 파라미터를 파싱하고 데이터베이스에서 사전을 로드하여 후처리기를 구동합니다.
    """
    parser = argparse.ArgumentParser(description="SQL Post-processor")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input SQL string to correct"
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="typo,ngram,semantic",
        help="Correction strategy (comma-separated for chaining: typo,ngram,semantic)",
    )
    parser.add_argument(
        "-b",
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database to extract schema dictionary from",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        choices=[50, 100, 200, 300],
        default=None,
        help="GloVe dimension to use for semantic correction (required if 'semantic' strategy is used)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Cosine similarity threshold for semantic correction (e.g. 0.5). If not set, all words in the cache are considered.",
    )

    args = parser.parse_args()

    if "semantic" in args.strategy and args.dim is None:
        parser.error("--dim is required when 'semantic' strategy is used.")

    dictionary = None
    if args.db:
        dictionary = load_dictionary_from_sqlite(args.db)
    else:
        # 리포지토리 내의 shop.db를 기본값으로 사용합니다.
        default_db_path = os.path.join(os.getcwd(), "data", "shop.db")
        if os.path.exists(default_db_path):
            dictionary = load_dictionary_from_sqlite(default_db_path)

    fixed_sql = process_sql(
        args.input, args.strategy, dictionary, args.dim, args.threshold
    )
    print(fixed_sql)


if __name__ == "__main__":
    main()
