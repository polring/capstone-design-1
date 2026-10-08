import sqlite3
import os


def load_dictionary_from_sqlite(db_path):
    """
    Extracts all table names and column names from a SQLite database
    and returns them as a list of strings to be used as a dictionary.
    """
    if not os.path.exists(db_path):
        return []

    dictionary = set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            dictionary.add(table_name)

            # Get all column names for the table
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            for col in columns:
                dictionary.add(col[1])

        conn.close()
    except sqlite3.Error as e:
        print(f"Error reading database: {e}")

    return list(dictionary)
