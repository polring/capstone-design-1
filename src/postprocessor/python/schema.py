import sqlite3
import os


def load_dictionary_from_sqlite(db_path):
    """
    SQLite 데이터베이스에서 모든 테이블명과 컬럼명을 추출하여,
    교정 사전으로 사용할 문자열 리스트로 반환합니다.
    """
    if not os.path.exists(db_path):
        return []

    dictionary = set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 모든 테이블명 조회
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            dictionary.add(table_name)

            # 해당 테이블의 모든 컬럼명 조회
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            for col in columns:
                dictionary.add(col[1])

        conn.close()
    except sqlite3.Error as e:
        print(f"Error reading database: {e}")

    return list(dictionary)
