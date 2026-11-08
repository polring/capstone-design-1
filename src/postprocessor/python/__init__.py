# 역할: SQL 식별자 교정을 위한 Python 후처리 모듈들을 묶는 패키지다.
# 사용처: SQL 후처리 CLI와 테스트가 파서·교정·재구성 기능을 가져올 때 사용한다.

from .core import process_sql
from .schema import load_dictionary_from_sqlite

__all__ = ["process_sql", "load_dictionary_from_sqlite"]
