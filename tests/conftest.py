import sys
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)

# Add project root to sys.path so tests can reliably import from `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """마커 필터가 테스트를 제외하기 전에 누락·중복 분류를 검사한다."""
    invalid_items = []
    for item in items:
        categories = {
            marker.name
            for marker in item.iter_markers()
            if marker.name in {"unit", "integration"}
        }
        if len(categories) != 1:
            reason = (
                "missing category" if not categories else "both unit and integration"
            )
            invalid_items.append(f"  {item.nodeid}: {reason}")

    if invalid_items:
        raise pytest.UsageError(
            "Each test must have exactly one category marker: unit or integration.\n"
            + "\n".join(invalid_items)
        )


def pytest_itemcollected(item):
    """테스트 수집 시 Docstring과 Parametrize 인자값을 함께 표시"""
    if item.obj.__doc__:
        # Docstring 첫 줄 가져오기
        doc = item.obj.__doc__.strip().split("\n")[0]

        # @pytest.mark.parametrize 사용 시 [값] 파라미터 텍스트 추출
        param_str = ""
        if "[" in item.name and "]" in item.name:
            param_str = " " + item.name[item.name.index("[") :]

        # Docstring + [파라미터값] 형태로 이름 교체
        item._nodeid = f"{item.nodeid.split('::')[0]} :: {doc}{param_str}"
