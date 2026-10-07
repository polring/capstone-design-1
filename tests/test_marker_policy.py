"""별도 pytest 프로세스로 실제 수집·마커 필터 동작을 검증한다."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def marker_project(pytester, monkeypatch):
    """임시 프로젝트에도 저장소의 실제 훅과 설정을 그대로 적용한다."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(
        (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makeini((PROJECT_ROOT / "pytest.ini").read_text(encoding="utf-8"))
    return pytester


@pytest.mark.parametrize("selection", [None, "unit", "integration"])
def test_missing_marker_fails_before_deselection(marker_project, selection):
    marker_project.makepyfile("def test_unmarked(): pass")
    arguments = [] if selection is None else ["-m", selection]
    result = marker_project.runpytest_subprocess(*arguments)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_unmarked*missing category*"])


def test_two_categories_are_rejected(marker_project):
    marker_project.makepyfile(
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "@pytest.mark.integration\n"
        "def test_conflicting(): pass\n"
    )
    result = marker_project.runpytest_subprocess()

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_conflicting*both unit and integration*"])


def test_unregistered_marker_is_rejected(marker_project):
    marker_project.makepyfile(
        "import pytest\n" "@pytest.mark.unt\n" "def test_typo(): pass\n"
    )
    result = marker_project.runpytest_subprocess()

    assert result.ret != pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*'unt' not found in*markers*configuration option*"])


@pytest.mark.parametrize(
    "selection, passed, deselected",
    [(None, 4, 0), ("unit", 3, 1), ("integration", 1, 3)],
)
def test_inherited_and_function_markers_partition_tests(
    marker_project, selection, passed, deselected
):
    marker_project.makepyfile(
        test_module=(
            "import pytest\n"
            "pytestmark = pytest.mark.unit\n"
            "@pytest.mark.parametrize('value', [1, 2])\n"
            "def test_module_marker(value): assert value > 0\n"
        ),
        test_mixed=(
            "import pytest\n"
            "@pytest.mark.unit\n"
            "class TestUnit:\n"
            "    def test_class_marker(self): pass\n"
            "@pytest.mark.integration\n"
            "def test_function_marker(): pass\n"
        ),
    )
    arguments = [] if selection is None else ["-m", selection]
    result = marker_project.runpytest_subprocess(*arguments)

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=passed, deselected=deselected)


def test_same_category_at_multiple_levels_is_allowed(marker_project):
    marker_project.makepyfile(
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "@pytest.mark.unit\n"
        "def test_same_category(): pass\n"
    )
    result = marker_project.runpytest_subprocess()

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1)


def test_unmarked_selection_collects_zero_tests(marker_project):
    marker_project.makepyfile(
        "import pytest\n" "pytestmark = pytest.mark.unit\n" "def test_marked(): pass\n"
    )
    result = marker_project.runpytest_subprocess(
        "--collect-only", "-m", "not unit and not integration"
    )

    # pytest는 선택된 테스트가 없으면 정상적인 'NO_TESTS_COLLECTED'(5)를 반환한다.
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    result.stdout.fnmatch_lines(["*no tests collected*1 deselected*"])
