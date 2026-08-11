from flimkit.utils.update_check import (
    _compare_versions,
    _parse_github_repo,
    _parse_version_tuple,
    format_update_report,
)
from flimkit._version import __version__


def test_parse_version_tuple_with_suffix():
    assert _parse_version_tuple("0.9.9-stable") == (0, 9, 9)


def test_compare_versions_detects_older():
    assert _compare_versions("0.9.8", "0.9.9") == -1


def test_compare_versions_detects_equal_even_with_prefix():
    assert _compare_versions("1.0.0", "v1.0.0") == 0


def test_parse_github_repo_https_and_ssh():
    assert _parse_github_repo("https://github.com/FLIMKit/FLIMKit.git") == "FLIMKit/FLIMKit"
    assert _parse_github_repo("git@github.com:FLIMKit/FLIMKit.git") == "FLIMKit/FLIMKit"


def test_format_report_includes_key_status_lines():
    current_version = __version__
    report = format_update_report(
        {
            "current_version": current_version,
            "git": {
                "is_repo": True,
                "branch": "main",
                "upstream": "origin/main",
                "ahead": 0,
                "behind": 0,
                "is_up_to_date": True,
                "error": None,
            },
            "release": {
                "latest_version": "0.9.9",
                "source": "release",
                "is_latest": True,
                "error": None,
            },
        }
    )
    assert f"Current version: {current_version}" in report
    assert "Git result: up to date with upstream" in report
    assert "Version result: local version is up to date" in report
