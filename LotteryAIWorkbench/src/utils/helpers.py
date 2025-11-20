from importlib.metadata import version, PackageNotFoundError

_PROJECT_NAME = "LotteryAIWorkbench"
_FALLBACK_VERSION = "0.1.0"


def get_project_version() -> str:
    """Return the installed distribution version or fallback.

    Uses importlib.metadata to query the installed package version. If the
    package is not installed (editable dev or unpacked), falls back to the
    static version declared at scaffold time.
    """
    try:
        return version(_PROJECT_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
