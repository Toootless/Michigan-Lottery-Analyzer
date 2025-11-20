import importlib

from src.utils.helpers import get_project_version


def test_version_available():
    v = get_project_version()
    assert isinstance(v, str)
    assert v.count('.') >= 1


def test_streamlit_app_import():
    mod = importlib.import_module('app.streamlit_app')
    assert hasattr(mod, 'predict_numbers')
