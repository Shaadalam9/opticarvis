"""Guard how the Alpamayo2-Super planner's interpreter and checkpoint resolve.

The Super backend used to read only its own config keys, so the documented
planner overrides (OPTICARVIS_ALPAMAYO_PYTHON / _MODEL) were silently ignored,
and a relative interpreter path only worked from the repository root. CPU only.
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import _paths  # noqa: E402,F401  (every src/<group>/ onto sys.path)

import pipeline_common as pc  # noqa: E402


class Settings:
    """Temporarily replace the root config and the planner env overrides."""

    def __init__(self, config, python_env=None, model_env=""):
        self.config = config
        self.python_env = python_env
        self.model_env = model_env

    def __enter__(self):
        self.saved_cache = list(pc._ROOT_CONFIG_CACHE)
        self.saved_model = pc.ALPAMAYO_MODEL
        self.saved_env = os.environ.pop("OPTICARVIS_ALPAMAYO_PYTHON", None)
        pc._ROOT_CONFIG_CACHE[:] = [self.config]
        pc.ALPAMAYO_MODEL = self.model_env
        if self.python_env is not None:
            os.environ["OPTICARVIS_ALPAMAYO_PYTHON"] = self.python_env
        return pc

    def __exit__(self, *_):
        pc._ROOT_CONFIG_CACHE[:] = self.saved_cache
        pc.ALPAMAYO_MODEL = self.saved_model
        os.environ.pop("OPTICARVIS_ALPAMAYO_PYTHON", None)
        if self.saved_env is not None:
            os.environ["OPTICARVIS_ALPAMAYO_PYTHON"] = self.saved_env


def test_model_env_override_beats_config_beats_default():
    with Settings({}) as p:
        assert p.alpamayo2_super_model() == "nvidia/Alpamayo2-Super"
    with Settings({"ALPAMAYO2_SUPER_MODEL_ID": "org/configured"}) as p:
        assert p.alpamayo2_super_model() == "org/configured"
    with Settings({"ALPAMAYO2_SUPER_MODEL_ID": "org/configured"}, model_env="org/env") as p:
        assert p.alpamayo2_super_model() == "org/env"


def test_relative_interpreter_resolves_from_the_repository_root():
    config = {"ALPAMAYO2_SUPER_PYTHON": "external/alpamayo2/.venv/bin/python"}
    with Settings(config) as p:
        resolved = p.alpamayo2_super_python()
    assert os.path.isabs(resolved)
    assert resolved == os.path.normpath(
        os.path.join(pc.PROJECT_ROOT, "external", "alpamayo2", ".venv", "bin", "python")
    )


def test_interpreter_env_override_is_used_verbatim():
    config = {"ALPAMAYO2_SUPER_PYTHON": "external/alpamayo2/.venv/bin/python"}
    # A POSIX path must survive untouched even when this runs on Windows.
    with Settings(config, python_env="/opt/alpamayo2/.venv/bin/python") as p:
        assert p.alpamayo2_super_python() == "/opt/alpamayo2/.venv/bin/python"


def test_no_interpreter_configured_falls_back_to_this_one():
    with Settings({}) as p:
        assert p.alpamayo2_super_python() == sys.executable


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print("PASS  %s" % name)
        except AssertionError as error:
            failures += 1
            print("FAIL  %s\n      %s" % (name, error))
    raise SystemExit(1 if failures else 0)
