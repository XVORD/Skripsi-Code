import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "verify_final_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_final_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_global_confusion_metrics():
    MODULE.verify_confusion_artifact("artifacts/global/global_metrics.json")


def test_decision_support_confusion_metrics():
    MODULE.verify_confusion_artifact("artifacts/global/decision_support_metrics.json")


def test_subject_partitions_are_disjoint():
    MODULE.verify_split()


def test_stagewise_reports_use_final_test_size():
    MODULE.verify_stagewise()
