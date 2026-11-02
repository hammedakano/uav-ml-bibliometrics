import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bibliometrics import flag_groups, METHODS, SENSORS, split_keywords  # noqa: E402


def test_synonyms_merge():
    assert split_keywords("UAV; Drone; Unmanned aerial vehicle; RF; Random Forests") == ["random forest", "uav"]


def test_empty_keywords():
    assert split_keywords(None) == [] and split_keywords("  ") == []


def test_flags():
    f = flag_groups("A U-Net model on UAV multispectral imagery", {**SENSORS, **METHODS})
    assert f["Multispectral"] and f["Deep learning"] and not f["LiDAR"]
