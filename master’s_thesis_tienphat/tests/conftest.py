import glob
import os
import sys
from pathlib import Path

import pytest

THESIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS))
# The training repository (sibling ../restran, or MFPLATE_REPO_ROOT). Importing settings
# also puts it on sys.path for src/ and configs/.
from mfplate.settings import REPO_ROOT as REPO, Settings  # noqa: E402
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

DATA = REPO / "dataset" / "data" / "train"
SERVING = THESIS / "artifacts" / "serving_config.json"
try:                                  # the served copy in model/ (or MFPLATE_MODEL)
    CHECKPOINT = Settings().load().model_path
except FileNotFoundError:
    CHECKPOINT = None
HAVE_MODEL = CHECKPOINT is not None and SERVING.exists()

needs_data = pytest.mark.skipif(not DATA.exists(), reason="dataset/ not available")
needs_model = pytest.mark.skipif(not HAVE_MODEL,
                                 reason="model/ checkpoint or serving_config.json missing")


def track_dir(track_id: str) -> Path:
    hits = glob.glob(str(DATA / "*" / "*" / track_id))
    if not hits:
        pytest.skip(f"{track_id} not in dataset/")
    return Path(hits[0])


def lr_files(track_id: str):
    d = track_dir(track_id)
    return [(p.name, p.read_bytes()) for p in sorted(d.glob("lr-*"))]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    if not HAVE_MODEL:
        pytest.skip("model/ checkpoint or serving_config.json missing")
    os.environ["MFPLATE_DB"] = str(tmp_path_factory.mktemp("db") / "test.sqlite3")
    from fastapi.testclient import TestClient
    from mfplate.api import app
    with TestClient(app) as c:
        yield c
