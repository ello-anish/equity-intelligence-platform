"""
FastAPI service smoke tests.

These tests skip if no OOS prediction artifacts exist yet (i.e., the ML
pipeline hasn't been run). Otherwise they verify that every endpoint returns
well-formed JSON.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ART = Path(__file__).resolve().parent.parent / "artifacts"


def _artifacts_ready() -> bool:
    return any(
        (ART / f).exists()
        for f in ("oos_preds_baseline.parquet", "oos_preds_transformer.parquet")
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_ready(),
    reason="no ML artifacts — run `python run_ml_pipeline.py --fast` first",
)


@pytest.fixture(scope="module")
def client():
    from service.app import app, _clear_cache
    _clear_cache()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "no_predictions")
    assert "available_models" in data


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200


def test_predictions_latest(client):
    # Try both models; at least one should work
    for m in ("transformer", "baseline_gbr"):
        r = client.get(f"/predictions/latest?model={m}&top_k=5")
        if r.status_code == 200:
            data = r.json()
            assert "rows" in data
            assert len(data["rows"]) <= 5
            return
    pytest.skip("no predictions available")


def test_allocate(client):
    r = client.post("/allocate", json={
        "model": "transformer",
        "policy": "width_scaled",
        "top_k": 3,
    })
    if r.status_code == 404:
        pytest.skip("no transformer predictions")
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "transformer"
    assert data["policy"] == "width_scaled"
    # 3 longs + 3 shorts = 6 rows
    assert len(data["weights"]) == 6
    # Gross should be 1.0 for width_scaled
    assert abs(data["gross_leverage"] - 1.0) < 1e-5


def test_regimes_latest(client):
    r = client.get("/regimes/latest?n=10")
    if r.status_code == 404:
        pytest.skip("no regime labels")
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
