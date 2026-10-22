"""Read-only monitoring and PnP trace tests."""

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.randomized.dashboard import make_app
from LION.pcm.randomized.monitoring import append_event, run_history, training_preview
from LION.pcm.randomized.sweep import solve


def test_append_event_and_live_history(tmp_path):
    folder = tmp_path / "runs" / "run"
    folder.mkdir(parents=True)
    append_event(folder / "history.jsonl", {"step": 1, "training_loss": 0.25})
    append_event(folder / "history.jsonl", {"step": 2, "training_loss": 0.15})
    with (folder / "history.jsonl").open("a") as f:
        f.write('{"incomplete":')
    rows = run_history(folder)
    assert [x["step"] for x in rows] == [1, 2]


def test_training_preview(tmp_path):
    clean = np.zeros((1, 1, 16, 16), dtype=np.float32)
    arrays = {}
    for domain, idx in (("natural", 0), ("pcm_synthetic", 2)):
        for suffix, value in (("_clean", 0.0), ("_input", 0.4), ("_prediction", 0.1)):
            arrays[f"{domain}_{idx}{suffix}"] = clean + value
    training_preview(tmp_path, 20, arrays)
    path = tmp_path / "validation_steps" / "step_000020.png"
    with Image.open(path) as image:
        assert image.width > image.height
        assert image.size[0] >= 450


def test_dashboard_is_read_only_and_manual(tmp_path):
    (tmp_path / "runs" / "trial").mkdir(parents=True)
    (tmp_path / "runs" / "trial" / "run.json").write_text(
        json.dumps({"status": "running"})
    )
    append_event(
        tmp_path / "runs" / "trial" / "history.jsonl", {"step": 1, "training_loss": 0.3}
    )
    client = TestClient(make_app(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "Refresh" in page.text
    assert 'type="checkbox" id="auto"' in page.text
    assert client.get("/plot/trial/training_loss").status_code == 200
    assert client.get("/plot/trial/model_state_dict").status_code == 404
    assert client.get("/image/runs/trial/run.json").status_code == 404
    assert client.get("/image/../../etc/passwd").status_code in (404, 422)


def test_pnp_diagnostics_not_used_for_solve():
    class Identity(torch.nn.Module):
        def forward(self, x, sigma=None):
            return x

    op = PhotocurrentMapOp(j_order=4, normalized=True, device="cpu")
    ground_truth = torch.rand(1, 1, 16, 16)
    y = op(ground_truth)
    config = {"algorithm": "pnp_admm", "rho": 0.1, "sigma": 0.2}
    result, normal = solve(op, y, Identity(), config, 6)
    snapshots = {}
    diagnostic_result, trace = solve(
        op, y, Identity(), config, 6, reference=ground_truth, snapshots=snapshots
    )
    torch.testing.assert_close(result, diagnostic_result, atol=1e-6, rtol=1e-6)
    assert [row["iteration"] for row in trace] == list(range(7))
    assert set(snapshots) >= {"0", "1", "5"}
    assert all(
        "psnr" in row and "ssim" in row and "relative_update_norm" in row
        for row in trace
    )
    assert "psnr" not in normal[0]
