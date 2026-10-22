"""Read-only dashboard of PCM runs: manual refresh by default."""
from __future__ import annotations
import argparse
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
import uvicorn

from LION.pcm.randomized.monitoring import run_history


def make_app(root: Path) -> FastAPI:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def checked(path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise HTTPException(404)
        return path

    @app.get("/", response_class=HTMLResponse)
    def home(run: str = ""):
        dirs = sorted(
            d for d in (root / "runs").glob("*") if (d / "run.json").is_file()
        )
        if not dirs:
            return "<h1>No runs found</h1>"
        folder = next((d for d in dirs if d.name == run), dirs[0])
        options = "".join(
            '<option value="'
            + html.escape(d.name)
            + '"'
            + (" selected" if d == folder else "")
            + ">"
            + html.escape(d.name)
            + "</option>"
            for d in dirs
        )
        metadata = json.loads((folder / "run.json").read_text())
        summary = "".join(
            "<tr><th>"
            + html.escape(str(k))
            + "</th><td>"
            + html.escape(str(metadata.get(k, "—")))
            + "</td></tr>"
            for k in (
                "status",
                "completed_updates",
                "best_step",
                "best_validation_mse",
                "best_selection_score",
                "test_mean_psnr",
                "elapsed_seconds",
            )
        )
        plots = "".join(
            '<div class="panel"><h3>'
            + field.replace("_", " ")
            + '</h3><img src="/plot/'
            + html.escape(folder.name)
            + "/"
            + field
            + "?v="
            + str(
                (folder / "history.jsonl").stat().st_mtime_ns
                if (folder / "history.jsonl").exists()
                else (folder / "history.json").stat().st_mtime_ns
                if (folder / "history.json").exists()
                else 0
            )
            + '"></div>'
            for field in (
                "training_loss",
                "gradient_norm",
                "validation_mse",
                "selection_score",
                "seconds",
            )
        )
        preview_list = sorted((folder / "validation_steps").glob("*.png"))
        previews = "".join(
            '<div class="panel"><h3>'
            + html.escape(p.name)
            + '</h3><img loading="lazy" src="/image/runs/'
            + html.escape(folder.name)
            + "/validation_steps/"
            + html.escape(p.name)
            + '"></div>'
            for p in preview_list
        )
        if not previews:
            previews = "<p>Older runs logged final examples only. Per-checkpoint images start with future runs.</p>"
        figures = (
            sorted((root / "monitoring" / "figures").glob("*.png"))
            + sorted((root / "report_bundle" / "figures").glob("*.png"))
            + sorted((root / "pnp_sweep" / "convergence" / "figures").glob("*.png"))
        )
        images = "".join(
            '<div class="panel"><h3>'
            + html.escape(p.name)
            + '</h3><a target="_blank" href="/image/'
            + html.escape(str(p.relative_to(root)))
            + '"><img loading="lazy" src="/image/'
            + html.escape(str(p.relative_to(root)))
            + '"></a></div>'
            for p in figures
        )
        return (
            """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>PCM monitoring</title>
<style>body{font:15px system-ui,sans-serif;background:#111820;color:#e8ecf4;margin:25px auto;max-width:1440px}
button,select{padding:8px;background:#293b4d;color:white;border:1px solid #556f85;border-radius:5px}
.panel{padding:12px;border:1px solid #435266;border-radius:10px;background:#182632}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(385px,1fr));gap:12px}
img{max-width:100%;height:auto}th,td{padding:6px 15px;text-align:left;border-bottom:1px solid #435266}
small{color:#acbace}h3{font-size:16px}</style></head><body><h1>PCM experiment dashboard</h1>
<form method="get"><label>Run: <select name="run">"""
            + options
            + """</select></label>
<button type="submit">View run</button> <button type="button" onclick="location.reload()">Refresh</button>
<label><input type="checkbox" id="auto" onchange="sessionStorage.setItem('pcm-dashboard-auto',this.checked ? 'yes' : 'no')"> Refresh every 5 seconds</label></form>
<small>Local read-only dashboard. Auto-refresh defaults off. Historical runs cannot reconstruct missing checkpoints.</small>
<div class="panel"><h3>Run summary</h3><table>"""
            + summary
            + """</table></div><h2>Training curves</h2><div class="grid">"""
            + plots
            + """</div><h2>Saved validation images</h2><div class="grid">"""
            + previews
            + """</div><h2>PnP and report figures</h2><div class="grid">"""
            + images
            + """</div><script>document.getElementById('auto').checked=sessionStorage.getItem('pcm-dashboard-auto')==='yes';setInterval(function(){if(document.getElementById('auto').checked){location.reload()}},5000);</script>
</body></html>"""
        )

    @app.get("/plot/{run}/{field}")
    def plot(run: str, field: str):
        allowed = {
            "training_loss",
            "gradient_norm",
            "validation_mse",
            "selection_score",
            "seconds",
        }
        if field not in allowed or "/" in run or run in (".", ".."):
            raise HTTPException(404)
        directory = checked(root / "runs" / run)
        if not (directory / "run.json").is_file():
            raise HTTPException(404)
        data = [
            (int(row["step"]), float(row[field]))
            for row in run_history(directory)
            if row.get(field) is not None
        ]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        if data:
            ax.plot([p[0] for p in data], [p[1] for p in data], color="#328ccd", lw=1.6)
            if field not in ("seconds",) and min(y for _, y in data) > 0:
                ax.set_yscale("log")
        ax.set(xlabel="Update", ylabel=field.replace("_", " "))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=110)
        plt.close(fig)
        return Response(
            stream.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/image/{relative:path}")
    def image(relative: str):
        candidate = checked(root / relative)
        if not candidate.is_file() or candidate.suffix.lower() not in (
            ".png",
            ".jpg",
            ".jpeg",
        ):
            raise HTTPException(404)
        return FileResponse(candidate, headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port}/ (manual refresh by default)", flush=True)
    uvicorn.run(
        make_app(args.root), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
