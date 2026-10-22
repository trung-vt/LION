"""The same generator pairs reconstruction and error images in both reports."""

from pathlib import Path

from PIL import Image

from LION.pcm.randomized.visual_report_layout import (
    ADDITIONAL_METRICS,
    CELLS,
    RATES,
    SCENARIOS,
    presentation_figures,
)


def test_paired_report_visual_order_and_no_missing_figures(tmp_path):
    source = tmp_path / "source"
    archive = source / "complete_image_archive" / "montages"
    plots = source / "analysis" / "figures"
    archive.mkdir(parents=True)
    plots.mkdir(parents=True)
    placeholder = Image.new("RGB", (40, 24), color="white")
    for name in (
        ("normalized_source_pair.png",)
        + ADDITIONAL_METRICS
        + (
            "psnr_vs_sampling_pcm_default_10ms.png",
            "psnr_vs_sampling_pcm_short_100us.png",
        )
    ):
        placeholder.save(plots / name)
    for rate in RATES:
        for scenario in SCENARIOS:
            for index, _ in CELLS:
                for kind in ("recon", "abs_error"):
                    placeholder.save(
                        archive / f"case_{index:02d}_r{rate:03d}_{scenario}_{kind}.png"
                    )
    result = presentation_figures(source, tmp_path / "paper", tmp_path / "slides")
    assert result["slide_backup_reconstruction_error_pairs"] == 24
    assert result["paper_appendix_reconstruction_error_pairs"] == 12
    assert result["copied_figures_to_each_repo"] == 60

    slides = (tmp_path / "slides" / "complete_visual_frames.tex").read_text()
    paper = (tmp_path / "paper" / "complete_visual_appendix.tex").read_text()
    # All slide error plots directly follow their corresponding reconstruction frame.
    for rate in RATES:
        for scenario in SCENARIOS:
            for index, _ in CELLS:
                prefix = f"case_{index:02d}_r{rate:03d}_{scenario}"
                a = slides.index(prefix + "_recon.png")
                b = slides.index(prefix + "_abs_error.png")
                assert a < b
                assert slides[a:b].count(r"\begin{frame}") == 1
            # Each paper setting has the CIGS/Perovskite reconstruction
            # followed immediately by the two corresponding error images.
            prefix = f"case_00_r{rate:03d}_{scenario}"
            a = paper.index(prefix + "_recon.png")
            b = paper.index(prefix + "_abs_error.png")
            assert a < b
            assert paper[a:b].count(r"\begin{figure}") == 1
