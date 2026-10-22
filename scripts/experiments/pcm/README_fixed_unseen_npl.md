# Fixed PCM validation/test and unseen NPL cells

This is a working research note. It concerns **216 GPU reconstructions** made
from two full, measured current maps using *simulated* undersampled/noisy PCM
acquisitions. It is not an independent compressed-acquisition experiment.

## Split design

Training images: the pre-existing DIV2K and PCM synthetic data. Random images,
crops, augmentation, sampling fractions and independent noise realizations are
allowed during training. The two new NPL cells were never used to train.

Validation images: four natural and four synthetic images from the manifest
validation split. A fixed grid is used; any SSDU checkpoint selection must use
the held-out measurement MSE, **never** the clean-image validation metric.

Ordinary test images: four natural and four synthetic images in the disjoint
manifest test split. Separate unseen tests use two previously untrained
measured cell maps (CIGS 256x256, perovskite centre 256x256).

## Fixed validation/test settings

Sampling rates: 10%, 20%, 50%, fixed seed/mask per case and rate, DC
retained, coarse-block plus uniform-tail sampling; coarseJ=J-2
(coarseJ=4 for 64x64, coarseJ=6 for 256x256).

Four noise settings: exact noiseless, independent Gaussian dark-only,
the existing NPL default PCM model (nominal 10 ms), and the NPL short
aperture extrapolation model (nominal 100 microseconds).

Normalized measurement-noise amplitudes from LION's NPL builder:

| Parameter | Default at 256 | Short at 256 |
| --- | ---: | ---: |
| Dark SD | 0.0402902 | 0.231027 |
| Beam innovation SD | 0.00012 | 0.00060 |
| Bias SD | 0.000029 | 0.000029 |
| Bias slope SD | 0.00000016 | 0.00000016 |

At 64x64, dark SD becomes 0.161161 (default) or 0.924107 (short).
AR=(-.9221,-.1709,-.0769,-.0354); MA=(.0377,-.3669); integrated
ARIMA order d=1. Signal-to-bias ratio 0.3. For Gaussian dark-only,
only the additive dark component is active, at the default amplitude.

The underlying NPL preset uses current_to_model_scale=1/(3.5e-8)
and a dark scaling factor of 2^-J. **These are model-unit scenarios,
not current measurements calibrated for the two new cells.**

Fixed seed bases are validation 1,300,000; test 2,300,000; unseen
3,300,000, with case/rate offsets. Measurement noise has a separate
independent random stream from the mask. The same input acquisition is
shared by all reconstruction methods, not resimulated per model.

## Immutable stored acquisition datasets

Data root:
/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot/fixed_acquisitions_v1/

Every split contains a manifest.json and multiple immutable NPZ files.
Each NPZ holds:
- reference: one clean normalized float32 image;
- sampled_indices: exact external Hadamard row indices, int64;
- measurement: exact simulated noisy coefficient vector, float32.

There are **96 validation, 96 test and 24 unseen acquisitions** in all,
around 6.6MB. Manifests record source hashes, NPZ hashes, fixed parameters,
noise/mask seeds and noise RMS.

Verify every file and mask without doing any training:

    micromamba activate lion_2
    cd /mnt/store_ext4/GIT/LION
    export PYTHONPATH=. LION_DATA_PATH=/mnt/store_ext4/GIT/LION/data
    export LION_TTV22_PATH=/mnt/store_ext4/store/LION/ttv22
    python -m LION.pcm.randomized.verify_fixed_cases \
      --root /mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot/fixed_acquisitions_v1

A new dataset can be recreated (without overwriting differing data)
using python -m LION.pcm.randomized.persist_fixed_cases --help.
The live experiment generation and test reconstruction use CUDA
(e.g. RTX 5090 on cuda:1); no CPU fallback was used for the unseen benchmark.

## Unseen specimens and results

Original archive and safe extracted raw maps:
/mnt/store_ext4/store/LION/ttv22/pcm/data/unseen_data/

The 768x1024 raw canvases have differing active bounds: CIGS 256x256,
perovskite 512x512. We use a predefined perovskite centre 256x256 crop.
Each cell's negative current magnitude is normalized to its own positive
99.5th-percentile scale and clipped solely to construct the displayed
reference target. This is a **contrast transform, not physical calibration**.
Screenshot file names are marked not_sure pending source confirmation.

Complete benchmark output:
/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot/unseen_npl_fixed_20261014_v1/

Figure rendering reuses the project's existing YAML and plotting utilities: reconstruction **afmhot** (0–1), absolute error **viridis** (0–0.5), and unchanged parameter-map **magma**. Use LION.pcm.randomized.rerender_unseen_figures and visual_report_layout for reproducible display-only refreshes; the original grayscale/magma visual previews are backed up in the results folder, and no numerical outputs are modified.

Nine methods are frozen: zero-fill; SPGL1; scalar anisotropic TV with
lambda=.02 and 256 PDHG iterations; U-TV and U-TGV (supervised and SSDU);
PnP-ADMM + DWTDNet and GS-PnP + DRUNet (20 iterations each with previously
validation-selected settings). Learned checkpoints were trained at 64x64,
transferred to 256x256 without further tuning, and only DWTDNet's
deterministic resolution-dependent attention masks were regenerated.

Output folder complete_image_archive contains **216 float32 reconstructions**,
240 single PNGs, 24 full method-grid images, and 24 absolute-error grids.
Every replayed reconstruction agrees with the original metrics to within
0.001 MSE (largest difference 0.00001751).

At 20% sampling, mean PSNR over two cells:
- Default NPL-like noise: zero-fill 26.10 dB; scalar TV 30.72 dB (+4.62).
- Short-aperture stress: zero-fill 16.23 dB; supervised U-TGV 24.28 dB (+8.05).

## Paper and slides

Complete report PDF, including every numerical plot and reconstruction/error
panel:
/mnt/store_ext4/GIT/LATEX/PCM-PnP-Paper/reports/unseen_npl_20261014/report.pdf

Standalone Beamer slides with all conditions in backup:
/mnt/store_ext4/GIT/LATEX/A3SmartML-month-18-meeting/experiments/unseen_npl_20261014/slides.pdf

Do not claim independent measurement reconstruction, physically calibrated
uncertainty or statistical superiority: only two cells and one simulated
noise realization per case are available. Request independently acquired
compressed sensing data and exposure-specific noise calibration from NPL.
