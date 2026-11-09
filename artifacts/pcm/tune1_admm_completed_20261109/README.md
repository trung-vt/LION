# Completed PnP-ADMM Tune-1 validation tuning

Status: 25/25 independently frozen ADMM parameter selections.
SSDU candidate configurations: 420. Four folds each = 1,680
complete ADMM solver reconstructions. No invalid candidates.
Mean selected validation SSDU loss: 0.0043147667823385215.

This is the ADMM CALIBRATION phase, not the full five-category test.
GS-PnP tuning and test set 45 image reconstruction have not started.

Report:
  report/ADMM_Tune1_Validation_Report.pdf
  report/ADMM_Tune1_Validation_Report.tex
  report/ADMM_25_Frozen_Selections.csv
  report/ADMM_25_Category_Summary.json

Source:
  source/report_admm_completed.py
  source/run_tune1.py
  source/ternary.py
  source/score_grid.py (not copied; use parent LION branch)
  source/README.md and PLAN.md

Each frozen validation selection is in:
  tuning/admm/<category>/<calibration_id>/frozen_selection.json
  tuning/admm/<category>/<calibration_id>/search_trace.json
Candidate records:
  tuning/admm/<category>/<calibration_id>/evaluations/*.json

Original blind measurement acquisitions and fixed fold arrays are
bundled separately. No validation ground truth or test images are
included. The 45 test image IDs are in prepared/test45.json.

This source report reproduces the completed ADMM tuning only;
no claims about test PSNR should be made until subsequent evaluation.
