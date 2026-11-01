# experiments_10 method audit
Commit: ce5dcb4780780d9779eee6c43117d73d001db4af
Samples: CIGS 611 and 68, both held-out in Training A.
Paired acquisition: 20% multilevel Hadamard coarse J=6, PCM default NPL noise.
Matched fresh CPU methods: zero-filled, SPGL1, PnP-ADMM DRUNet, GS-PnP GS DRUNet, supervised U-TV epoch38.
Requested WVTNet: no exact implementation name; code supplies DWTDNet / GS-DWTDNet but their configured weights are absent.
Zero-shot U-TV: saved September output on same target but DIFFERENT acquisition; do not compare scores as a paired experiment.
Paired input measurement SHA256 is recorded per sample in provenance.json.
GPU compute modes were not changed and GPU processes were left untouched.
