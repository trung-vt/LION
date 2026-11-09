# Test set 45: random category-specific Tune-1 comparison

Completed 9 November 2026 on tuni. Fixed 20% multilevel Hadamard sampling
with NPL-default noise, original 45 images (9 each of 5 categories).
The full test set 450 remains unchanged.

Exactly ten methods are compared: zero-filled, SPGL1-BP, SPGL1-BPDN,
PnP-ADMM-tuned-5, GS-PnP-tuned-oracle, scalar TV, UTV-C60,
GS-PnP-tuned-5, PnP-ADMM Tune-1 and GS-PnP Tune-1.

For each PnP method and category, one of five already frozen
validation-tuned parameter configurations was randomly drawn and fixed.
45 new ADMM and 45 new GS reconstructions were evaluated.
No test-image metric was used to choose parameters.

The nine-page report embeds one example per category, with the complete
45-page visual comparison as a separate PDF.

Raw arrays for all 90 new reconstructions and all baseline replays,
each matched by acquisition and mask SHA256, remain on tuni:
/store/LION/ttv22/pcm/results/pcm/tune1_ternary_20261109_v1/test45/

SPGL1-BP can differ slightly when replayed; main tables use original
archived benchmark metrics, and visual captions score the displayed BP
images. No parameter was retuned to match test scores.
