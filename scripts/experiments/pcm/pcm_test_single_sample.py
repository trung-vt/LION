"""Script to run a single PCM sample experiment."""

from __future__ import annotations

import os

from LION.pcm.config_cli import parse_pcm_experiment_config_cli
from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment


def main() -> None:
    """Run the selected PCM experiment preset.

    Example usage:
    - To see all options:
        ```
        python scripts/experiments/pcm/<this_script_name>.py --help
        ```
    - Example of running a preset experiment:
        ```
        python scripts/experiments/pcm/<this_script_name>.py --preset cigs_example_256x256
        ```
    """
    single_sample_config = parse_pcm_experiment_config_cli()
    # Optional fixed naming and independent noise seed for reproducible sweeps.
    # Ordinary invocations with no environment overrides remain unchanged.
    sample_name = os.environ.get("PCM_SINGLE_SAMPLE_NAME") or None
    noise_seed_text = os.environ.get("PCM_SINGLE_SAMPLE_NOISE_SEED")
    noise_seed = int(noise_seed_text) if noise_seed_text is not None else None
    experiment_log_dir = run_single_sample_experiment(
        single_sample_config,
        sample_name=sample_name,
        noise_seed=noise_seed,
    )
    print(f"Results written to: {experiment_log_dir}")


if __name__ == "__main__":
    main()
