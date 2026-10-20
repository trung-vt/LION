"""Tests for the PnP iteration-analysis command-line parser."""

import pytest

from LION.pcm.pnp_iteration_metrics import parse_pnp_iteration_metrics_cli


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_pnp_iteration_metrics_help_without_required_arguments(help_flag, capsys):
    with pytest.raises(SystemExit) as info:
        parse_pnp_iteration_metrics_cli([help_flag])
    assert info.value.code == 0
    output = capsys.readouterr().out
    assert "--method-ids" in output
    assert "--preset" in output
