import json
from pathlib import Path

import pytest

from eval.cli import _resolve_completed_run, build_parser


def test_cli_exposes_only_fixed_dataset_workflow() -> None:
    parser = build_parser()

    prepare = parser.parse_args(["data", "prepare", "--dataset", "hotpotqa-distractor", "--seed", "7"])
    rebuild = parser.parse_args(["index", "rebuild", "--dataset", "hotpotqa-distractor"])
    generation = parser.parse_args(["run", "generation", "--dataset", "hotpotqa-distractor"])

    assert (prepare.command, prepare.data_command, prepare.seed) == ("data", "prepare", 7)
    assert (rebuild.command, rebuild.index_command) == ("index", "rebuild")
    assert (generation.command, generation.run_command) == ("run", "generation")


def test_cli_resolves_completed_run_id_and_rejects_partial_runs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    completed = reports / "run-1"
    completed.mkdir(parents=True)
    (completed / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    partial = reports / "run-2"
    partial.mkdir()
    (partial / "manifest.json").write_text(json.dumps({"status": "partial"}), encoding="utf-8")

    assert _resolve_completed_run(Path("run-1"), reports) == completed
    with pytest.raises(ValueError, match="Only completed"):
        _resolve_completed_run(Path("run-2"), reports)
