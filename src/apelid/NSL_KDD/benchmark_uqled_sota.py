"""End-to-end one-seed UQ-LED CL-MCD-E benchmark for NSL-KDD."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from apelid.EdgeIIoTset.benchmark_uqled_sota import (
    CANONICAL_CONFIG,
    _is_canonical,
    parse_noise_rates_list,
    require_complete_metrics,
    resolve_downstream_device,
    run_step,
    validate_single_seed,
)

MODEL_ORDER = ["XGB", "CATB", "BAGGING", "LGBM", "RF", "DNN"]
CLASSICAL_MODELS = ["xgb", "catb", "bagging", "lgbm", "rf"]

NOISE_FILE_PATTERN = re.compile(
    r"^nslkdd_train_clean_merged_noise_(\d+)$",
    flags=re.IGNORECASE,
)


def extract_noise_pct(path: Path) -> int | None:
    match = NOISE_FILE_PATTERN.match(path.stem)
    return int(match.group(1)) if match else None


def parse_training_summary(text: str) -> list[dict]:
    pattern = re.compile(
        r"\b(XGB|CATB|BAGGING|HISTGBM|GBM|LGBM|RF|DNN)\s*-\s*"
        r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*F1\s*\(Macro\):\s*([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )
    found: dict[str, dict] = {}
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            model = match.group(1).upper()
            found[model] = {
                "model": model,
                "accuracy": float(match.group(2)),
                "f1_macro": float(match.group(3)),
            }
    return [found[model] for model in MODEL_ORDER if model in found]


def _build_training_command(
    python_exe: str,
    training_script: Path,
    train_csv: Path,
    output_dir: Path,
    models: list[str],
    device: str,
) -> list[str]:
    """Run a selected model subset without changing NSL-KDD's trainer."""
    child_argv = [
        str(training_script),
        "--model",
        "all",
        "-r",
        "nslkdd",
        "--train-in",
        str(train_csv),
        "--device",
        device,
        "--log-level",
        "INFO",
        "--output-dir",
        str(output_dir),
    ]
    bootstrap = (
        "import importlib.util,os,sys;"
        f"_p={str(training_script)!r};"
        f"os.chdir({str(training_script.parent.parent)!r});"
        "_s=importlib.util.spec_from_file_location('_uqled_nsl_training',_p);"
        "_m=importlib.util.module_from_spec(_s);"
        "sys.modules[_s.name]=_m;"
        "_s.loader.exec_module(_m);"
        f"_m.MODEL_ORDER={models!r};"
        f"sys.argv={child_argv!r};"
        "_m.main()"
    )
    return [python_exe, "-c", bootstrap]


def run_training(
    python_exe: str,
    training_script: Path,
    train_csv: Path,
    output_dir: Path,
    log_file: Path,
    device: str,
) -> list[dict]:
    downstream_device = resolve_downstream_device(device)
    if downstream_device == "CPU":
        classical_output = run_step(
            f"model_training classical ({train_csv.stem})",
            _build_training_command(
                python_exe,
                training_script,
                train_csv,
                output_dir,
                CLASSICAL_MODELS,
                "CPU",
            ),
            log_file.with_name(f"{log_file.stem}_classical.log"),
        )
        dnn_output = run_step(
            f"model_training DNN ({train_csv.stem})",
            _build_training_command(
                python_exe,
                training_script,
                train_csv,
                output_dir,
                ["dnn"],
                "auto",
            ),
            log_file.with_name(f"{log_file.stem}_dnn.log"),
        )
        output = classical_output + "\n" + dnn_output
        log_file.write_text(output, encoding="utf-8")
    else:
        output = run_step(
            f"model_training ({train_csv.stem})",
            _build_training_command(
                python_exe,
                training_script,
                train_csv,
                output_dir,
                [model.lower() for model in MODEL_ORDER],
                downstream_device,
            ),
            log_file,
        )
    return parse_training_summary(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-seed UQ-LED CL-MCD-E benchmark for NSL-KDD")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--noise-rates", default="30,40,50,60,70")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benign-label", default="Benign")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scopes", default="both", choices=["both", "global", "benign_only"])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--n-splits", type=int, default=CANONICAL_CONFIG["n_splits"])
    parser.add_argument("--mc-passes", type=int, default=CANONICAL_CONFIG["mc_passes"])
    parser.add_argument("--dropout", type=float, default=CANONICAL_CONFIG["dropout"])
    parser.add_argument("--hidden-dims", default=CANONICAL_CONFIG["hidden_dims"])
    parser.add_argument("--batch-size", type=int, default=CANONICAL_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CANONICAL_CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=CANONICAL_CONFIG["patience"])
    parser.add_argument("--validation-fraction", type=float, default=CANONICAL_CONFIG["validation_fraction"])
    parser.add_argument("--learning-rate", type=float, default=CANONICAL_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=CANONICAL_CONFIG["weight_decay"])
    parser.add_argument(
        "--clean-input",
        default=str(SCRIPT_DIR / "resources" / "NSLKDD" / "clean_merged" / "nslkdd_train_clean_merged.csv"),
    )
    parser.add_argument("--sota-dir", default=str(SCRIPT_DIR / "reports" / "sota_uqled"))
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()

    try:
        validate_single_seed(args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    clean_input = Path(args.clean_input).expanduser().resolve()
    if not clean_input.exists():
        raise SystemExit(f"Clean input not found: {clean_input}")
    noise_rates = parse_noise_rates_list(args.noise_rates)
    canonical = _is_canonical(args)
    if not canonical:
        print("[WARNING] Non-canonical UQ-LED configuration: intended only for smoke/debug runs.")

    noise_script = SCRIPT_DIR / "symmetric_label_noise.py"
    defense_script = SCRIPT_DIR / "uqled_defense.py"
    training_script = SCRIPT_DIR / "training" / "model_training.py"
    for required in (noise_script, defense_script, training_script):
        if not required.exists():
            raise SystemExit(f"Required script not found: {required}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.sota_dir) / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for noise_pct in noise_rates:
        tag = f"noise_{noise_pct}"
        noise_dir = root_dir / tag / "00_targeted_label_noise"
        defense_dir = root_dir / tag / "01_uqled_cl_mcd_e"
        global_training_dir = root_dir / tag / "02_global_model_training"
        benign_training_dir = root_dir / tag / "03_benign_only_model_training"
        for directory in (noise_dir, defense_dir, global_training_dir, benign_training_dir):
            directory.mkdir(parents=True, exist_ok=True)

        noisy_csv = noise_dir / f"nslkdd_train_clean_merged_noise_{noise_pct}.csv"
        run_step(
            f"targeted_label_noise ({tag}, seed={args.seed})",
            [
                args.python_exe,
                str(noise_script),
                "--input",
                str(clean_input),
                "--noise-rate",
                str(noise_pct / 100.0),
                "--seed",
                str(args.seed),
                "--benign-label",
                args.benign_label,
                "--keep-tracking",
                "--log-level",
                "INFO",
                "--output",
                str(noisy_csv),
            ],
            noise_dir / f"{tag}_targeted_label_noise.log",
        )

        global_csv = defense_dir / f"{tag}_uqled_global.csv"
        benign_csv = defense_dir / f"{tag}_uqled_benign_only.csv"
        report_json = defense_dir / f"{tag}_uqled_report.json"
        artifacts_npz = defense_dir / f"{tag}_uqled_oof_mcd.npz"
        run_step(
            f"UQ-LED CL-MCD-E ({tag})",
            [
                args.python_exe,
                str(defense_script),
                "--input",
                str(noisy_csv),
                "--output-global",
                str(global_csv),
                "--output-benign-only",
                str(benign_csv),
                "--report",
                str(report_json),
                "--artifacts",
                str(artifacts_npz),
                "--benign-label",
                args.benign_label,
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--n-splits",
                str(args.n_splits),
                "--mc-passes",
                str(args.mc_passes),
                "--dropout",
                str(args.dropout),
                "--hidden-dims",
                args.hidden_dims,
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--validation-fraction",
                str(args.validation_fraction),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
            ],
            defense_dir / f"{tag}_uqled.log",
        )

        scope_inputs: list[tuple[str, Path, Path]] = []
        if args.scopes in {"both", "global"}:
            scope_inputs.append(("global", global_csv, global_training_dir))
        if args.scopes in {"both", "benign_only"}:
            scope_inputs.append(("benign_only", benign_csv, benign_training_dir))
        for scope, train_csv, training_dir in scope_inputs:
            metrics = [] if args.skip_training else run_training(
                args.python_exe,
                training_script,
                train_csv,
                training_dir,
                training_dir / f"{tag}_{scope}_model_training.log",
                args.device,
            )
            if not args.skip_training:
                try:
                    require_complete_metrics(metrics, MODEL_ORDER, f"{tag}/{scope}")
                except RuntimeError as exc:
                    raise SystemExit(str(exc)) from exc
            if args.skip_training:
                metrics = [{"model": "SKIPPED", "accuracy": float("nan"), "f1_macro": float("nan")}]
            for row in metrics:
                row.update(
                    {
                        "dataset": "NSL-KDD",
                        "noise_pct": noise_pct,
                        "seed": args.seed,
                        "downstream_device": resolve_downstream_device(args.device),
                        "method": "UQ-LED",
                        "variant": "CL-MCD-E",
                        "scope": scope,
                        "canonical_config": canonical,
                        "noisy_file": str(noisy_csv),
                        "cleaned_train_file": str(train_csv),
                        "defense_report": str(report_json),
                        "oof_artifacts": str(artifacts_npz),
                    }
                )
                all_rows.append(row)

    summary_csv = Path(args.summary_csv) if args.summary_csv else root_dir / "sota_uqled_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
    print("=" * 100)
    print("[DONE] UQ-LED CL-MCD-E BENCHMARK COMPLETED (NSL-KDD)")
    print(f"[DONE] Canonical configuration: {canonical}")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] Report folder: {root_dir}")


if __name__ == "__main__":
    main()
