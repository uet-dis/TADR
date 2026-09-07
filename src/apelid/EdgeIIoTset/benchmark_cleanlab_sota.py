"""
Benchmark SOTA Cleanlab defense for EdgeIIoTset.

"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["XGB", "CATB", "BAGGING", "LGBM", "RF", "DNN"]


def run_step(step_name: str, cmd: list[str], log_file: Path) -> str:
    print("=" * 100)
    print(f"[RUN] {step_name}")
    print("[CMD] " + " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        print(combined)
        raise SystemExit(f"Step failed: {step_name}. See log: {log_file}")

    print(f"[OK] {step_name} -> log: {log_file}")
    return combined


def parse_training_summary(text: str) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(
        r"\b(XGB|CATB|BAGGING|HISTGBM|LGBM|RF|DNN)\s*-\s*"
        r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*F1\s*\(Macro\):\s*([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )

    found: dict[str, dict] = {}
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        model = m.group(1).upper()
        found[model] = {
            "model": model,
            "accuracy": float(m.group(2)),
            "f1_macro": float(m.group(3)),
        }

    for model in MODEL_ORDER:
        if model in found:
            rows.append(found[model])

    return rows


def parse_noise_rates_list(text: str) -> list[int]:
    default_rates: list[int] = [30, 40, 50, 60, 70]
    if text is None:
        return list(default_rates)
    vals: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0 or v > 100:
            raise ValueError(f"Invalid noise percentage: {v}")
        vals.append(v)
    vals = sorted(set(vals))
    return vals if vals else list(default_rates)


def run_training(python_exe: str, training_script: Path, train_csv: Path, output_dir: Path, log_file: Path, device: str) -> list[dict]:
    out = run_step(
        f"model_training ({train_csv.stem})",
        [
            python_exe,
            str(training_script),
            "--model",
            "all",
            "-r",
            "edgeiot",
            "--train-in",
            str(train_csv),
            "--device",
            str(device),
            "--log-level",
            "INFO",
            "--output-dir",
            str(output_dir),
        ],
        log_file,
    )
    return parse_training_summary(out)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Benchmark SOTA Cleanlab defense for EdgeIIoTset")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--noise-rates", type=str, default="30,40,50,60,70", help="Optional filter, e.g. '30,40,50'")
    parser.add_argument("--model-type", type=str, default="lgbm", choices=["lgbm", "rf"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--only-benign", action="store_true", help="Run Cleanlab sanitization only on current Normal rows")
    parser.add_argument("--benign-label", type=str, default="Normal")
    parser.add_argument("--device", type=str, default="auto", choices=["CPU", "auto"])
    parser.add_argument("--sota-dir", type=str, default=str(script_dir / "reports" / "sota_cleanlab_benign_only"))
    parser.add_argument("--summary-csv", type=str, default=None)
    default_clean_input = script_dir / "resources" / "edgeiot" / "clean_merged" / "edgeiot_train_clean_merged.csv"
    parser.add_argument("--clean-input", type=str, default=str(default_clean_input), help="Path to the clean EdgeIIoT training CSV")
    args = parser.parse_args()

    clean_input = Path(args.clean_input)
    if not clean_input.exists():
        raise SystemExit(f"Clean input not found: {clean_input}")

    noise_rates = parse_noise_rates_list(args.noise_rates)
    if not noise_rates:
        raise SystemExit("No noise rates to process")

    python_exe = args.python_exe
    sota_dir = Path(args.sota_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = sota_dir / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)

    symmetric_script = script_dir / "symmetric_label_noise.py"
    cleanlab_script = script_dir / "cleanlab_defense.py"
    training_script = script_dir / "model_training.py"
    if not symmetric_script.exists() or not cleanlab_script.exists() or not training_script.exists():
        raise SystemExit("Required script not found (symmetric_label_noise.py/cleanlab_defense.py/model_training.py)")

    all_rows: list[dict] = []

    for noise_pct in noise_rates:
        tag = f"noise_{noise_pct}"

        phase_dirs = {
            "00_symmetric": root_dir / tag / "00_symmetric_label_noise",
            "01_sanitization": root_dir / tag / "01_cleanlab_sanitization",
            "02_training": root_dir / tag / "02_model_training",
        }
        for phase_dir in phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)

        noise_output = phase_dirs["00_symmetric"] / f"{tag}_data.csv"
        symmetric_cmd = [
            python_exe,
            str(symmetric_script),
            "--input",
            str(clean_input),
            "--noise-rate",
            str(noise_pct / 100.0),
            "--keep-tracking",
            "--log-level",
            "INFO",
            "--output",
            str(noise_output),
        ]
        run_step(
            f"symmetric_label_noise ({tag})",
            symmetric_cmd,
            phase_dirs["00_symmetric"] / f"{tag}_symmetric_label_noise.log",
        )
        if not noise_output.exists():
            raise SystemExit(f"Symmetric output file not found: {noise_output}")

        sanitized_csv = phase_dirs["01_sanitization"] / f"{tag}_cleanlab_sanitized.csv"
        cleanlab_report = phase_dirs["01_sanitization"] / f"{tag}_cleanlab_report.json"

        cleanlab_cmd = [
            python_exe,
            str(cleanlab_script),
            "--input",
            str(noise_output),
            "--output",
            str(sanitized_csv),
            "--report",
            str(cleanlab_report),
            "--model-type",
            str(args.model_type),
            "--threshold",
            str(args.threshold),
        ]
        if args.only_benign:
            cleanlab_cmd.append("--only-benign")
            cleanlab_cmd.extend(["--benign-label", str(args.benign_label)])

        run_step(
            f"cleanlab_sanitization ({tag})",
            cleanlab_cmd,
            phase_dirs["01_sanitization"] / f"{tag}_cleanlab_sanitization.log",
        )

        metrics = run_training(
            python_exe,
            training_script,
            sanitized_csv,
            phase_dirs["02_training"],
            phase_dirs["02_training"] / f"{tag}_model_training.log",
            args.device,
        )

        for row in metrics:
            row["noise_pct"] = noise_pct
            row["stage"] = "sota_cleanlab"
            row["model_type"] = str(args.model_type)
            row["threshold"] = float(args.threshold)
            row["only_benign"] = bool(args.only_benign)
            row["clean_input"] = str(clean_input)
            row["generated_noisy_file"] = str(noise_output)
            row["input_noisy_file"] = str(noise_output)
            row["sanitized_train_file"] = str(sanitized_csv)
            row["cleanlab_report"] = str(cleanlab_report)
            all_rows.append(row)

    summary_csv = Path(args.summary_csv) if args.summary_csv else (root_dir / "sota_cleanlab_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        all_rows,
        columns=[
            "noise_pct",
            "stage",
            "model",
            "accuracy",
            "f1_macro",
            "model_type",
            "threshold",
            "only_benign",
            "input_noisy_file",
            "clean_input",
            "generated_noisy_file",
            "sanitized_train_file",
            "cleanlab_report",
        ],
    ).to_csv(summary_csv, index=False)

    print("=" * 100)
    print("[DONE] SOTA CLEANLAB BENCHMARK COMPLETED (EDGEIOTSET)")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] SOTA folder: {root_dir}")


if __name__ == "__main__":
    main()