"""Build claim-bounded manuscript text, figures, and numerical audit sources.

This program is intentionally gated on completion of the frozen W17 analysis.
It never reads source pixels or prediction payloads; it consumes only derived
aggregate tables and signed completeness receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
import pandas as pd  # noqa: E402

GENERATED = ROOT / "manuscript" / "generated"
FIGURES = ROOT / "manuscript" / "figures"
SOURCE_DATA = ROOT / "evidence" / "figure_source_data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def ci_phrase(row: pd.Series, low: str = "simultaneous_low", high: str = "simultaneous_high") -> str:
    estimate = float(row["estimate"])
    lower = float(row[low])
    upper = float(row[high])
    return f"{signed(estimate)} (CI {signed(lower)} to {signed(upper)})"


def direction(row: pd.Series) -> str:
    lower, upper = float(row["simultaneous_low"]), float(row["simultaneous_high"])
    if lower > 0:
        return "favoured NP over the comparator"
    if upper < 0:
        return "favoured the comparator over NP"
    return "was compatible with effects in both directions"


def require_receipts() -> dict[str, dict[str, Any]]:
    requirements = {
        "W11": (ROOT / "evidence/stronger_generator/W11_QUALITY_COMPLETENESS.json", {"PASS_WITH_SCOPE_LIMITATION"}),
        "W12": (ROOT / "evidence/detector_core/CORE_COMPLETENESS.json", {"COMPLETE", "PASS"}),
        "W14": (ROOT / "evidence/low_data/W14_DETECTOR_COMPLETENESS.json", {"COMPLETE", "PASS"}),
        "W15": (ROOT / "evidence/resolution/W15_COMPLETENESS.json", {"PASS"}),
        "W16": (ROOT / "evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json", {"COMPLETE", "PASS"}),
        "W17-test": (ROOT / "evidence/test_campaign/W17_TEST_COMPLETENESS.json", {"COMPLETE"}),
        "W17-statistics": (ROOT / "evidence/final_statistics/W17_STATISTICS_COMPLETENESS.json", {"PASS"}),
    }
    values: dict[str, dict[str, Any]] = {}
    errors = []
    for name, (path, accepted) in requirements.items():
        if not path.exists():
            errors.append(f"{name}: missing {path.relative_to(ROOT)}")
            continue
        value = load_json(path)
        if value.get("status") not in accepted:
            errors.append(f"{name}: status {value.get('status')} not in {sorted(accepted)}")
        values[name] = value
    if errors:
        raise RuntimeError("Submission artifact gate is not open:\n" + "\n".join(errors))
    return values


def save_figure_data(name: str, frame: pd.DataFrame) -> Path:
    path = SOURCE_DATA / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def configure_plots() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "figure.dpi": 160,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def build_figures(primary: pd.DataFrame, test: pd.DataFrame, low: pd.DataFrame,
                  grouped: pd.DataFrame, w09: pd.DataFrame, path: pd.DataFrame,
                  quality: pd.DataFrame, resolution: pd.DataFrame) -> list[Path]:
    plt = configure_plots()
    from matplotlib.ticker import FuncFormatter
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    colors = {"NP": "#B2182B", "NB": "#2166AC", "NF": "#4D9221", "N1": "#6A3D9A", "NR": "#E08214"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    aggregate = path.sort_values("proxy_lambda")
    axes[0].errorbar(aggregate["proxy_lambda"], aggregate["validation_mse_mean"],
                     yerr=aggregate["validation_mse_sd"], marker="o", capsize=3, color="#2166AC")
    axes[0].set_xscale("symlog", linthresh=0.0002)
    axes[0].set_xlabel("Optical-residual weight")
    axes[0].set_ylabel("Validation reconstruction MSE")
    axes[0].set_title("a  Regularization path")
    labels = ["Optical", "Boundary + moment", "Interaction"]
    y = np.arange(len(w09))
    axes[1].errorbar(w09["mean_validation_mse_effect"], y,
                     xerr=[w09["mean_validation_mse_effect"] - w09["student_t_95_ci_low"],
                           w09["student_t_95_ci_high"] - w09["mean_validation_mse_effect"]],
                     fmt="o", capsize=3, color="#B2182B")
    axes[1].axvline(0, color="0.45", lw=1)
    axes[1].set_yticks(y, labels)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value * 1e4:.1f}"))
    axes[1].set_xlabel(r"Paired validation MSE effect ($\times 10^{-4}$)")
    axes[1].set_title("b  Frozen 2 x 2 factorial")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        target = FIGURES / f"figure_1_generator_mechanism.{extension}"
        fig.savefig(target, bbox_inches="tight")
        outputs.append(target)
    plt.close(fig)
    save_figure_data("figure_1_regularization_path", aggregate)
    save_figure_data("figure_1_factorial_effects", w09)

    selected_quality = quality[["arm", "PFFD10_train_scaled_mean", "PFFD10_train_scaled_sd",
                                "FID_InceptionV3_Mixed6e_768_mean", "FID_InceptionV3_Mixed6e_768_sd"]].copy()
    selected_quality = selected_quality.sort_values("arm")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    x = np.arange(len(selected_quality))
    for axis, mean_col, sd_col, title, ylabel in (
        (axes[0], "PFFD10_train_scaled_mean", "PFFD10_train_scaled_sd", "a  Domain morphology distance", "PFFD10"),
        (axes[1], "FID_InceptionV3_Mixed6e_768_mean", "FID_InceptionV3_Mixed6e_768_sd", "b  ImageNet-feature distance", "FID (Mixed_6e)"),
    ):
        axis.bar(x, selected_quality[mean_col], yerr=selected_quality[sd_col], capsize=3,
                 color=[colors.get(arm, "#777777") for arm in selected_quality["arm"]])
        axis.set_xticks(x, selected_quality["arm"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        target = FIGURES / f"figure_2_synthetic_quality.{extension}"
        fig.savefig(target, bbox_inches="tight")
        outputs.append(target)
    plt.close(fig)
    save_figure_data("figure_2_synthetic_quality", selected_quality)

    primary_test = test[(test["source_block"] == "W12_core") & (test["family"] == "fasterrcnn")]
    arm_summary = primary_test.groupby("arm", as_index=False)["test_COCO_AP"].agg(["mean", "std"]).reset_index()
    arm_order = [arm for arm in ("N0", "N1", "NR", "NB", "NF", "NP", "NS", "ND") if arm in set(arm_summary["arm"])]
    arm_summary["order"] = arm_summary["arm"].map({arm: i for i, arm in enumerate(arm_order)})
    arm_summary = arm_summary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    x = np.arange(len(arm_summary))
    axes[0].errorbar(x, arm_summary["mean"], yerr=arm_summary["std"], fmt="o", capsize=3, color="#2166AC")
    axes[0].set_xticks(x, arm_summary["arm"], rotation=45)
    axes[0].set_ylabel("Held-out pooled COCO AP")
    axes[0].set_title("a  Primary detector arms")
    y = np.arange(len(primary))
    axes[1].errorbar(primary["estimate"], y,
                     xerr=[primary["estimate"] - primary["simultaneous_low"],
                           primary["simultaneous_high"] - primary["estimate"]],
                     fmt="o", capsize=3, color="#B2182B")
    axes[1].axvline(0, color="0.45", lw=1)
    axes[1].set_yticks(y, primary["contrast"])
    axes[1].set_xlabel("Paired AP difference (98.75% CI)")
    axes[1].set_title("b  Multiplicity-controlled contrasts")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        target = FIGURES / f"figure_3_primary_detection.{extension}"
        fig.savefig(target, bbox_inches="tight")
        outputs.append(target)
    plt.close(fig)
    save_figure_data("figure_3_arm_summary", arm_summary.drop(columns="order"))
    save_figure_data("figure_3_primary_contrasts", primary)

    resolution_effects = resolution.pivot_table(index=["run_id", "arm", "pipeline_seed"],
                                                columns="resolution_path", values="COCO_AP").reset_index()
    resolution_effects["common64_minus_native300"] = resolution_effects["common64"] - resolution_effects["native300"]
    resolution_summary = resolution_effects.groupby("arm", as_index=False)["common64_minus_native300"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
    for contrast, frame in low.groupby("contrast"):
        axes[0].plot(frame["subset_size"], frame["estimate"], marker="o", label=contrast)
        axes[0].fill_between(frame["subset_size"].astype(float), frame["ordinary_low"].astype(float),
                             frame["ordinary_high"].astype(float), alpha=0.12)
    axes[0].axhline(0, color="0.45", lw=1)
    axes[0].set_xlabel("Training specimens")
    axes[0].set_ylabel("NP contrast in held-out AP")
    axes[0].set_title("a  Low-data sensitivity")
    axes[0].legend(frameon=False)
    gy = np.arange(len(grouped))
    axes[1].errorbar(grouped["estimate"], gy,
                     xerr=[grouped["estimate"] - grouped["minimum_partition_effect"],
                           grouped["maximum_partition_effect"] - grouped["estimate"]], fmt="o", capsize=3)
    axes[1].axvline(0, color="0.45", lw=1)
    axes[1].set_yticks(gy, grouped["contrast"])
    axes[1].set_xlabel("OOF AP difference (partition range)")
    axes[1].set_title("b  Development OOF")
    rx = np.arange(len(resolution_summary))
    axes[2].bar(rx, resolution_summary["mean"], yerr=resolution_summary["std"], capsize=3, color="#4D9221")
    axes[2].axhline(0, color="0.45", lw=1)
    axes[2].set_xticks(rx, resolution_summary["arm"])
    axes[2].set_ylabel("Common64 - native300 AP")
    axes[2].set_title("c  Validation resolution path")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        target = FIGURES / f"figure_4_sensitivity.{extension}"
        fig.savefig(target, bbox_inches="tight")
        outputs.append(target)
    plt.close(fig)
    save_figure_data("figure_4_low_data", low)
    save_figure_data("figure_4_grouped_oof", grouped)
    save_figure_data("figure_4_resolution", resolution_summary)
    return outputs


def latex_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 4) -> str:
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule",
             " & ".join(tex(label) for _, label in columns) + r" \\", "\\midrule"]
    for _, row in frame.iterrows():
        values = []
        for name, _ in columns:
            value = row[name]
            if isinstance(value, (float, np.floating)):
                values.append(num(value, digits))
            else:
                values.append(tex(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_text(primary: pd.DataFrame, test: pd.DataFrame, low: pd.DataFrame,
               grouped: pd.DataFrame, w09: pd.DataFrame, path: pd.DataFrame,
               smoothing: pd.DataFrame, quality: pd.DataFrame, resolution: pd.DataFrame,
               receipts: dict[str, dict[str, Any]]) -> dict[str, str]:
    primary = primary.set_index("contrast", drop=False)
    test_primary = test[(test["source_block"] == "W12_core") & (test["family"] == "fasterrcnn")]
    arm_means = test_primary.groupby("arm")["test_COCO_AP"].mean().to_dict()
    strongest = max(arm_means, key=arm_means.get)
    all_cross = all(float(row["simultaneous_low"]) <= 0 <= float(row["simultaneous_high"]) for _, row in primary.iterrows())
    contrast_sentence = "; ".join(
        f"{name} {ci_phrase(row)}" for name, row in primary.iterrows()
    )
    abstract_interpretation = (
        "All four simultaneous intervals included zero, so the held-out results did not isolate a robust NP advantage over the predeclared controls."
        if all_cross else
        "At least one multiplicity-controlled interval excluded zero; the direction and scope are reported without equivalence or external-validity claims."
    )
    abstract = (
        "Synthetic imagery may expand training data for laser powder bed fusion monitoring, but an image-domain residual is not evidence of calibrated process physics. "
        "We tested a conditional coordinate-field generator with a dimensionless optical residual against data-only, donor-matched non-neural, smoothing, diffusion, conventional-augmentation, and additional-real-exposure controls. "
        "The frozen study used 5,632 images from 176 specimens, specimen-grouped train/validation/test roles, ten paired full-pipeline replicates, and pooled COCO AP@[.50:.95] on one locked held-out campaign. "
        f"The four primary paired AP contrasts were {contrast_sentence}. {abstract_interpretation} "
        "Low-data, resolution, architecture, and grouped out-of-fold analyses were secondary. All conclusions concern the inherited operational bounding-box target; particle semantics and external validity were not independently established."
    )

    proxy = w09[w09["effect"] == "proxy_main_effect"].iloc[0]
    boundary = w09[w09["effect"] == "boundary_moment_main_effect"].iloc[0]
    best_path = path.loc[path["validation_mse_mean"].idxmin()]
    smooth = smoothing.iloc[0]
    quality_by_arm = quality.set_index("arm")
    nd_quality = quality_by_arm.loc["ND"]
    non_nd_quality = quality_by_arm.drop(index="ND")
    nd_quality_sentence = (
        "Among the five synthetic arms, ND had the largest aggregate distances "
        f"(PFFD10 {num(nd_quality['PFFD10_train_scaled_mean'], 3)}; "
        f"FID {num(nd_quality['FID_InceptionV3_Mixed6e_768_mean'], 3)}); "
        "the corresponding ranges across NB, NF, NP, and NS were "
        f"{num(non_nd_quality['PFFD10_train_scaled_mean'].min(), 3)}--"
        f"{num(non_nd_quality['PFFD10_train_scaled_mean'].max(), 3)} and "
        f"{num(non_nd_quality['FID_InceptionV3_Mixed6e_768_mean'].min(), 3)}--"
        f"{num(non_nd_quality['FID_InceptionV3_Mixed6e_768_mean'].max(), 3)}, respectively. "
    )
    generator_results = (
        "\\subsection{Frozen generator diagnostics}\n"
        f"Across ten paired factorial seeds, the optical main effect on validation reconstruction MSE was {signed(proxy['mean_validation_mse_effect'], 6)} "
        f"(95\\% CI {signed(proxy['student_t_95_ci_low'], 6)} to {signed(proxy['student_t_95_ci_high'], 6)}; Holm-adjusted $p={num(proxy['holm_adjusted_p_three_effects'], 3)}$). "
        f"The boundary-and-moment main effect was {signed(boundary['mean_validation_mse_effect'], 6)} "
        f"(95\\% CI {signed(boundary['student_t_95_ci_low'], 6)} to {signed(boundary['student_t_95_ci_high'], 6)}; adjusted $p={num(boundary['holm_adjusted_p_three_effects'], 3)}$). "
        f"The validation-only path reached its lowest mean reconstruction MSE ({num(best_path['validation_mse_mean'], 6)}) at optical weight {best_path['proxy_lambda']}. "
        f"The closest Gaussian/heat smoothing control remained {num(100*smooth['absolute_relative_gap'], 1)}\\% from that target and therefore failed the frozen 5\\% matching tolerance (Fig.~1). These are development diagnostics, not held-out detector outcomes.\n\n"
        "The common-roster quality analysis compared domain morphology, ImageNet-feature distribution, texture, diversity, and automated similarity screens (Fig.~2). "
        + nd_quality_sentence
        + "These measures were interpreted descriptively: ImageNet features are not domain-calibrated, and the automated screen can neither prove nor exclude memorization. "
        "Generated arrays and inherited boxes passed roster alignment, but independent physical label validation was outside scope."
    )
    primary_lines = []
    for name, row in primary.iterrows():
        primary_lines.append(f"{name} was {ci_phrase(row)} and {direction(row)}")
    primary_results = (
        "\\subsection{Held-out detector utility}\n"
        f"The highest unadjusted mean held-out AP among the eight primary arms was observed for {strongest} ({num(arm_means[strongest])}); this ranking was not itself a confirmatory comparison. "
        + "; ".join(primary_lines) + ". "
        "Intervals used the same resampled specimens and pipeline indices for every arm and a 98.75\\% individual confidence level for the four-comparison family (Fig.~3). "
        "The fast pooled-AP implementation agreed with pycocotools in all 50 primary cells before resampling."
    )
    low_summary = low.groupby("subset_size")["estimate"].agg(["min", "max"]).reset_index()
    low_phrase = "; ".join(
        f"$n={int(row.subset_size)}$: {signed(row['min'])} to {signed(row['max'])}" for _, row in low_summary.iterrows()
    )
    grouped_phrase = "; ".join(
        f"{row.contrast} {signed(row.estimate)} (partition range {signed(row.minimum_partition_effect)} to {signed(row.maximum_partition_effect)})"
        for _, row in grouped.iterrows()
    )
    res_pivot = resolution.pivot_table(index=["run_id", "arm", "pipeline_seed"], columns="resolution_path", values="COCO_AP").reset_index()
    res_pivot["delta"] = res_pivot["common64"] - res_pivot["native300"]
    res_summary = res_pivot.groupby("arm")["delta"].mean().to_dict()
    sensitivity = (
        "\\subsection{Sensitivity and internal robustness}\n"
        f"Across the predeclared low-data contrasts, mean NP effects ranged as follows ({low_phrase}); these 95\\% intervals were secondary and descriptive. "
        f"The three-partition development-cohort grouped out-of-fold effects were {grouped_phrase}. "
        "Because those predictions reused the development cohort through grouped out-of-fold construction, they provide an internal robustness check rather than external validation. "
        "The validation-only common-64 resolution pathway changed mean AP relative to native-300 evaluation by "
        + ", ".join(f"{arm} {signed(value)}" for arm, value in sorted(res_summary.items()))
        + "; the unavailable 1024-pixel branch was not inferred (Fig.~4)."
    )
    results = "\n\n".join([generator_results, primary_results, sensitivity])

    directions = [direction(row) for _, row in primary.iterrows()]
    if all_cross:
        central = (
            "The central result is cautionary: under donor, label, exposure, and compute controls, the optical-residual arm did not show a multiplicity-robust held-out advantage over any of the four predeclared comparators. "
            "This does not demonstrate equivalence or absence of a practically meaningful effect; the simultaneous intervals define the remaining uncertainty."
        )
    else:
        central = (
            "At least one predeclared multiplicity-controlled contrast separated from zero. The result supports only the observed internal operational-target comparison and does not validate the optical residual as process physics."
        )
    discussion = (
        central + " The generator diagnostics reinforce the distinction between numerical regularization and downstream utility: the validation reconstruction path, morphology distances, and detector endpoint address different questions and need not rank methods identically. "
        "The donor-matched non-neural and additional-real-exposure arms are particularly important because gains over a weak real-only baseline can otherwise be attributed to target reuse or additional optimizer exposure.\n\n"
        "Three limitations set the boundary of interpretation. First, the inherited box is an operational region rather than an independently adjudicated particle label; no claim is made about particle count, streak identity, or thermophysical state. Second, the 176-specimen corpus provides a specimen-grouped internal hold-out but no independent machine, build, site, material, or acquisition campaign, so external validity remains unknown. Third, the coordinate field operates at 64 pixels and the higher-resolution branch was unsupported by native information; the resolution audit quantifies a pathway effect without recovering absent detail.\n\n"
        "Within those limits, the study contributes a falsifiable controlled evaluation rather than a physics-validity claim. Future work should obtain independently adjudicated physical targets and a prospectively acquired external cohort before using the residual or detector for process inference. A calibrated time-resolved measurement would also be required before replacing the optical operator with a governing thermal or flow residual."
    )

    figure_legends = (
        "\\textbf{Figure 1. Generator mechanism diagnostics.} (a) Validation reconstruction MSE across the frozen optical-weight path (mean and standard deviation over three seeds). (b) Paired $2\\times2$ factorial effects with 95\\% Student-$t$ intervals over ten seeds. Lower MSE is better; neither panel uses held-out test outcomes.\\par\n"
        "\\textbf{Figure 2. Synthetic-image quality diagnostics.} Mean and standard deviation over ten paired seeds for (a) the training-scaled ten-feature morphology Fr\\'echet distance (PFFD10) and (b) Fr\\'echet distance in ImageNet Inception-v3 Mixed\\_6e features. Lower values indicate closer aggregate distributions but are not measures of physical validity.\\par\n"
        "\\textbf{Figure 3. Frozen primary detector results.} (a) Mean and standard deviation of held-out pooled COCO AP@[.50:.95] across ten full-pipeline replicates. (b) Four paired primary contrasts with Bonferroni-adjusted 98.75\\% individual bootstrap intervals. Positive values favour NP.\\par\n"
        "\\textbf{Figure 4. Secondary sensitivity analyses.} (a) Low-data held-out effects with ordinary 95\\% intervals. (b) Grouped out-of-fold development-cohort effects with ranges across three partition seeds. (c) Validation AP change after the common-64 information pathway relative to native-300 evaluation."
    )
    macros = (
        r"\newcommand{\ReleaseURL}{\url{https://github.com/Jinhong-Yang/scientific-reports-lpbf-spatter-public}}" + "\n" +
        r"\newcommand{\ReleaseVersion}{v1.0.0}" + "\n"
    )

    quality_table = latex_table(quality, [
        ("arm", "Arm"), ("PFFD10_train_scaled_mean", "PFFD10"),
        ("FID_InceptionV3_Mixed6e_768_mean", "FID768"),
        ("KID_InceptionV3_Mixed6e_768_mean", "KID768"),
        ("memorization_screen_flag_rate_mean", "Screen flag rate"),
    ], 4)
    generator_table = latex_table(path, [
        ("proxy_lambda", "Optical weight"), ("seeds", "Seeds"),
        ("validation_mse_mean", "Validation MSE"), ("validation_mse_sd", "SD"),
    ], 6)
    primary_table = latex_table(primary.reset_index(drop=True), [
        ("contrast", "Contrast"), ("estimate", "Estimate"),
        ("ordinary_low", "95% low"), ("ordinary_high", "95% high"),
        ("simultaneous_low", "98.75% low"), ("simultaneous_high", "98.75% high"),
    ], 4)
    low_table = latex_table(low, [
        ("subset_size", "Specimens"), ("contrast", "Contrast"), ("estimate", "Estimate"),
        ("ordinary_low", "95% low"), ("ordinary_high", "95% high"),
    ], 4)
    grouped_table = latex_table(grouped, [
        ("contrast", "Contrast"), ("estimate", "Estimate"),
        ("minimum_partition_effect", "Minimum"), ("maximum_partition_effect", "Maximum"),
    ], 4)
    detector_supp = (
        f"The W12 core matrix contained {int(receipts['W12']['completed_trajectories'])} completed detector trajectories; "
        f"the W14 low-data matrix contained {int(receipts['W14']['completed_trajectories'])}; "
        f"and the single W17 campaign evaluated {int(receipts['W17-test']['completed_evaluations'])} frozen checkpoints. "
        "Every planned, failed, interrupted, and excluded trajectory is represented in the accompanying ledgers."
    )
    statistics_supp = (
        "\\subsection{Primary held-out comparisons}\n" + primary_table +
        "\n\\subsection{Low-data held-out comparisons}\n" + low_table +
        "\n\\subsection{Development grouped out-of-fold comparisons}\n" + grouped_table
    )
    reproducibility_supp = (
        "All final numerical statements were generated from aggregate evidence files after W17 completion. "
        "The release manifest records each included file's SHA-256 digest. Licensed source pixels, model checkpoints, predictions, local absolute paths, and secrets are excluded. "
        "The N2 test-access ledger records one pre-unlock display of historical specimen identifiers from a combined group-view manifest; no held-out pixels, coordinates, predictions, metrics, or outcomes were accessed, and all development folds used separate train and validation manifests."
    )
    return {
        "abstract.tex": abstract,
        "results.tex": results + "\n\n\\begin{figure}[p]\n\\centering\\includegraphics[width=\\linewidth]{figures/figure_1_generator_mechanism.pdf}\\caption{Generator mechanism diagnostics.}\\end{figure}\n"
                       "\\begin{figure}[p]\n\\centering\\includegraphics[width=\\linewidth]{figures/figure_2_synthetic_quality.pdf}\\caption{Synthetic-image quality diagnostics.}\\end{figure}\n"
                       "\\begin{figure}[p]\n\\centering\\includegraphics[width=\\linewidth]{figures/figure_3_primary_detection.pdf}\\caption{Frozen primary detector results.}\\end{figure}\n"
                       "\\begin{figure}[p]\n\\centering\\includegraphics[width=\\linewidth]{figures/figure_4_sensitivity.pdf}\\caption{Secondary sensitivity analyses.}\\end{figure}",
        "discussion.tex": discussion,
        "figure_legends.tex": figure_legends,
        "results_macros.tex": macros,
        "supplement_generator.tex": generator_table,
        "supplement_quality.tex": quality_table,
        "supplement_detector.tex": detector_supp,
        "supplement_statistics.tex": statistics_supp,
        "supplement_reproducibility.tex": reproducibility_supp,
    }


def build() -> dict[str, Any]:
    receipts = require_receipts()
    primary = pd.read_parquet(ROOT / "evidence/final_statistics/primary_comparisons.parquet")
    low = pd.read_parquet(ROOT / "evidence/final_statistics/low_data_comparisons.parquet")
    grouped = pd.read_parquet(ROOT / "evidence/final_statistics/grouped_oof_comparisons.parquet")
    test = pd.read_parquet(ROOT / "evidence/test_campaign/test_metrics.parquet")
    resolution = pd.read_parquet(ROOT / "evidence/resolution/resolution_sensitivity.parquet")
    w09 = pd.read_csv(ROOT / "evidence/generator_factorial/factorial_effects_summary.csv")
    path = pd.read_csv(ROOT / "evidence/prior_sweep/regularization_path_aggregate.csv")
    smoothing = pd.read_csv(ROOT / "evidence/prior_sweep/smoothing_match.csv")
    quality = pd.read_csv(ROOT / "evidence/stronger_generator/quality_arm_summary.csv")
    figure_paths = build_figures(primary, test, low, grouped, w09, path, quality, resolution)
    fragments = build_text(primary, test, low, grouped, w09, path, smoothing, quality, resolution, receipts)
    for name, value in fragments.items():
        atomic_text(GENERATED / name, value)
    contrast_summary = "; ".join(
        f"{row.contrast} {signed(row.estimate)} (98.75% CI {signed(row.simultaneous_low)} to {signed(row.simultaneous_high)})"
        for _, row in primary.iterrows()
    )
    public_url = "https://github.com/Jinhong-Yang/scientific-reports-lpbf-spatter-public"
    release_notes = f"""# Scientific Reports LPBF study - v1.0.0

This release is the source-data-free reproducibility and submission-preparation package for *Optical residual regularization for LPBF spatter synthesis and detection*.

Primary held-out paired AP contrasts: {contrast_summary}.

The release contains frozen configurations, aggregate and specimen-level numerical evidence permitted by the publication audit, figure source data, manuscript sources, tests, and checksum manifests. Licensed AI-Hub source pixels, model checkpoints, prediction payloads, private paths, credentials, and the excluded independent-review materials are not included.

All results concern the operational `inherited_bbox_region` target. The study does not establish independently adjudicated particle semantics or external validity, and confidence intervals crossing zero are not interpreted as evidence of equivalence or no effect. This is a research artifact and submission draft, not a journal acceptance or a deployed process-control system.
"""
    readme = f"""# Optical residual regularization for LPBF spatter synthesis and detection

This repository contains the frozen, source-data-free reproducibility materials and Scientific Reports submission draft for the LPBF synthetic-image control study.

## Study boundary

The N2 study uses 5,632 images from 176 specimens with specimen-grouped training, validation, and one locked internal held-out campaign. Results concern the inherited operational `inherited_bbox_region` target. Independent human label review was excluded by the project decision owner, and no external cohort was available; physical particle semantics and external validity are therefore not claimed.

## Reproduce the public package

Create a Python 3.11 environment, install `requirements-reporting.txt`, and run `reproduce.ps1` on Windows or `reproduce.sh` on POSIX. The default route verifies the archive and public-package tests. The optional manuscript rebuild consumes only released aggregate matrices and does not recompute the bootstrap from excluded prediction payloads.

## Availability

- Repository: {public_url}
- Immutable release: `v1.0.0`
- Source dataset: AI-Hub Metal 3D-Printing Spark Image Data, dataset 71476; obtain separately under the provider's terms.
- Raw source pixels, checkpoints, and predictions are not redistributed.
- No project-wide open-source license is asserted in this release; third-party components retain their own terms.

See `REPRODUCIBILITY_GUIDE.md`, `DATA_AVAILABILITY.md`, `CODE_AVAILABILITY.md`, and `docs/SUBMISSION_READINESS.md` for the audit boundary and submission status.
"""
    code_availability = f"""# Code availability

The custom code, frozen configuration, aggregate evidence, manuscript sources, and public-package tests are available at {public_url} in immutable release `v1.0.0`. The release manifest and SHA-256 ledger identify the archived version used for the manuscript.

Large model weights, licensed source images, local environments, dependency vendors, secrets, caches, raw runtime checkpoints, and prediction payloads are excluded. No project-wide open-source license is asserted; availability for inspection does not grant rights beyond applicable repository and third-party terms.
"""
    data_availability = f"""# Data availability

The source image pixels originate from the AI-Hub Metal 3D-Printing Spark Image Data resource (dataset 71476) and are not redistributed. Users must obtain the source under the provider's current terms. The public release at {public_url}, version `v1.0.0`, contains shareable split metadata, aggregate numerical results, bootstrap outputs, figure source data, artifact hashes, and reconstruction instructions.

The study corpus contains 5,632 images from 176 specimens: 112 training, 32 validation, and 32 internal held-out specimens, with views kept together. Historical work had already used the held-out role, and no qualified same-task external cohort was available. The release therefore supports internal reproducibility but does not establish a pristine external validation or independently adjudicated physical particle semantics.
"""
    support_docs = {
        ROOT / "docs/GITHUB_RELEASE_NOTES.md": release_notes,
        ROOT / "README.md": readme,
        ROOT / "CODE_AVAILABILITY.md": code_availability,
        ROOT / "DATA_AVAILABILITY.md": data_availability,
    }
    for support_path, value in support_docs.items():
        atomic_text(support_path, value)
    numeric_map = {
        "schema_version": 1,
        "status": "PASS",
        "primary_contrasts": primary.to_dict("records"),
        "primary_arm_means": test[(test["source_block"] == "W12_core") & (test["family"] == "fasterrcnn")].groupby("arm")["test_COCO_AP"].mean().to_dict(),
        "source_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                ROOT / "evidence/final_statistics/primary_comparisons.parquet",
                ROOT / "evidence/final_statistics/low_data_comparisons.parquet",
                ROOT / "evidence/final_statistics/grouped_oof_comparisons.parquet",
                ROOT / "evidence/test_campaign/test_metrics.parquet",
                ROOT / "evidence/resolution/resolution_sensitivity.parquet",
                ROOT / "evidence/generator_factorial/factorial_effects_summary.csv",
                ROOT / "evidence/prior_sweep/regularization_path_aggregate.csv",
                ROOT / "evidence/stronger_generator/quality_arm_summary.csv",
            )
        },
        "generated_hashes": {name: sha256_file(GENERATED / name) for name in fragments},
        "support_document_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in support_docs
        },
        "figure_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in figure_paths},
        "test_payload_read_by_builder": False,
    }
    atomic_json(ROOT / "evidence/manuscript/MANUSCRIPT_NUMERIC_MAP.json", numeric_map)
    return numeric_map


def preflight() -> dict[str, Any]:
    try:
        values = require_receipts()
        return {"status": "PASS", "receipts": {name: value["status"] for name, value in values.items()}}
    except RuntimeError as error:
        return {"status": "WAITING_FOR_RESULTS", "reason": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--build", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
