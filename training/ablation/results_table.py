"""
Generate publication-ready ablation results tables.

Outputs:
  - LaTeX table (paste directly into paper)
  - Markdown table (for README)
  - Interactive HTML table (for website)

All styled using the 6-color project palette.

Usage:
    python results_table.py --results_dir training/ablation/results \
                            --output_dir training/ablation/tables
"""

import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger("results_table")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─── Project Color Palette ────────────────────────────────────────────────────
PALETTE = {
    "mint": "#F1F6F4",
    "gold": "#FFC801",
    "teal": "#114C5A",
    "sage": "#D9E8E2",
    "orange": "#FF9932",
    "dark": "#172B36",
}

STUDY_TITLES: dict[str, str] = {
    "base_model": "Ablation 1: Base Model Comparison",
    "lora_rank": "Ablation 2: LoRA Rank (r) Search",
    "data_fraction": "Ablation 3: Training Data Volume",
    "language_exclusion": "Ablation 4: Language Exclusion (Cross-Lingual Transfer)",
}


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all ablation JSON result files from a directory.

    Args:
        results_dir: Directory containing study_name.json files.

    Returns:
        Dict mapping study name to list of result dicts.
    """
    all_results: dict[str, list[dict]] = {}

    for json_file in sorted(results_dir.glob("*.json")):
        study_name = json_file.stem
        with open(json_file, encoding="utf-8") as fh:
            results = json.load(fh)
        all_results[study_name] = results
        logger.info("Loaded %d results for study: %s", len(results), study_name)

    return all_results


def find_best_row(results: list[dict]) -> int:
    """Find the index of the result with highest f1_macro.

    Args:
        results: List of ablation result dicts.

    Returns:
        Index of best-performing configuration.
    """
    return max(range(len(results)), key=lambda i: results[i].get("f1_macro", 0.0))


def to_latex_table(study_name: str, results: list[dict]) -> str:
    """Generate a LaTeX table for one ablation study.

    Args:
        study_name: Study identifier key.
        results: List of result dicts.

    Returns:
        LaTeX table string.
    """
    best_idx = find_best_row(results)
    title = STUDY_TITLES.get(study_name, study_name)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + title + r"}",
        r"\label{tab:" + study_name + r"}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Overall F1} & \textbf{Hindi F1} & "
        r"\textbf{Tamil F1} & \textbf{Bengali F1} & \textbf{Code-Mix F1} & \textbf{Params (M)} \\",
        r"\midrule",
    ]

    for i, res in enumerate(results):
        params_m = res.get("trainable_params", 0) / 1e6
        row = (
            f"{'\\textbf{' if i == best_idx else ''}"
            f"{res['config_name']}"
            f"{'} ↑' if i == best_idx else ''}"
            f" & {res.get('f1_macro', 0.0):.1f}"
            f" & {res.get('f1_hindi', 0.0):.1f}"
            f" & {res.get('f1_tamil', 0.0):.1f}"
            f" & {res.get('f1_bengali', 0.0):.1f}"
            f" & {res.get('f1_code_mix', 0.0):.1f}"
            f" & {params_m:.1f}"
            r" \\"
        )
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def to_markdown_table(study_name: str, results: list[dict]) -> str:
    """Generate a Markdown table for one ablation study.

    Args:
        study_name: Study identifier key.
        results: List of result dicts.

    Returns:
        Markdown table string.
    """
    title = STUDY_TITLES.get(study_name, study_name)
    best_idx = find_best_row(results)

    lines = [
        f"### {title}",
        "",
        "| Config | Overall F1 | Hindi F1 | Tamil F1 | Bengali F1 | Code-Mix F1 | Params (M) |",
        "|--------|-----------|----------|----------|------------|-------------|------------|",
    ]

    for i, res in enumerate(results):
        params_m = res.get("trainable_params", 0) / 1e6
        marker = " ⭐" if i == best_idx else ""
        lines.append(
            f"| {res['config_name']}{marker}"
            f" | **{res.get('f1_macro', 0.0):.1f}**" if i == best_idx
            else f" | {res.get('f1_macro', 0.0):.1f}"
            + f" | {res.get('f1_hindi', 0.0):.1f}"
            + f" | {res.get('f1_tamil', 0.0):.1f}"
            + f" | {res.get('f1_bengali', 0.0):.1f}"
            + f" | {res.get('f1_code_mix', 0.0):.1f}"
            + f" | {params_m:.1f} |"
        )

    return "\n".join(lines)


def to_html_table(study_name: str, results: list[dict]) -> str:
    """Generate a styled HTML table for website embedding.

    Uses the 6-color design palette with sortable columns.

    Args:
        study_name: Study identifier key.
        results: List of result dicts.

    Returns:
        Self-contained HTML string with inline CSS.
    """
    title = STUDY_TITLES.get(study_name, study_name)
    best_idx = find_best_row(results)

    rows_html = ""
    for i, res in enumerate(results):
        params_m = res.get("trainable_params", 0) / 1e6
        is_best = i == best_idx
        row_bg = PALETTE["gold"] if is_best else (
            PALETTE["sage"] if i % 2 == 0 else PALETTE["mint"]
        )
        row_weight = "bold" if is_best else "normal"
        rows_html += f"""
        <tr style="background:{row_bg}; font-weight:{row_weight};">
          <td style="padding:8px 12px;">{res['config_name']}{'  ★' if is_best else ''}</td>
          <td style="padding:8px 12px; text-align:center;">{res.get('f1_macro', 0.0):.1f}</td>
          <td style="padding:8px 12px; text-align:center;">{res.get('f1_hindi', 0.0):.1f}</td>
          <td style="padding:8px 12px; text-align:center;">{res.get('f1_tamil', 0.0):.1f}</td>
          <td style="padding:8px 12px; text-align:center;">{res.get('f1_bengali', 0.0):.1f}</td>
          <td style="padding:8px 12px; text-align:center;">{res.get('f1_code_mix', 0.0):.1f}</td>
          <td style="padding:8px 12px; text-align:center;">{params_m:.1f}M</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <style>
    body {{ font-family: 'Inter', sans-serif; background: {PALETTE['mint']}; color: {PALETTE['dark']}; }}
    .table-wrapper {{ max-width: 900px; margin: 24px auto; border-radius: 12px; overflow: hidden;
                      box-shadow: 0 2px 12px rgba(23,43,54,0.10); }}
    h2 {{ color: {PALETTE['teal']}; margin-bottom: 12px; font-size: 1.1rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead tr {{ background: {PALETTE['teal']}; color: {PALETTE['mint']}; }}
    thead th {{ padding: 10px 12px; text-align: left; font-weight: 600; font-size: 0.85rem; }}
    tbody tr:hover {{ filter: brightness(0.97); }}
    td {{ border-bottom: 1px solid {PALETTE['sage']}; font-size: 0.9rem; color: {PALETTE['dark']}; }}
  </style>
</head>
<body>
  <div class="table-wrapper">
    <h2>{title}</h2>
    <table>
      <thead>
        <tr>
          <th>Config</th>
          <th>Overall F1</th>
          <th>Hindi F1</th>
          <th>Tamil F1</th>
          <th>Bengali F1</th>
          <th>Code-Mix F1</th>
          <th>Params</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""
    return html


def generate_all_tables(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate LaTeX, Markdown, and HTML tables for all ablation studies.

    Args:
        results_dir: Directory containing *.json ablation results.
        output_dir: Output directory for generated tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = load_results(results_dir)

    if not all_results:
        logger.warning("No results found in %s. Run run_ablation.py first.", results_dir)
        return

    # ── Combined Markdown (all studies) ───────────────────────────────────
    md_lines = ["# Ablation Study Results\n"]
    latex_lines = [
        r"\section{Ablation Studies}",
        r"The following tables report results for each ablation study.",
        r"Best-performing configurations are highlighted in bold.",
        "",
    ]

    for study_name, results in all_results.items():
        if not results:
            continue

        # LaTeX
        latex_lines.append(to_latex_table(study_name, results))
        latex_lines.append("")

        # Markdown
        md_lines.append(to_markdown_table(study_name, results))
        md_lines.append("")

        # HTML per study
        html_path = output_dir / f"{study_name}.html"
        html_path.write_text(to_html_table(study_name, results), encoding="utf-8")
        logger.info("HTML table saved → %s", html_path)

    # Save combined files
    (output_dir / "ablation_tables.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )
    logger.info("LaTeX tables → %s/ablation_tables.tex", output_dir)

    (output_dir / "ablation_tables.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    logger.info("Markdown tables → %s/ablation_tables.md", output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate ablation results tables")
    parser.add_argument("--results_dir", default="training/ablation/results")
    parser.add_argument("--output_dir", default="training/ablation/tables")
    args = parser.parse_args()

    generate_all_tables(
        Path(args.results_dir),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
