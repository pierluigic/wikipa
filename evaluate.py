import os
import json
from collections import defaultdict

import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):  
        return x

from whipa.code.scripts.metrics import STIPA_METRICS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 24})


OUT_HEATMAP_PER_BROAD = "heatmap_per.svg"
OUT_HEATMAP_PFER_BROAD = "heatmap_pfer.svg"
OUT_HEATMAP_PER_NARROW = "heatmap_per_narrow.svg"
OUT_HEATMAP_PFER_NARROW = "heatmap_pfer_narrow.svg"
OUT_HEATMAP_PER_COMBINED = "heatmap_per_combined.svg"
OUT_HEATMAP_PFER_COMBINED = "heatmap_pfer_combined.svg"
OUT_TSV_ROOT = "models_output"

GROUND_TRUTH_PATH = "ground_truth.jsonl"
PRED_DIR = "predictions"

OUT_LATEX_PER_BROAD = "results_table_per.tex.txt"
OUT_LATEX_PFER_BROAD = "results_table_pfer.tex.txt"
OUT_LATEX_PER_NARROW = "results_table_per_narrow.tex.txt"
OUT_LATEX_PFER_NARROW = "results_table_pfer_narrow.tex.txt"

max_per = 100
max_pfer = 30

model2name = {
    'zipa_large_crctc_0.5_scale_800000_avg10': 'ZIPA-CR-NS-LARGE',
    'zipa_large_noncausal_500000_avg10': 'ZIPA-T-LARGE',
    'zipa_small_crctc_extended_0.5_scale_700000_avg10': 'ZIPA-CR-NS-SMALL',
    'zipa_small_noncausal_500000_avg10': 'ZIPA-T-SMALL',
    'wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns': 'MultIPA',
    'whipa-base-cv': 'Whipa Base CV',
    'whipa-large-cv': 'Whipa Large CV',
    'lowhipa-base-cv': 'LoWhipa Base CV',
    'lowhipa-base-asc': 'LoWhipa Base ASC',
    'lowhipa-base-comb': 'LoWhipa Base Comb.',
    'lowhipa-large-cv': 'LoWhipa Large CV',
    'lowhipa-large-asc': 'LoWhipa Large ASC',
    'lowhipa-large-comb': 'LoWhipa Large Comb.',
}

def main():
    # ---------- Load ground truth (both broad and narrow) ----------
    broad_truth = []
    narrow_truth = []
    languages = []
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            b = row.get("broad")
            n = row.get("narrow")
            # Treat both as optional lists; take first if present
            broad_truth.append(b[0] if (isinstance(b, list) and len(b) > 0) else (b if isinstance(b, str) else None))
            narrow_truth.append(n[0] if (isinstance(n, list) and len(n) > 0) else (n if isinstance(n, str) else None))
            languages.append(row.get("language"))

    # ---------- Language filtering: only keep languages with >=50 samples ----------
    # ---------- Language filtering (per label set) ----------
    from collections import Counter

    def count_langs_with_label(truth_list):
        ctr = Counter()
        for lang, gold in zip(languages, truth_list):
            if gold is not None and len(str(gold)) > 0:
                ctr[lang] += 1
        return ctr

    counts_broad = count_langs_with_label(broad_truth)
    counts_narrow = count_langs_with_label(narrow_truth)

    eligible_langs_broad = {lang for lang, c in counts_broad.items() if c >= 50}
    eligible_langs_narrow = {lang for lang, c in counts_narrow.items() if c >= 50}

    print(f"Keeping {len(eligible_langs_broad)} languages with ≥50 BROAD samples:")
    print(sorted(eligible_langs_broad))
    print(f"Keeping {len(eligible_langs_narrow)} languages with ≥50 NARROW samples:")
    print(sorted(eligible_langs_narrow))

    # Stable order per label set: first appearance among rows that have that label
    def make_lang_order(eligible_set, truth_list):
        order, seen = [], set()
        for lang, gold in zip(languages, truth_list):
            if gold is None or len(str(gold)) == 0:
                continue
            if lang in eligible_set and lang not in seen:
                seen.add(lang)
                order.append(lang)
        return order

    lang_order_broad = make_lang_order(eligible_langs_broad, broad_truth)
    lang_order_narrow = make_lang_order(eligible_langs_narrow, narrow_truth)


    # ---------- Load model predictions ----------
    zipa_preds = {}
    if not os.path.isdir(PRED_DIR):
        raise FileNotFoundError(f"Prediction directory not found: {PRED_DIR}")
    for fname in os.listdir(PRED_DIR):
        if fname.endswith(".txt"):
            model = fname[:-4]
            with open(os.path.join(PRED_DIR, fname), "r", encoding="utf-8") as f:
                zipa_preds[model] = [line.rstrip("\n") for line in f]

    # Align lengths
    n = len(broad_truth)
    for m, preds in zipa_preds.items():
        if len(preds) < n:
            preds.extend([""] * (n - len(preds)))
        elif len(preds) > n:
            zipa_preds[m] = preds[:n]

    eval_metrics = STIPA_METRICS()


    def evaluate_labelset(label_truth_list, label_name: str, eligible_langs, lang_order):
        """
        Returns:
          df_per, df_pfer: DataFrames (languages x models) for this label set.
        Side effects:
          saves heatmaps, latex tables, and per-(model,language) TSVs.
        """
        # model -> lang -> list of {'per','pfer','gold','pred'}
        results_accum = defaultdict(lambda: defaultdict(list))

        for model in tqdm(list(zipa_preds.keys()), desc=f"Processing models ({label_name})"):
            preds = zipa_preds[model]
            for j in range(n):
                gold = label_truth_list[j]
                if gold is None or len(str(gold)) == 0:
                    continue

                if languages[j] not in eligible_langs:
                    continue

                if 'whipa' in model:
                    pred = preds[j][:20]
                else:
                    pred = preds[j]

                if pred is None or len(str(pred)) == 0:
                    continue

                res = eval_metrics.compute_all(pred=pred, gold=gold, char_based=False)
                per, pfer = res.get("per"), res.get("pfer")
                if per is not None and pfer is not None:
                    lang = languages[j]
                    results_accum[model][lang].append({
                        "per": per,
                        "pfer": pfer,
                        "gold": gold,
                        "pred": pred
                    })

        # ---------- NEW: write per-(model, language) TSVs ----------
        def _safe_filename(name: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(name))

        # Root: by label set, then model (friendly name), then <Language>.tsv
        tsv_root = os.path.join(OUT_TSV_ROOT, label_name)
        os.makedirs(tsv_root, exist_ok=True)

        for model_id, langs in results_accum.items():
            model_dir_name = model2name.get(model_id, model_id)
            model_dir = os.path.join(tsv_root, _safe_filename(model_dir_name))
            os.makedirs(model_dir, exist_ok=True)

            for lang, entries in langs.items():
                # Sort by PFER descending
                entries_sorted = sorted(
                    entries,
                    key=lambda e: (float("-inf") if e.get("pfer") is None else e["pfer"]),
                    reverse=True
                )
                out_path = os.path.join(model_dir, f"{_safe_filename(lang)}.tsv")
                with open(out_path, "w", encoding="utf-8") as f:
                    for e in entries_sorted:
                        pfer_val = e.get("pfer")
                        pfer_str = "--" if pfer_val is None else f"{pfer_val:.3f}"
                        gold_str = "" if e.get("gold") is None else str(e["gold"])
                        pred_str = "" if e.get("pred") is None else str(e["pred"])
                        f.write(f"{pfer_str}\t{gold_str}\t{pred_str}\n")

        print(f"Per-(model, language) TSVs written under: {tsv_root}")

        def mean_or_nan(vals):
            return float("nan") if not vals else sum(vals) / len(vals)


        def _map_and_consolidate_columns(df: pd.DataFrame) -> pd.DataFrame:
            # 1) map internal ids -> friendly names
            mapped = {col: model2name.get(col, col) for col in df.columns}
            df = df.rename(columns=mapped)

            # 2) consolidate duplicate-friendly-named columns by mean
            if len(set(df.columns)) != df.shape[1]:
                df = df.groupby(axis=1, level=0).mean(numeric_only=True)

            # 3) enforce desired order
            desired_order = [
                "MultIPA",
                "ZIPA-T-SMALL",
                "ZIPA-T-LARGE",
                "ZIPA-CR-NS-SMALL",
                "ZIPA-CR-NS-LARGE",
                "Whipa Base",
                "Whipa Large",
                "LoWhipa Base CV",
                "LoWhipa Large CV",
                "LoWhipa Base ASC",
                "LoWhipa Large ASC",
                "LoWhipa Base Combined",
                "LoWhipa Large Combined",
            ]

            front = [c for c in desired_order if c in df.columns]
            back = [c for c in df.columns if c not in set(front)]
            df = df.reindex(columns=front + back)

            return df

        def build_metric_df(metric_name: str) -> pd.DataFrame:
            """metric_name: 'per' or 'pfer'"""
            rows = []
            for lang in lang_order:
                vals = []
                for model in sorted(results_accum.keys()):
                    mvals = [e[metric_name] for e in results_accum[model].get(lang, [])]
                    vals.append(mean_or_nan(mvals))
                rows.append(vals)

            df = pd.DataFrame(rows, index=lang_order, columns=sorted(results_accum.keys())).round(3)

            # Drop languages that have no valid scores (all NaN) for this metric
            df = df.dropna(how="all")

            # Sort languages by their average error (lowest to highest)
            df["__avg__"] = df.mean(axis=1, skipna=True)
            df = df.sort_values("__avg__", ascending=True).drop(columns="__avg__")

            # rename columns using model2name and consolidate duplicates
            df = _map_and_consolidate_columns(df)

            return df

        # ---------- Build and sort DataFrames ----------
        df_per = build_metric_df("per")
        df_pfer = build_metric_df("pfer")

        print(f"\n== [{label_name.upper()}] Aggregated PER (mean) per Language × Model (sorted by avg PER) ==\n")
        print(df_per)
        print(f"\n== [{label_name.upper()}] Aggregated PFER (mean) per Language × Model (sorted by avg PFER) ==\n")
        print(df_pfer)

        # ---------- Heatmap helper ----------
        def plot_heatmap(df_input: pd.DataFrame, out_path: str, metric: str):
            """
            Expects df_input shaped Languages x Models.
            Renders as rows=models, cols=languages (so we transpose).
            """
            # models x languages
            H = df_input.T  # rows=models, cols=languages

            # Build figure size dynamically
            h, w = H.shape
            fig_w = max(6, min(18, 0.7 * w + 2.5))
            fig_h = max(4.5, min(18, 0.55 * h + 2.0))

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            if metric == 'PER':
                vmax = max_per
            else:
                vmax = max_pfer

            im = ax.imshow(
                H.values,
                aspect="auto",
                cmap="viridis_r",  # reversed viridis
                vmin=0,
                vmax=vmax,
            )

            # Tick labels
            ax.set_xticks(np.arange(w))
            ax.set_yticks(np.arange(h))
            ax.set_xticklabels(H.columns, rotation=90)
            ax.set_yticklabels(H.index)  # already mapped to friendly model names

            # Colorbar
            #cbar = fig.colorbar(im, ax=ax)
            #cbar.ax.set_ylabel("Error (%)", rotation=90, va="center")

            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)



        # ---------- Generate heatmaps ----------
        if label_name == "broad":
            plot_heatmap(df_per, OUT_HEATMAP_PER_BROAD, "PER")
            plot_heatmap(df_pfer, OUT_HEATMAP_PFER_BROAD, "PFER")
            print(f"PER heatmap saved to: {OUT_HEATMAP_PER_BROAD}")
            print(f"PFER heatmap saved to: {OUT_HEATMAP_PFER_BROAD}")
        else:
            plot_heatmap(df_per, OUT_HEATMAP_PER_NARROW, "PER")
            plot_heatmap(df_pfer, OUT_HEATMAP_PFER_NARROW, "PFER")
            print(f"PER heatmap saved to: {OUT_HEATMAP_PER_NARROW}")
            print(f"PFER heatmap saved to: {OUT_HEATMAP_PFER_NARROW}")

        # ---------- LaTeX table builder ----------
        def latex_tabular_single_metric(df_input: pd.DataFrame,
                                        metric_label: str,
                                        caption: str = None,
                                        label: str = None) -> str:
            models = list(df_input.columns)
            align = "l" + "c" * len(models)
            header = " & ".join(["\\textbf{Language}"] + [str(m) for m in models]) + " \\\\"

            def fmt(x):
                if pd.isna(x):
                    return "--"
                return f"{x:.3f}"

            body_lines = []
            for lang, row in df_input.iterrows():
                cells = [str(lang)] + [fmt(row[m]) for m in models]
                body_lines.append(" & ".join(cells) + " \\\\")

            # Average row
            avg_row = df_input.mean(axis=0, skipna=True)
            avg_cells = ["\\textbf{Average}"] + [fmt(avg_row[m]) for m in models]
            avg_line = " & ".join(avg_cells) + " \\\\"

            tabular_lines = []
            tabular_lines.append("\\begin{tabular}{" + align + "}")
            tabular_lines.append("\\toprule")
            tabular_lines.append(header)
            tabular_lines.append("\\midrule")
            tabular_lines.extend(body_lines)
            tabular_lines.append("\\midrule")
            tabular_lines.append(avg_line)
            tabular_lines.append("\\bottomrule")
            tabular_lines.append("\\end{tabular}")
            tabular_block = "\n".join(tabular_lines)

            # Defaults for caption/label if not provided
            if caption is None:
                caption = f"{metric_label} results by language and model."
            if label is None:
                label = f"tab:{metric_label.lower()}-results"

            table_block = (
                "\\begin{table}[t]\n"
                "\\centering\n"
                "\\resizebox{\\textwidth}{!}{%\n"
                f"{tabular_block}\n"
                "}\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{{label}}}\n"
                "\\end{table}\n"
            )
            return table_block

        latex_per = latex_tabular_single_metric(
            df_per, "PER",
            caption=f"PER ({label_name.title()}) results by language and model.",
            label=f"tab:{label_name}-per"
        )
        latex_pfer = latex_tabular_single_metric(
            df_pfer, "PFER",
            caption=f"PFER ({label_name.title()}) results by language and model.",
            label=f"tab:{label_name}-pfer"
        )

        if label_name == "broad":
            with open(OUT_LATEX_PER_BROAD, "w", encoding="utf-8") as f:
                f.write(latex_per)
            with open(OUT_LATEX_PFER_BROAD, "w", encoding="utf-8") as f:
                f.write(latex_pfer)
            print(f"\nLaTeX PER table saved to: {OUT_LATEX_PER_BROAD}")
            print(f"LaTeX PFER table saved to: {OUT_LATEX_PFER_BROAD}\n")
        else:
            with open(OUT_LATEX_PER_NARROW, "w", encoding="utf-8") as f:
                f.write(latex_per)
            with open(OUT_LATEX_PFER_NARROW, "w", encoding="utf-8") as f:
                f.write(latex_pfer)
            print(f"\nLaTeX PER table saved to: {OUT_LATEX_PER_NARROW}")
            print(f"LaTeX PFER table saved to: {OUT_LATEX_PFER_NARROW}\n")

        return df_per, df_pfer


    # ---------- Run for BROAD then NARROW ----------
    df_per_broad, df_pfer_broad = evaluate_labelset(broad_truth, "broad", eligible_langs_broad, lang_order_broad)
    df_per_narrow, df_pfer_narrow = evaluate_labelset(narrow_truth, "narrow", eligible_langs_narrow, lang_order_narrow)


    def plot_heatmap_pair(
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        out_path: str,
        metric: str,
        title_left: str = "BROAD",
        title_right: str = "NARROW",
    ):
        """
        Draw two heatmaps side-by-side with a single, bottom colorbar.
        - Rows (models) are aligned via union of model names.
        - Columns (languages) are independent per subplot (no forced union).
        """

        # Transpose to H = models x languages (rows=models, cols=languages)
        H_left = df_left.T
        H_right = df_right.T

        # --- Align rows (models) but keep independent columns (languages) ---
        models_order = list(H_left.index)
        for m in H_right.index:
            if m not in models_order:
                models_order.append(m)

        H_left = H_left.reindex(index=models_order)
        H_right = H_right.reindex(index=models_order)

        # Panel-specific language orders
        langs_left = list(H_left.columns)
        langs_right = list(H_right.columns)

        # Figure sizing that respects both panels
        h = len(models_order)
        w_left = max(1, len(langs_left))
        w_right = max(1, len(langs_right))
        # width scales with sum of both language counts
        fig_w = max(10, min(24, 0.7 * (w_left + w_right) + 4.0))
        fig_h = max(5.5, min(20, 0.55 * h + 2.5))

        # Shared scale
        metric_upper = metric.upper()
        if metric_upper == "PER":
            vmin, vmax = 0, max_per
            cbar_label = "PER (%)"
        else:
            vmin, vmax = 0, max_pfer
            cbar_label = "PFER (%)"

        # --- Layout with a dedicated colorbar row ---
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            nrows=2, ncols=2,
            height_ratios=[1, 0.06],  # bottom row for colorbar
            width_ratios=[w_left, w_right],
            wspace=0.12, hspace=0.35
        )

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[1, :])  # spans both columns

        # Plot left heatmap
        im_left = ax_left.imshow(
            H_left.values,
            aspect="auto",
            cmap="viridis_r",
            vmin=vmin, vmax=vmax
        )
        ax_left.set_title(title_left)
        ax_left.set_yticks(np.arange(h))
        ax_left.set_yticklabels(models_order)
        ax_left.set_xticks(np.arange(w_left))
        ax_left.set_xticklabels(langs_left, rotation=90)

        # Plot right heatmap
        im_right = ax_right.imshow(
            H_right.values,
            aspect="auto",
            cmap="viridis_r",
            vmin=vmin, vmax=vmax
        )
        ax_right.set_title(title_right)
        # share Y tick labels visually by hiding on the right
        ax_right.set_yticks(np.arange(h))
        ax_right.set_yticklabels([])
        ax_right.set_xticks(np.arange(w_right))
        ax_right.set_xticklabels(langs_right, rotation=90)

        # Single horizontal colorbar, anchored to bottom row
        cbar = fig.colorbar(im_right, cax=cbar_ax, orientation="horizontal")
        #cbar.ax.set_xlabel(cbar_label)

        #fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)




    plot_heatmap_pair(
        df_left=df_per_broad,
        df_right=df_per_narrow,
        out_path=OUT_HEATMAP_PER_COMBINED,
        metric="PER",
        title_left="BROAD",
        title_right="NARROW",
    )
    print(f"Combined PER heatmap saved to: {OUT_HEATMAP_PER_COMBINED}")


    # PFER
    plot_heatmap_pair(
        df_left=df_pfer_broad,
        df_right=df_pfer_narrow,
        out_path=OUT_HEATMAP_PFER_COMBINED,
        metric="PFER",
        title_left="BROAD",
        title_right="NARROW",
    )
    print(f"Combined PFER heatmap saved to: {OUT_HEATMAP_PFER_COMBINED}")



if __name__ == "__main__":
    main()
