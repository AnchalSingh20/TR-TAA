"""
Significance tests and seed stability for TR-TAA vs. TAA (ViT-B/14).

mAP and mIoU are set-level metrics, so a per-image paired t-test on them
is not valid. Instead:

  * Segmentation   -> paired Wilcoxon signed-rank test over per-image IoU
                      (TAA vs. TR-TAA under the same transformation).
  * Classification -> paired bootstrap over images: resample image indices
                      with replacement, recompute set-level mAP for both
                      attacks on the same resample, and form the
                      distribution of the mAP difference (95% CI and
                      one-sided p-value).
  * Across seeds   -> mean +/- std of each metric over the seed runs.

Two stages:
  collect  (GPU): evaluates the cached clean/adversarial images with the
           transformation pipeline of evaluate_eot_taa_multilabel.py and
           stores per-image scores (classification) and IoU (segmentation)
           in a .npz file. Run once per seed.
  analyze  (CPU): reads the .npz files and writes the significance tests,
           the mean +/- std table and a LaTeX table.

USAGE
    # TR-TAA images per seed: generate_eot_taa_clean.py --model_name dinov2_vitb14 --K 8 --seed <s>
    for s in 0 1 2; do
      python scripts/statistical_analysis.py collect --pascalvoc_root <VOC2012> \
          --model_name dinov2_vitb14 \
          --new_cache_dir ./outputs/cached_images_clean_seed$s \
          --seed $s --out ./outputs/results/stats_perimage_vitb14_seed$s.npz
    done
    python scripts/statistical_analysis.py analyze \
        --npz ./outputs/results/stats_perimage_vitb14_seed{0,1,2}.npz \
        --latex_out ./outputs/results/table_stats_vitb14.tex \
        --json_out ./outputs/results/table_stats_vitb14.json
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

# scipy/sklearn are needed by the *analyze* stage (no torch required there).
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: torch and the model/src imports are done lazily inside collect_main()
# so that the analyze stage runs on any machine without a GPU or torch.


# ==================================================================
# ---------------------  PURE-STATS CORE  --------------------------
#  (no torch; unit-testable on any host)
# ==================================================================

ATTACKS = ["TAA", "EoT-TAA K=4", "EoT-TAA K=8"]     # full default (both K)
COMPARISONS = [("EoT-TAA K=4", "TAA"), ("EoT-TAA K=8", "TAA")]

ATTACKS_K8_ONLY = ["TAA", "EoT-TAA K=8"]            # runs without a K=4 cache
COMPARISONS_K8_ONLY = [("EoT-TAA K=8", "TAA")]


def compute_map(scores, labels):
    """
    mean Average Precision over classes, in percent.

    Vectorized step-wise AP, numerically identical to
    sklearn.metrics.average_precision_score (verified to <1e-8 including
    heavy-tie regimes), but computed for all classes in one pass. The
    bootstrap calls this thousands of times, so the sklearn per-class
    Python loop was the dominant cost.
    """
    S = np.asarray(scores, dtype=np.float64)
    L = np.asarray(labels, dtype=np.float64)
    N, C = S.shape
    order = np.argsort(-S, axis=0, kind="mergesort")
    Ls = np.take_along_axis(L, order, axis=0)
    Ss = np.take_along_axis(S, order, axis=0)
    tp = np.cumsum(Ls, axis=0)
    prec = tp / np.arange(1, N + 1)[:, None]
    npos = L.sum(0)
    # tie handling: keep only the last index of each equal-score group
    keep = np.ones_like(Ss, dtype=bool)
    keep[:-1] = Ss[:-1] != Ss[1:]
    aps = []
    for c in range(C):
        if npos[c] == 0:
            continue
        k = keep[:, c]
        r = tp[k, c] / npos[c]
        pr = prec[k, c]
        r_prev = np.concatenate(([0.0], r[:-1]))
        aps.append(float(np.sum((r - r_prev) * pr)))
    return float(np.mean(aps)) * 100.0 if aps else 0.0


def paired_bootstrap_map_diff(scores_a, scores_b, labels, n_boot=2000,
                              rng=None, alternative="less"):
    """
    Paired bootstrap for the difference in set-mAP between attack A and
    attack B over the SAME resampled image set.

    We test whether attack A drives mAP LOWER than attack B (a lower mAP
    means a stronger attack). For TR-TAA vs. TAA, A = TR-TAA and
    B = TAA, and we expect diff = mAP(A) - mAP(B) < 0.

    Returns dict with observed diff, 95% CI, and one-sided bootstrap p.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = scores_a.shape[0]

    obs = compute_map(scores_a, labels) - compute_map(scores_b, labels)

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)          # resample with replacement
        la = labels[idx]
        diffs[b] = (compute_map(scores_a[idx], la)
                    - compute_map(scores_b[idx], la))

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # One-sided bootstrap p-value: how often the bootstrap diff contradicts
    # the alternative. For "A lower than B" (alternative='less'), the null
    # is diff >= 0, so p = P(bootstrap diff >= 0).
    if alternative == "less":
        p = float(np.mean(diffs >= 0.0))
    elif alternative == "greater":
        p = float(np.mean(diffs <= 0.0))
    else:  # two-sided
        p = 2.0 * min(np.mean(diffs >= 0.0), np.mean(diffs <= 0.0))
    p = min(max(p, 1.0 / n_boot), 1.0)            # clamp; never report p=0
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p": p, "n_boot": n_boot, "n_images": int(N)}


def wilcoxon_paired_iou(iou_a, iou_b, alternative="less"):
    """
    Paired Wilcoxon signed-rank test on per-image IoU.

    A stronger attack yields LOWER per-image IoU, so for A = TR-TAA and
    B = TAA we test alternative='less' (IoU_A < IoU_B). Reports the median
    per-image IoU difference and the rank-biserial effect size.

    Underflow handling: with tens of thousands of pairs the true p-value can
    fall below the smallest representable double (~1e-308), and scipy then
    returns exactly 0.0. Reporting "p = 0" would be misleading, so
    we also compute the normal-approximation z statistic and log10(p) and
    return those. Report as "p < 10^-300" (or quote z) when p underflows.
    """
    from scipy.stats import norm

    iou_a = np.asarray(iou_a, dtype=np.float64)
    iou_b = np.asarray(iou_b, dtype=np.float64)
    d = iou_a - iou_b
    nonzero = np.count_nonzero(d)
    median_diff = float(np.median(d))

    if nonzero == 0:
        return {"stat": float("nan"), "p": 1.0, "median_diff": median_diff,
                "rank_biserial": 0.0, "n_images": int(len(d)),
                "n_nonzero": 0, "z": 0.0, "log10_p": 0.0}

    stat, p = wilcoxon(iou_a, iou_b, alternative=alternative,
                       zero_method="wilcox")

    # Rank-biserial correlation as an effect size in [-1, 1].
    ranks = _rankdata(np.abs(d[d != 0]))
    signs = np.sign(d[d != 0])
    w_plus = ranks[signs > 0].sum()
    w_minus = ranks[signs < 0].sum()
    total = w_plus + w_minus
    rank_biserial = float((w_plus - w_minus) / total) if total > 0 else 0.0

    # Normal approximation -> z and log10(p), valid when p underflows to 0.
    n = nonzero
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    # w_plus is the sum of ranks of positive differences.
    z = float((w_plus - mu) / sigma) if sigma > 0 else 0.0
    if alternative == "less":          # evidence that A < B  =>  z very negative
        log10_p = float(norm.logsf(-z) / np.log(10.0))
    elif alternative == "greater":
        log10_p = float(norm.logsf(z) / np.log(10.0))
    else:
        log10_p = float((norm.logsf(abs(z)) + np.log(2.0)) / np.log(10.0))

    return {"stat": float(stat), "p": float(p), "median_diff": median_diff,
            "rank_biserial": rank_biserial, "n_images": int(len(d)),
            "n_nonzero": int(nonzero), "z": z, "log10_p": log10_p}


def fmt_p(p, log10_p=None):
    """Format a p-value, never printing 'p = 0'."""
    if p is not None and p > 0:
        return f"{p:.2e}"
    if log10_p is not None and np.isfinite(log10_p):
        return f"1e{log10_p:.0f} (normal approx.)"
    return "<1e-300"


def _rankdata(a):
    """Average ranks, ties averaged (small dependency-free helper)."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction. Input: list of (key, p).
    Returns dict key -> (p_adjusted, reject@0.05)."""
    items = sorted(pvals, key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        adj = min((m - i) * p, 1.0)
        running = max(running, adj)          # enforce monotonicity
        out[key] = (running, running < 0.05)
    return out


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def aggregate_seeds(per_seed_metrics):
    """
    per_seed_metrics: list (one per seed) of dict
        {attack: {transform: {"cls_map":.., "seg_miou":..}}}
    Returns {attack: {transform: {"cls_map_mean/std", "seg_miou_mean/std"}}}.
    """
    attacks = list(per_seed_metrics[0].keys())
    transforms = list(per_seed_metrics[0][attacks[0]].keys())
    agg = {}
    for a in attacks:
        agg[a] = {}
        for t in transforms:
            cls = np.array([s[a][t]["cls_map"] for s in per_seed_metrics])
            seg = np.array([s[a][t]["seg_miou"] for s in per_seed_metrics])
            agg[a][t] = {
                "cls_map_mean": float(cls.mean()), "cls_map_std": float(cls.std(ddof=1) if len(cls) > 1 else 0.0),
                "seg_miou_mean": float(seg.mean()), "seg_miou_std": float(seg.std(ddof=1) if len(seg) > 1 else 0.0),
                "n_seeds": int(len(cls)),
            }
    return agg


# ==================================================================
# ---------------------  ANALYZE STAGE  ----------------------------
# ==================================================================

def _npz_key(kind, attack, transform):
    safe = attack.replace(" ", "_").replace("=", "")
    saft = transform.replace(" ", "_").replace("%", "pct").replace("=", "")
    return f"{kind}__{safe}__{saft}"


def load_npz_runs(npz_paths):
    """Load per-seed .npz artifacts produced by the collect stage.
    Returns (runs, transforms, meta) where runs is a list of dicts."""
    runs = []
    transforms = None
    meta = None
    for path in npz_paths:
        z = np.load(path, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        # Attacks actually present in this artifact (older artifacts without
        # the field predate multi-K support and only ever contained TAA+K8).
        attacks = m.get("attacks", ATTACKS_K8_ONLY)
        if meta is None:
            meta = m
            transforms = m["transforms"]
        run = {"seed": m["seed"], "cls_labels": z["cls_labels"],
               "attacks": attacks, "cls_scores": {}, "seg_iou": {}}
        for a in attacks:
            run["cls_scores"][a] = {}
            run["seg_iou"][a] = {}
            for t in transforms:
                run["cls_scores"][a][t] = z[_npz_key("clsscore", a, t)]
                run["seg_iou"][a][t] = z[_npz_key("segiou", a, t)]
        runs.append(run)
    return runs, transforms, meta


def per_seed_setlevel(run, transforms):
    """Compute set-level mAP and mIoU per (attack, transform) for one seed."""
    out = {}
    for a in run.get("attacks", ATTACKS):
        out[a] = {}
        for t in transforms:
            cls_map = compute_map(run["cls_scores"][a][t], run["cls_labels"])
            seg_miou = float(np.mean(run["seg_iou"][a][t])) * 100.0
            out[a][t] = {"cls_map": cls_map, "seg_miou": seg_miou}
    return out


def analyze_main(args):
    npz_paths = []
    for pat in args.npz:
        npz_paths.extend(sorted(glob.glob(pat)))
    if not npz_paths:
        raise SystemExit(f"No .npz files matched: {args.npz}")
    print(f"Loaded {len(npz_paths)} per-seed artifact(s):")
    for p in npz_paths:
        print(f"  {p}")

    runs, transforms, meta = load_npz_runs(npz_paths)
    model = meta.get("model_name", "?")
    rng = np.random.default_rng(args.boot_seed)

    # Attacks present across the loaded runs (intersection, TAA-first order).
    present = set(runs[0]["attacks"])
    for r in runs[1:]:
        present &= set(r["attacks"])
    attacks = [a for a in ATTACKS if a in present] + \
              [a for a in present if a not in ATTACKS]
    eot_attacks = [a for a in attacks if a != "TAA"]
    comparisons = [(e, "TAA") for e in eot_attacks]
    if "TAA" not in attacks:
        raise SystemExit("Artifacts contain no TAA baseline to compare against.")

    # ---- (1) Multi-seed mean +/- std of set-level metrics ----
    per_seed = [per_seed_setlevel(r, transforms) for r in runs]
    agg = aggregate_seeds(per_seed)

    print("\n" + "=" * 78)
    print(f"SET-LEVEL METRICS  (mean +/- std over {len(runs)} seed(s))  model={model}")
    print(f"Attacks present: {', '.join(attacks)}")
    print("=" * 78)

    def cell(a, t, k):
        return f"{agg[a][t][k+'_mean']:.1f}±{agg[a][t][k+'_std']:.1f}"

    cls_cols = "  ".join(f"{a+' mAP':>16s}" for a in attacks)
    seg_cols = "  ".join(f"{a+' mIoU':>16s}" for a in attacks)
    hdr = f"{'Transform':16s} | {cls_cols} | {seg_cols}"
    print(hdr)
    print("-" * len(hdr))
    for t in transforms:
        cls_vals = "  ".join(f"{cell(a, t, 'cls_map'):>16s}" for a in attacks)
        seg_vals = "  ".join(f"{cell(a, t, 'seg_miou'):>16s}" for a in attacks)
        print(f"{t:16s} | {cls_vals} | {seg_vals}")

    # ---- (2) Paired significance tests (computed per seed, then pooled) ----
    #   Segmentation: Wilcoxon signed-rank on per-image IoU (adversarial).
    #   Classification: paired bootstrap on set-mAP difference.
    #   Lower adv-metric = stronger attack, so alternative = 'less'.
    print("\n" + "=" * 78)
    print("PAIRED SIGNIFICANCE  (EoT-TAA vs TAA, adversarial metric under transform)")
    print("Segmentation: Wilcoxon signed-rank on per-image IoU | "
          "Classification: paired bootstrap on mAP")
    print("Alternative: EoT drives the metric LOWER than TAA (stronger attack)")
    print("=" * 78)

    # We report the transform-averaged test on the first seed for the headline,
    # and also give per-transform results. Pooling across seeds is done by
    # concatenating per-image IoU (seg) and averaging bootstrap diffs (cls).
    sig_rows = []
    pforcorrection = []

    exclude_clean = "No transform"
    tf_under = [t for t in transforms if t != exclude_clean]

    for (a_name, b_name) in comparisons:
        # ----- Segmentation: pool per-image IoU across seeds and transforms -----
        seg_a = np.concatenate([np.concatenate([r["seg_iou"][a_name][t] for t in tf_under])
                                for r in runs])
        seg_b = np.concatenate([np.concatenate([r["seg_iou"][b_name][t] for t in tf_under])
                                for r in runs])
        seg_res = wilcoxon_paired_iou(seg_a, seg_b, alternative="less")

        # ----- Classification: bootstrap per seed & transform, average diffs -----
        cls_diffs, cls_ps, cls_los, cls_his = [], [], [], []
        if not getattr(args, "pooled_cls", False):
            # The pooled classification bootstrap runs 9 transforms x 3 seeds
            # of 2000 resamples each (~70 min) to produce a Fisher-combined
            # p-value whose pairs are NOT independent (see the per-transform
            # tests below). It is therefore skipped by default and the
            # per-transform bootstrap is reported instead. Pass
            # --pooled_cls to compute it anyway.
            cls_mean_diff = float("nan"); cls_p_fisher = float("nan")
        else:
          for r in runs:
            for t in tf_under:
                res = paired_bootstrap_map_diff(
                    r["cls_scores"][a_name][t], r["cls_scores"][b_name][t],
                    r["cls_labels"], n_boot=args.n_boot, rng=rng,
                    alternative="less")
                cls_diffs.append(res["diff"]); cls_ps.append(res["p"])
                cls_los.append(res["ci_low"]); cls_his.append(res["ci_high"])
          cls_mean_diff = float(np.mean(cls_diffs))
          # Combine per-condition bootstrap p-values conservatively (Fisher).
          cls_p_fisher = _fisher_combine(cls_ps)

        sig_rows.append({
            "comparison": f"{a_name} vs {b_name}",
            "seg_median_iou_diff_pp": seg_res["median_diff"] * 100.0,
            "seg_rank_biserial": seg_res["rank_biserial"],
            "seg_p": seg_res["p"], "seg_n": seg_res["n_images"],
            "seg_log10_p": seg_res.get("log10_p"), "seg_z": seg_res.get("z"),
            "cls_mean_map_diff_pp": cls_mean_diff,
            "cls_p": cls_p_fisher,
        })
        pforcorrection.append((f"{a_name} vs {b_name} [seg]", seg_res["p"]))
        if not np.isnan(cls_p_fisher):
            pforcorrection.append((f"{a_name} vs {b_name} [cls]", cls_p_fisher))

    corrected = holm_bonferroni(pforcorrection)

    # ---- (2b) PER-TRANSFORM tests: the statistically clean version ----
    # The pooled test above concatenates the same image across 9 transforms
    # and 3 seeds, so its "n" counts each image 27 times and the pairs are
    # NOT independent (Wilcoxon assumes independence). The per-transform
    # test below uses each image once per transform -> genuinely independent
    # pairs, n = number of seg images. This is the test reported in the
    # paper; the pooled number is only a convenience summary.
    print("\n" + "=" * 78)
    print("PER-TRANSFORM PAIRED TESTS  (independent pairs -- report these)")
    print("Seg: Wilcoxon on per-image IoU | Cls: paired image-level bootstrap on mAP")
    print("Both averaged over seeds so each image contributes exactly one pair")
    print("=" * 78)
    per_tf_rows = []
    for (a_name, b_name) in comparisons:
        tf_p = []
        for t in tf_under:
            # --- segmentation: average each image's IoU across seeds ---
            ia = np.mean([r["seg_iou"][a_name][t] for r in runs], axis=0)
            ib = np.mean([r["seg_iou"][b_name][t] for r in runs], axis=0)
            res = wilcoxon_paired_iou(ia, ib, alternative="less")

            # --- classification: average each image's scores across seeds,
            #     then bootstrap over images (independent pairs) ---
            sa_ = np.mean([r["cls_scores"][a_name][t] for r in runs], axis=0)
            sb_ = np.mean([r["cls_scores"][b_name][t] for r in runs], axis=0)
            bres = paired_bootstrap_map_diff(sa_, sb_, runs[0]["cls_labels"],
                                             n_boot=args.n_boot, rng=rng,
                                             alternative="less")

            per_tf_rows.append({"comparison": f"{a_name} vs {b_name}",
                                "transform": t,
                                "median_iou_diff_pp": res["median_diff"] * 100.0,
                                "rank_biserial": res["rank_biserial"],
                                "p": res["p"], "log10_p": res["log10_p"],
                                "z": res["z"], "n": res["n_images"],
                                "cls_map_diff_pp": bres["diff"],
                                "cls_ci_low": bres["ci_low"],
                                "cls_ci_high": bres["ci_high"],
                                "cls_p": bres["p"],
                                "cls_n": bres["n_images"]})
            tf_p.append((t, res["p"] if res["p"] > 0 else 1e-300))
        tf_corr = holm_bonferroni(tf_p)
        cls_corr = holm_bonferroni(
            [(r["transform"], r["cls_p"]) for r in per_tf_rows
             if r["comparison"] == f"{a_name} vs {b_name}"])

        print(f"\n{a_name} vs {b_name}")
        print(f"  {'Transform':16s} | {'dIoU(pp)':>9s} {'r':>7s} {'Holm p':>11s}"
              f" | {'dmAP(pp)':>9s} {'95% CI':>18s} {'Holm p':>9s}")
        print("  " + "-" * 88)
        for row in [r for r in per_tf_rows if r["comparison"] == f"{a_name} vs {b_name}"]:
            padj = tf_corr[row["transform"]][0]
            cadj = cls_corr[row["transform"]][0]
            ci = f"[{row['cls_ci_low']:+.1f},{row['cls_ci_high']:+.1f}]"
            print(f"  {row['transform']:16s} | {row['median_iou_diff_pp']:+9.2f} "
                  f"{row['rank_biserial']:+7.3f} {padj:11.2e}"
                  f" | {row['cls_map_diff_pp']:+9.2f} {ci:>18s} {cadj:9.2e}")
        print(f"  (seg n={per_tf_rows[-1]['n']:,} images/transform; "
              f"cls n={per_tf_rows[-1]['cls_n']:,} images/transform)")

    for row in sig_rows:
        cmp = row["comparison"]
        seg_p_adj = corrected[f"{cmp} [seg]"][0]
        cls_p_adj = corrected.get(f"{cmp} [cls]", (float("nan"), False))[0]
        print(f"\n{cmp}   (averaged over {len(tf_under)} transforms x {len(runs)} seed(s))")
        print(f"  Segmentation  median per-image IoU diff = {row['seg_median_iou_diff_pp']:+.2f} pp "
              f"(negative = EoT stronger)")
        print(f"                rank-biserial r = {row['seg_rank_biserial']:+.3f}  "
              f"| Wilcoxon p = {fmt_p(row['seg_p'], row.get('seg_log10_p'))}  (Holm-adj {seg_p_adj:.2e}) {stars(seg_p_adj)}  "
              f"| n = {row['seg_n']:,} images")
        if np.isnan(row["cls_mean_map_diff_pp"]):
            print("  Classification pooled bootstrap: skipped "
                  "(pairs not independent; see per-transform table above. "
                  "Use --pooled_cls to force it.)")
        else:
            print(f"  Classification mean set-mAP diff = {row['cls_mean_map_diff_pp']:+.2f} pp "
                  f"(negative = EoT stronger)")
            print(f"                bootstrap p (Fisher-combined) = {row['cls_p']:.2e}  "
                  f"(Holm-adj {cls_p_adj:.2e}) {stars(cls_p_adj)}")

    # ---- (3) LaTeX ----
    if args.latex_out:
        _write_latex(args.latex_out, model, transforms, agg, attacks,
                     len(runs))
        print(f"\nLaTeX table written to: {args.latex_out}")

    # ---- machine-readable dump ----
    if args.json_out:
        dump = {"model": model, "n_seeds": len(runs),
                "aggregate": agg, "significance": sig_rows,
                "per_transform_significance": per_tf_rows}
        with open(args.json_out, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"JSON summary written to: {args.json_out}")


def _fisher_combine(pvals):
    """Fisher's method to combine independent one-sided p-values."""
    from scipy.stats import chi2
    pvals = np.clip(np.asarray(pvals, dtype=np.float64), 1e-12, 1.0)
    stat = -2.0 * np.sum(np.log(pvals))
    return float(chi2.sf(stat, df=2 * len(pvals)))


def _latex_name(a):
    return a.replace("EoT-TAA ", "").replace("K=", "K=")  # -> "TAA", "K=8"


def _write_latex(path, model, transforms, agg, attacks, n_seeds):
    n = len(attacks)
    colspec = "l" + "c" * n + "|" + "c" * n
    head = " & ".join(_latex_name(a) for a in attacks)
    lines = []
    lines.append("% Auto-generated by scripts/statistical_analysis.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{Adversarial classification mAP and segmentation "
                 f"mIoU (\\%) under transforms for {model}, "
                 f"mean\\,$\\pm$\\,std over {n_seeds} seed(s). Lower is a stronger "
                 f"attack.}}")
    lines.append("\\label{tab:stats}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(f" & \\multicolumn{{{n}}}{{c|}}{{Classification mAP$\\downarrow$}} "
                 f"& \\multicolumn{{{n}}}{{c}}{{Segmentation mIoU$\\downarrow$}} \\\\")
    lines.append(f"Transform & {head} & {head} \\\\")
    lines.append("\\midrule")
    for t in transforms:
        def c(a, k):
            return f"{agg[a][t][k+'_mean']:.1f}$\\pm${agg[a][t][k+'_std']:.1f}"
        cls = " & ".join(c(a, "cls_map") for a in attacks)
        seg = " & ".join(c(a, "seg_miou") for a in attacks)
        t_escaped = t.replace("%", "\\%")     # precomputed OUTSIDE the f-string:
                                              # a backslash inside {...} is a
                                              # SyntaxError on Python < 3.12
        lines.append(f"{t_escaped} & {cls} & {seg} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ==================================================================
# ---------------------  COLLECT STAGE  ----------------------------
#  (needs torch + GPU + the image caches; mirrors the exact
#   transform pipeline of evaluate_eot_taa_multilabel.py)
# ==================================================================

def collect_main(args):
    import io
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image, ImageFilter

    from src.reproducibility import set_seed
    from src.models import get_dinov2_model, get_model_info, normalize_for_model
    from src.heads import SegmentationHead
    from src.heads_multilabel import MultiLabelClassificationHead
    from src.datasets import PascalVOCClassification
    from src.datasets_multilabel import PascalVOCClassificationMultiLabel

    set_seed(args.seed)
    device = args.device
    model_info = get_model_info(args.model_name)

    # ---- transform pipeline: copied verbatim so numbers match main table ----
    def apply_jpeg(images, quality=50):
        out = []
        for i in range(images.shape[0]):
            im = TF.to_pil_image(images[i].cpu().clamp(0, 1))
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality); buf.seek(0)
            out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
        return torch.stack(out).to(images.device)

    def apply_blur(images, sigma=2.0):
        out = []
        for i in range(images.shape[0]):
            im = TF.to_pil_image(images[i].cpu().clamp(0, 1))
            out.append(TF.to_tensor(im.filter(ImageFilter.GaussianBlur(radius=sigma))))
        return torch.stack(out).to(images.device)

    def apply_resize(images, scale=0.5):
        B, C, H, W = images.shape
        small = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)

    def apply_rotation(images, angle=15):
        out = []
        for i in range(images.shape[0]):
            im = TF.to_pil_image(images[i].cpu().clamp(0, 1))
            out.append(TF.to_tensor(TF.rotate(im, angle)))
        return torch.stack(out).to(images.device)

    def apply_noise(images, sigma=0.03):
        return (images + torch.randn_like(images) * sigma).clamp(0, 1)

    def apply_pipeline(images):
        images = apply_resize(images, 0.5); images = apply_blur(images, 1.5)
        images = apply_noise(images, 0.02); images = apply_jpeg(images, 60)
        return images

    transforms_dict = {
        "No transform":   (lambda x: x, {}),
        "JPEG q=50":      (apply_jpeg,    {"quality": 50}),
        "JPEG q=30":      (apply_jpeg,    {"quality": 30}),
        "Blur s=1.0":     (apply_blur,    {"sigma": 1.0}),
        "Blur s=2.0":     (apply_blur,    {"sigma": 2.0}),
        "Resize 50%":     (apply_resize,  {"scale": 0.5}),
        "Resize 25%":     (apply_resize,  {"scale": 0.25}),
        "Rotation 15deg": (apply_rotation, {"angle": 15}),
        "Noise s=0.03":   (apply_noise,   {"sigma": 0.03}),
        "Print-scan":     (apply_pipeline, {}),
    }
    transforms = list(transforms_dict.keys())

    # ---- model + heads ----
    backbone = get_dinov2_model(args.model_name, device)
    cls_head = MultiLabelClassificationHead(model_info["dim"], 20).to(device)
    cls_head.load_state_dict(torch.load(os.path.join(
        args.heads_dir, f"{args.model_name}_classification_head_multilabel.pth"),
        map_location=device, weights_only=False)); cls_head.eval()
    seg_head = SegmentationHead(model_info["dim"], 21, model_info["patch_size"],
                                args.image_size).to(device)
    seg_head.load_state_dict(torch.load(os.path.join(
        args.heads_dir, f"{args.model_name}_segmentation_head.pth"),
        map_location=device, weights_only=False)); seg_head.eval()

    # ---- caches ----
    old_cache = os.path.join(args.old_cache_dir, args.model_name)
    new_cache = os.path.join(args.new_cache_dir, args.model_name)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)

    def have(*names):
        return all(os.path.exists(os.path.join(new_cache, n)) for n in names)

    cache = {
        "TAA": {"cls_adv": load(os.path.join(old_cache, "cls_adv_images.pt")),
                "seg_adv": load(os.path.join(old_cache, "seg_adv_images.pt")),
                "seg_masks": load(os.path.join(old_cache, "seg_masks.pt"))},
    }
    # EoT variants are optional: include whichever K-caches actually exist so a
    # K=8-only seed (e.g. when saving compute) works without editing anything.
    if have("cls_adv_images_K8.pt", "seg_adv_images_K8.pt"):
        cache["EoT-TAA K=8"] = {
            "cls_adv": load(os.path.join(new_cache, "cls_adv_images_K8.pt")),
            "seg_adv": load(os.path.join(new_cache, "seg_adv_images_K8.pt")),
            "seg_masks": load(os.path.join(new_cache, "seg_masks.pt"))}
    if have("cls_adv_images_K4.pt", "seg_adv_images_K4.pt"):
        cache["EoT-TAA K=4"] = {
            "cls_adv": load(os.path.join(new_cache, "cls_adv_images_K4.pt")),
            "seg_adv": load(os.path.join(new_cache, "seg_adv_images_K4.pt")),
            "seg_masks": load(os.path.join(new_cache, "seg_masks.pt"))}

    attacks_present = [a for a in ATTACKS if a in cache]
    if len(attacks_present) < 2:
        raise SystemExit(
            f"Need TAA plus at least one EoT variant in {new_cache}; "
            f"found only: {attacks_present}")
    print(f"[seed {args.seed}] attacks present: {', '.join(attacks_present)}")

    # ---- aligned multi-label GT ----
    sl = PascalVOCClassification(root=args.pascalvoc_root, split="val",
                                 image_size=args.image_size, transform=None)
    ml = PascalVOCClassificationMultiLabel(root=args.pascalvoc_root, split="val",
                                           image_size=args.image_size, transform=None)
    assert len(sl) == len(ml), "single/multi-label datasets differ in length"
    cls_labels = torch.from_numpy(np.stack(ml.labels, 0).astype(np.float32))
    N_cls = cache["TAA"]["cls_adv"].shape[0]
    cls_labels = cls_labels[:N_cls]

    def per_image_cls_scores(imgs):
        outs = []
        for s in range(0, imgs.shape[0], args.eval_batch_size):
            e = min(s + args.eval_batch_size, imgs.shape[0])
            with torch.no_grad():
                x = normalize_for_model(imgs[s:e].to(device))
                feats = backbone.forward_features(x)
                logits = cls_head(feats["x_norm_clstoken"])
                outs.append(torch.sigmoid(logits).cpu().numpy())
            torch.cuda.empty_cache()
        return np.concatenate(outs, 0)

    def per_image_seg_iou(imgs, masks, num_classes=21):
        ious = []
        for s in range(0, imgs.shape[0], args.eval_batch_size):
            e = min(s + args.eval_batch_size, imgs.shape[0])
            with torch.no_grad():
                x = normalize_for_model(imgs[s:e].to(device))
                feats = backbone.forward_features(x)
                logits = seg_head(feats["x_norm_clstoken"], feats["x_norm_patchtokens"])
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks[s:e].shape[-2:],
                                           mode="bilinear", align_corners=False)
                preds = logits.argmax(1).cpu()
            m = masks[s:e]
            for b in range(preds.shape[0]):
                valid = m[b] != -1
                p = preds[b][valid].view(-1)
                g = m[b][valid].view(-1)
                present = torch.unique(torch.cat([p, g]))
                per_cls = []
                for cidx in present.tolist():
                    inter = ((p == cidx) & (g == cidx)).sum().item()
                    union = ((p == cidx) | (g == cidx)).sum().item()
                    if union > 0:
                        per_cls.append(inter / union)
                ious.append(float(np.mean(per_cls)) if per_cls else 0.0)
            torch.cuda.empty_cache()
        return np.asarray(ious, dtype=np.float32)

    save_arrays = {"cls_labels": cls_labels.numpy()}
    for a in attacks_present:
        print(f"[seed {args.seed}] collecting {a} ...")
        for t in transforms:
            tf_func, tf_kw = transforms_dict[t]
            cls_imgs = tf_func(cache[a]["cls_adv"].clone(), **tf_kw) \
                if tf_kw or t != "No transform" else cache[a]["cls_adv"]
            # apply transform batchwise to bound memory
            scores = _batched_transform_then(cls_imgs if False else cache[a]["cls_adv"],
                                             tf_func, tf_kw, per_image_cls_scores,
                                             args.eval_batch_size)
            save_arrays[_npz_key("clsscore", a, t)] = scores

            seg_iou = _batched_transform_then_seg(cache[a]["seg_adv"],
                                                  cache[a]["seg_masks"],
                                                  tf_func, tf_kw, per_image_seg_iou,
                                                  args.eval_batch_size)
            save_arrays[_npz_key("segiou", a, t)] = seg_iou
            print(f"    {t:16s}  mAP={compute_map(scores, save_arrays['cls_labels']):.1f}"
                  f"  mIoU={np.mean(seg_iou)*100:.1f}")

    meta = {"seed": args.seed, "model_name": args.model_name,
            "transforms": transforms, "n_cls": int(N_cls),
            "attacks": attacks_present}
    save_arrays["meta"] = json.dumps(meta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **save_arrays)
    print(f"\nSaved per-image artifact: {args.out}")


def _batched_transform_then(imgs, tf_func, tf_kw, fn, bs):
    """Apply transform batchwise then run fn on the transformed batch; concat."""
    import torch
    outs = []
    for s in range(0, imgs.shape[0], bs):
        e = min(s + bs, imgs.shape[0])
        batch = tf_func(imgs[s:e].clone(), **tf_kw)
        outs.append(fn(batch))
        del batch
    return np.concatenate(outs, 0)


def _batched_transform_then_seg(imgs, masks, tf_func, tf_kw, fn, bs):
    outs = []
    for s in range(0, imgs.shape[0], bs):
        e = min(s + bs, imgs.shape[0])
        batch = tf_func(imgs[s:e].clone(), **tf_kw)
        outs.append(fn(batch, masks[s:e]))
        del batch
    return np.concatenate(outs, 0)


# ==================================================================
# ---------------------------  CLI  --------------------------------
# ==================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Statistical analysis for TR-TAA vs TAA")
    sub = p.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("collect", help="dump per-image arrays (needs GPU+caches)")
    c.add_argument("--pascalvoc_root", required=True)
    c.add_argument("--heads_dir", default="./checkpoints")
    c.add_argument("--model_name", default="dinov2_vitb14")
    c.add_argument("--image_size", type=int, default=518)
    c.add_argument("--eval_batch_size", type=int, default=16)
    c.add_argument("--device", default="cuda")
    c.add_argument("--old_cache_dir", default="./outputs/cached_images",
                   help="dir holding the plain-TAA cache for this seed")
    c.add_argument("--new_cache_dir", default="./outputs/cached_images_clean",
                   help="dir holding the EoT-TAA K=4/K=8 cache for this seed")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--out", required=True, help="output .npz path")

    a = sub.add_parser("analyze", help="run tests + emit tables (no GPU)")
    a.add_argument("--npz", nargs="+", required=True,
                   help="one or more .npz (globs ok); one per seed")
    a.add_argument("--n_boot", type=int, default=2000)
    a.add_argument("--pooled_cls", action="store_true",
                   help="also run the slow pooled classification bootstrap "
                        "(~70 min; not reported in the paper)")
    a.add_argument("--boot_seed", type=int, default=12345)
    a.add_argument("--latex_out", default=None)
    a.add_argument("--json_out", default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.stage == "collect":
        collect_main(args)
    else:
        analyze_main(args)


if __name__ == "__main__":
    main()
