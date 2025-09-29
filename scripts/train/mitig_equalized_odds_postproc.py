#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np, pandas as pd
from collections import defaultdict

# === Configura qui le tue classi e gruppi ===
CLS_ORDER = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
GROUPS = ['Light','Dark']

# ---------- Metriche binarie per EqOdds ----------
def _bin_metrics(y_true_bin, y_pred_bin):
    tp = np.sum((y_true_bin==1) & (y_pred_bin==1))
    fp = np.sum((y_true_bin==0) & (y_pred_bin==1))
    tn = np.sum((y_true_bin==0) & (y_pred_bin==0))
    fn = np.sum((y_true_bin==1) & (y_pred_bin==0))
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return tpr, fpr, acc

def _one_vs_rest_columns(df):
    return [f"p_{k}" for k in CLS_ORDER]

# ---------- Fit soglie Equalized Odds su validation ----------
def fit_equalized_odds_thresholds(val_df, lambda_acc=0.05, grid=101):
    """
    Impara soglie per classe e gruppo minimizzando |ΔTPR|+|ΔFPR| + λ * calo-accuracy.
    val_df: columns ['y','group'] + 'p_<class>' per ogni classe.
    Ritorna: dict thresholds[k][group] = soglia float
    """
    thresholds = {k:{} for k in CLS_ORDER}
    for k in CLS_ORDER:
        y_bin = (val_df['y'] == k).astype(int).values
        scores = val_df[f'p_{k}'].values

        # Candidati = quantili (stabili e veloci)
        cand = np.unique(np.quantile(scores, np.linspace(0, 1, grid)))
        best = (1e9, 0.5, 0.5)  # (objective, t_L, t_D)

        idxL = (val_df['group'] == 'Light').values
        idxD = (val_df['group'] == 'Dark').values

        yL, sL = y_bin[idxL], scores[idxL]
        yD, sD = y_bin[idxD], scores[idxD]

        # accuracy di riferimento (soglia 0.5 per confronto)
        yhatL_b = (sL >= 0.5).astype(int); yhatD_b = (sD >= 0.5).astype(int)
        _,_,accL_b = _bin_metrics(yL, yhatL_b); _,_,accD_b = _bin_metrics(yD, yhatD_b)
        acc_ref = 0.5*(accL_b + accD_b)

        for tL in cand:
            yhatL = (sL >= tL).astype(int)
            tprL, fprL, accL = _bin_metrics(yL, yhatL)
            for tD in cand:
                yhatD = (sD >= tD).astype(int)
                tprD, fprD, accD = _bin_metrics(yD, yhatD)

                gap = abs(tprL - tprD) + abs(fprL - fprD)
                acc_mean = 0.5*(accL + accD)
                obj = gap + lambda_acc * max(0.0, (acc_ref - acc_mean))

                if obj < best[0]:
                    best = (obj, float(tL), float(tD))

        thresholds[k]['Light'] = best[1]
        thresholds[k]['Dark']  = best[2]
    return thresholds

# ---------- Applicazione soglie al test (multiclasse) ----------
def apply_equalized_odds(test_df, thresholds):
    """
    Schema one-vs-rest: per ogni classe k e gruppo g applica la soglia t_{k,g}.
    Se più classi positive -> sceglie quella con prob massima.
    Se nessuna -> fallback argmax.
    Ritorna: np.array di predizioni (stringhe classe).
    """
    preds = []
    prob_cols = _one_vs_rest_columns(test_df)
    probs_mat = test_df[prob_cols].values
    groups = test_df['group'].values
    for i in range(len(test_df)):
        s = probs_mat[i]
        g = groups[i]
        positives = []
        for j,k in enumerate(CLS_ORDER):
            t = thresholds[k][g]
            if s[j] >= t:
                positives.append((s[j], j))
        if positives:
            j_star = max(positives)[1]
        else:
            j_star = int(np.argmax(s))
        preds.append(CLS_ORDER[j_star])
    return np.array(preds)

# ---------- Metriche riassuntive ----------
def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))

def macro_f1(y_true, y_pred):
    # implementazione semplice senza sklearn
    f1s = []
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    for k in CLS_ORDER:
        yt = (y_true == k).astype(int)
        yp = (y_pred == k).astype(int)
        tp = np.sum((yt==1)&(yp==1)); fp = np.sum((yt==0)&(yp==1))
        fn = np.sum((yt==1)&(yp==0))
        prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
        f1 = 2*prec*rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def per_group_scores(y_true, y_pred, groups):
    res = {}
    y_true = np.array(y_true); y_pred = np.array(y_pred); groups = np.array(groups)
    for g in GROUPS:
        idx = (groups==g)
        if not idx.any():
            res[g] = {'acc': np.nan, 'macro_f1': np.nan,
                      'weighted_f1': np.nan, 'micro_f1': np.nan}
            continue
        yt_g = y_true[idx]; yp_g = y_pred[idx]
        res[g] = {
            'acc': accuracy(yt_g, yp_g),
            'macro_f1': macro_f1(yt_g, yp_g),
            'weighted_f1': weighted_f1(yt_g, yp_g),
            'micro_f1': micro_f1(yt_g, yp_g),
        }
    return res


def f1_per_class(y_true, y_pred):
    """Restituisce dict {classe: (f1, support)}."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    stats = {}
    for k in CLS_ORDER:
        yt = (y_true == k).astype(int)
        yp = (y_pred == k).astype(int)
        tp = np.sum((yt==1)&(yp==1))
        fp = np.sum((yt==0)&(yp==1))
        fn = np.sum((yt==1)&(yp==0))
        support = tp + fn
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2*prec*rec / (prec + rec + 1e-12)
        stats[k] = (float(f1), int(support))
    return stats

def weighted_f1(y_true, y_pred):
    """Media pesata per supporto dei F1 per classe (aka weighted-F1)."""
    stats = f1_per_class(y_true, y_pred)
    num = 0.0; den = 0
    for k, (f1, supp) in stats.items():
        num += f1 * supp
        den += supp
    return float(num / (den + 1e-12))

def micro_f1(y_true, y_pred):
    """
    Micro-F1 su problema multi-classe a etichetta singola.
    (Equivale all'accuracy, ma lo calcoliamo esplicitamente.)
    """
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = fp = fn = 0
    for k in CLS_ORDER:
        yt = (y_true == k).astype(int)
        yp = (y_pred == k).astype(int)
        tp += np.sum((yt==1)&(yp==1))
        fp += np.sum((yt==0)&(yp==1))
        fn += np.sum((yt==1)&(yp==0))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    return float(2*prec*rec / (prec + rec + 1e-12))

# ---------- TPR/FPR per classe e gruppo (one-vs-rest) ----------
def tpr_fpr_by_class_group(y_true, y_pred, groups):
    table = []
    y_true = np.array(y_true); y_pred = np.array(y_pred); groups = np.array(groups)
    for k in CLS_ORDER:
        y_bin = (y_true == k).astype(int)
        p_bin = (y_pred == k).astype(int)
        for g in GROUPS:
            idx = (groups==g)
            if not idx.any():
                tpr=fpr=np.nan
            else:
                tpr, fpr, _ = _bin_metrics(y_bin[idx], p_bin[idx])
            table.append({'class': k, 'group': g, 'TPR': tpr, 'FPR': fpr})
    return pd.DataFrame(table)

# ---------- Utility ----------
def _ensure_columns(df):
    missing = []
    base = ['y','group'] + _one_vs_rest_columns(df)
    for c in base:
        if c not in df.columns:
            missing.append(c)
    if missing:
        raise ValueError(f"Mancano colonne: {missing}. Attese: 'y','group' e 'p_<classe>' per {CLS_ORDER}")

def _coerce_labels(df):
    # Se y è numerica 0..C-1, mappala a nomi classe
    if np.issubdtype(df['y'].dtype, np.number):
        idx = df['y'].astype(int).values
        if (idx.min() < 0) or (idx.max() >= len(CLS_ORDER)):
            raise ValueError("Valori 'y' fuori range per CLS_ORDER.")
        df['y'] = [CLS_ORDER[i] for i in idx]
    else:
        df['y'] = df['y'].astype(str)
    # group in stringa
    df['group'] = df['group'].astype(str)
    return df

def _argmax_pred(df):
    probs = df[_one_vs_rest_columns(df)].values
    j = np.argmax(probs, axis=1)
    return np.array([CLS_ORDER[i] for i in j])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Equalized Odds (multiclasse, one-vs-rest) — fit su validation, applica su test.")
    ap.add_argument("--val_csv",  required=True, help="CSV con colonne: y, group, p_<classe>")
    ap.add_argument("--test_csv", required=True, help="CSV con colonne: y, group, p_<classe>")
    ap.add_argument("--out_dir",  required=True, help="Cartella output")
    ap.add_argument("--lambda_acc", type=float, default=0.05, help="Penalità sul calo di accuracy (default 0.05)")
    ap.add_argument("--grid", type=int, default=101, help="Numero soglie (quantili) per gruppo (default 101)")
    ap.add_argument("--emit_latex", action="store_true", help="Se passato, salva una tabella LaTeX con TPR/FPR per classe/gruppo (prima vs dopo).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Carica CSV
    val_df  = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    # Check colonne e coercizioni
    _ensure_columns(val_df); _ensure_columns(test_df)
    val_df  = _coerce_labels(val_df)
    test_df = _coerce_labels(test_df)

    # --- Baseline (argmax) su test ---
    y_true_test = test_df['y'].values
    groups_test = test_df['group'].values
    y_hat_base  = _argmax_pred(test_df)

    base_overall = {
    "acc": accuracy(y_true_test, y_hat_base),
    "macro_f1": macro_f1(y_true_test, y_hat_base),
    "weighted_f1": weighted_f1(y_true_test, y_hat_base),
    "micro_f1": micro_f1(y_true_test, y_hat_base),}

    base_groups  = per_group_scores(y_true_test, y_hat_base, groups_test)
    base_tpfpr   = tpr_fpr_by_class_group(y_true_test, y_hat_base, groups_test)

    # --- Fit thresholds su validation ---
    thresholds = fit_equalized_odds_thresholds(val_df, lambda_acc=args.lambda_acc, grid=args.grid)
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # --- Applica su test ---
    y_hat_eq = apply_equalized_odds(test_df, thresholds)

    # --- Metriche dopo EqOdds ---
    eq_overall = {
    "acc": accuracy(y_true_test, y_hat_eq),
    "macro_f1": macro_f1(y_true_test, y_hat_eq),
    "weighted_f1": weighted_f1(y_true_test, y_hat_eq),
    "micro_f1": micro_f1(y_true_test, y_hat_eq),}

    eq_groups  = per_group_scores(y_true_test, y_hat_eq, groups_test)
    eq_tpfpr   = tpr_fpr_by_class_group(y_true_test, y_hat_eq, groups_test)

    # --- Salva CSV con predizioni corrette ---
    out_csv = os.path.join(args.out_dir, "test_preds_eq.csv")
    out_df = test_df.copy()
    out_df["y_pred_baseline"] = y_hat_base
    out_df["y_pred_eqodds"]   = y_hat_eq
    out_df.to_csv(out_csv, index=False)

    # --- Report JSON (prima vs dopo) ---
    report = {
    "overall": {"baseline": base_overall, "eq_odds": eq_overall},
    "per_group": {"baseline": base_groups, "eq_odds": eq_groups},}

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- TPR/FPR tables ---
    base_tpfpr.to_csv(os.path.join(args.out_dir, "tpr_fpr_baseline.csv"), index=False)
    eq_tpfpr.to_csv(os.path.join(args.out_dir, "tpr_fpr_eqodds.csv"), index=False)

    # --- (Opzionale) Tabella LaTeX compatta ---
    if args.emit_latex:
        def _compact(df):
            rows = []
            for k in CLS_ORDER:
                row = [k]
                for g in GROUPS:
                    r = df[(df['class']==k) & (df['group']==g)]
                    tpr = float(r['TPR']) if len(r) else np.nan
                    fpr = float(r['FPR']) if len(r) else np.nan
                    row += [tpr, fpr]
                rows.append(row)
            cols = ["Class", "TPR_L", "FPR_L", "TPR_D", "FPR_D"]
            return pd.DataFrame(rows, columns=cols)

        tab_before = _compact(base_tpfpr)
        tab_after  = _compact(eq_tpfpr)

        with open(os.path.join(args.out_dir, "tpr_fpr_tables.tex"), "w") as f:
            f.write("% Baseline\n")
            f.write(tab_before.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\n% After Equalized Odds\n")
            f.write(tab_after.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- Log sintetico a schermo ---
    gap_base = base_groups['Light']['acc'] - base_groups['Dark']['acc']
    gap_eq   = eq_groups['Light']['acc']  - eq_groups['Dark']['acc']

    print("\n== Baseline (test) ==")
    print(f"Overall: acc={base_overall['acc']:.3f}  macroF1={base_overall['macro_f1']:.3f}  "
        f"weightedF1={base_overall['weighted_f1']:.3f}  microF1={base_overall['micro_f1']:.3f}")
    print("Per-group:")
    print(f"  Light: acc={base_groups['Light']['acc']:.3f}  macroF1={base_groups['Light']['macro_f1']:.3f}  "
        f"weightedF1={base_groups['Light']['weighted_f1']:.3f}  microF1={base_groups['Light']['micro_f1']:.3f}")
    print(f"  Dark : acc={base_groups['Dark']['acc']:.3f}  macroF1={base_groups['Dark']['macro_f1']:.3f}  "
        f"weightedF1={base_groups['Dark']['weighted_f1']:.3f}  microF1={base_groups['Dark']['micro_f1']:.3f}")
    print(f"  Gap(L-D) acc = {gap_base:+.3f}")

    print("\n== Equalized Odds (test) ==")
    print(f"Overall: acc={eq_overall['acc']:.3f}  macroF1={eq_overall['macro_f1']:.3f}  "
        f"weightedF1={eq_overall['weighted_f1']:.3f}  microF1={eq_overall['micro_f1']:.3f}")
    print("Per-group:")
    print(f"  Light: acc={eq_groups['Light']['acc']:.3f}  macroF1={eq_groups['Light']['macro_f1']:.3f}  "
        f"weightedF1={eq_groups['Light']['weighted_f1']:.3f}  microF1={eq_groups['Light']['micro_f1']:.3f}")
    print(f"  Dark : acc={eq_groups['Dark']['acc']:.3f}  macroF1={eq_groups['Dark']['macro_f1']:.3f}  "
        f"weightedF1={eq_groups['Dark']['weighted_f1']:.3f}  microF1={eq_groups['Dark']['micro_f1']:.3f}")
    print(f"  Gap(L-D) acc = {gap_eq:+.3f}")

    if args.emit_latex:
        print(f"Tabelle LaTeX in: {os.path.join(args.out_dir, 'tpr_fpr_tables.tex')}")

if __name__ == "__main__":
    main()



# bash: 
# python mitig_equalized_odds_postproc.py \
#   --val_csv val_preds.csv --test_csv test_preds.csv \
#   --out_dir out_eqodds_l008_g101 \
#   --lambda_acc 0.08 --grid 101