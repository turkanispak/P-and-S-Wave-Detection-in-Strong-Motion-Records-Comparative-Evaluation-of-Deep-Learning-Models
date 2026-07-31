#!/usr/bin/env python3
# =============================================================================
# bootstrap_ci.py
# Record-level bootstrap confidence intervals for every table cell, computed
# purely from the frozen per-sample prediction CSVs (no model, no rerun).
# Differences whose intervals overlap are not interpretable as
# rankings. Records (trace_name) are the resampling unit, so the class
# variants of one record stay together (cluster bootstrap).
# Edit CONFIG and click Run.
# =============================================================================
import os, glob, re
import numpy as np
import pandas as pd

# ------------------------------- CONFIG --------------------------------------
CSV_DIR   = './holdout_inference'
OUT_DIR   = './holdout_inference'
N_BOOT    = 1000
SEED      = 42            # bootstrap resampling seed (analysis reproducibility)
CI        = 95            # percent
# -----------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score

def metrics(df):
    y, yp = df.y_true.values, df.pred_class.values
    p_gt, s_gt = (y >= 1), (y == 2)
    out = {}
    for ph, gt, prob in (('p', p_gt, df.p_prob.values),
                         ('s', s_gt, df.s_prob.values)):
        out[f'{ph}_auc'] = (roc_auc_score(gt.astype(int), prob)
                            if 0 < gt.sum() < len(df) else np.nan)
    praw = (df.p_pred_raw if 'p_pred_raw' in df.columns else df.p_pred).values
    sraw = (df.s_pred_raw if 's_pred_raw' in df.columns else df.s_pred).values
    pe, se = p_gt & (praw >= 0), s_gt & (sraw >= 0)
    out['p_mae'] = float(np.abs(praw[pe] - df.p_true.values[pe]).mean()) if pe.any() else np.nan
    out['s_mae'] = float(np.abs(sraw[se] - df.s_true.values[se]).mean()) if se.any() else np.nan
    out['p_tpr'] = float((p_gt & (yp >= 1)).sum() / max(p_gt.sum(), 1))
    out['s_tpr'] = float((s_gt & (yp == 2)).sum() / max(s_gt.sum(), 1))
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lo_q, hi_q = (100 - CI) / 200, 1 - (100 - CI) / 200
    rows = []
    files = sorted(glob.glob(os.path.join(CSV_DIR, 'predictions_*.csv')))
    files = [f for f in files if '_seed' not in os.path.basename(f)]
    assert files, f'no prediction CSVs under {CSV_DIR}'
    for fp in files:
        m = re.match(r'predictions_(.+)_(\d+)s\.csv', os.path.basename(fp))
        model, dur = m.group(1), int(m.group(2))
        df = pd.read_csv(fp)
        recs = df.trace_name.unique()
        groups = {r: g for r, g in df.groupby('trace_name')}
        point = metrics(df)
        boots = {k: [] for k in point}
        for _ in range(N_BOOT):
            pick = rng.choice(recs, size=len(recs), replace=True)
            bdf = pd.concat([groups[r] for r in pick], ignore_index=True)
            bm = metrics(bdf)
            for k, v in bm.items():
                if not np.isnan(v):
                    boots[k].append(v)
        for k, v in point.items():
            b = np.asarray(boots[k])
            rows.append({'model': model, 'duration_s': dur, 'metric': k,
                         'value': v,          # unrounded: display must match
                         'ci_lo': float(np.quantile(b, lo_q)) if len(b) else np.nan,
                         'ci_hi': float(np.quantile(b, hi_q)) if len(b) else np.nan,
                         'n_boot_valid': len(b)})   # the main tables digit-for-digit
        print(f'{model:14s} {dur:>3d}s done')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'bootstrap_ci.csv'), index=False)
    for metric in ('p_auc', 's_auc', 'p_mae', 's_mae'):
        sub = out[out.metric == metric]
        grid = sub.pivot(index='model', columns='duration_s', values=['value', 'ci_lo', 'ci_hi'])
        print(f'\n=== {metric} with {CI}% bootstrap CI (record-level, B={N_BOOT}) ===')
        for mdl in sub.model.unique():
            cells = []
            for d in sorted(sub.duration_s.unique()):
                r = sub[(sub.model == mdl) & (sub.duration_s == d)]
                if len(r) and not np.isnan(r.value.iloc[0]):
                    cells.append(f'{d}s {r.value.iloc[0]:.2f} [{r.ci_lo.iloc[0]:.2f}-{r.ci_hi.iloc[0]:.2f}]')
                else:
                    cells.append(f'{d}s NA')
            print(f'  {mdl:14s} ' + ' | '.join(cells))
    print(f'\n-> {os.path.join(OUT_DIR, "bootstrap_ci.csv")}')

if __name__ == '__main__':
    main()