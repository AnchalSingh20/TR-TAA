"""
metrics_retrieval_ref.py
========================
Reference R-Oxford scorer, faithful to the official revisitop implementation
(filipradenovic/revisitop, python/evaluate.py). Same public signatures as the
existing src/metrics_retrieval.py so it is a drop-in:

    ranks = rank_database(q_feats, db_feats)      # (n_db, n_q)
    m, aps = compute_map(ranks, gnd_proto)        # m in [0, 1]

The key correctness detail vs. many reimplementations: JUNK images are REMOVED
from the ranked list (positive positions are shifted left by the number of junk
ranked above them) before AP is computed. Skipping this deflates every AP and
is the usual cause of a ~15-20 pt uniform mAP gap on R-Oxford.

gnd_proto : list (len n_q) of dicts, each { 'ok': int-array, 'junk': int-array }
            exactly what build_difficulty_gnd(cfg)[protocol] already returns.
"""

import numpy as np


def compute_ap(pos, nres):
    """
    Average precision for the (junk-removed, 0-based) sorted positions `pos`
    of the positive images. `nres` = number of positives. Trapezoidal PR area,
    identical to revisitop.
    """
    nimgranks = len(pos)
    ap = 0.0
    recall_step = 1.0 / nres
    for j in range(nimgranks):
        rank = pos[j]
        precision_0 = 1.0 if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def rank_database(q_feats, db_feats):
    """
    q_feats : (n_q, D), db_feats : (n_db, D). Descriptors should already be
    L2-normalized (cosine == dot product). Returns integer `ranks` of shape
    (n_db, n_q); ranks[:, i] are db indices for query i, best match first.
    """
    q = np.asarray(q_feats, dtype=np.float32)
    db = np.asarray(db_feats, dtype=np.float32)
    sims = q @ db.T                      # (n_q, n_db)
    ranks = np.argsort(-sims, axis=1).T  # (n_db, n_q)
    return np.ascontiguousarray(ranks)


def compute_map(ranks, gnd):
    """
    Mean average precision with correct junk removal. Queries with no positives
    are skipped. Returns (map, aps) with map in [0, 1].
    """
    n_q = len(gnd)
    aps = np.zeros(n_q, dtype=np.float64)
    map_ = 0.0
    nempty = 0

    for i in range(n_q):
        ok = np.asarray(gnd[i].get("ok", []), dtype=int)
        if ok.size == 0:
            aps[i] = np.nan
            nempty += 1
            continue
        junk = np.asarray(gnd[i].get("junk", []), dtype=int)

        col = ranks[:, i]
        pos = np.arange(ranks.shape[0])[np.isin(col, ok)]
        jnk = np.arange(ranks.shape[0])[np.isin(col, junk)]

        # Remove junk: shift each positive left by #junk ranked above it.
        if jnk.size:
            k = 0
            ij = 0
            ip = 0
            pos = pos.copy()
            while ip < len(pos):
                while ij < len(jnk) and pos[ip] > jnk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        ap = compute_ap(pos, ok.size)
        map_ += ap
        aps[i] = ap

    denom = max(1, (n_q - nempty))
    return map_ / denom, aps
