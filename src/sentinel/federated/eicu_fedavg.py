"""Phase 9 Step 6b: federated training across REAL eICU hospitals.

Phase 5 federated over SIMULATED sites (MIMIC care units, ~5 sites). Here the
sites are genuine institutions (eICU `hospitalid`), so the privacy-cost
measurement is on a realistic federation — many small, heterogeneous hospitals —
not a care-unit proxy.

Design (mirrors Phase 5: in-distribution test + held-out-site generalization):
  * Qualifying sites = hospitals with >= MIN_SITE_STAYS stays AND >= MIN_SITE_POS
    POSITIVE stays. eICU medication/culture reporting varies by site: 38 of 142
    hospitals with >=200 stays report ZERO sepsis (an all-negative site
    contributes a useless local update and biases FedAvg toward never-alert).
    Excluding them is both necessary and an honest real-federation finding.
  * N_UNSEEN_SITES hospitals are held out ENTIRELY as an unseen-hospital test —
    the eICU analog of Phase 5's held-out CVICU unit (does a federated model
    generalize to a NEW institution?).
  * Among the remaining federation hospitals, stays split into train / val /
    in-distribution-test. Centralized = one GRU on pooled train (val early-stop).
    Federated = FedAvg across hospital sites, no pooling. Both share architecture
    (GRUClassifier hidden=96) for a fair comparison.
  * Privacy cost Δ = Federated − Centralized AUPRC on each test set.

This trains ON eICU (a federation experiment over eICU hospitals), distinct from
the frozen MIMIC→eICU external validation of Steps 5/6a. Data split is fixed
(SPLIT_SEED) across model seeds; only model init/training varies per seed.
Self-contained FedAvg — reuses Phase 5 `_local_update` / `_fedavg`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import CohortConfig
from ..features.dataset import load_hourly, load_split
from ..features.partition import partition_path
from ..logging_utils import get_logger
from ..paths import PATHS
from ..baselines.gru import GRUClassifier
from ..baselines.torch_seq import EpisodeSeqDataset, device, predict_seq, train_seq
from ..federated.fedavg import _fedavg, _local_update
from ..eval import metrics as M

log = get_logger("federated.eicu")

MIN_SITE_STAYS = 200      # same floor as Phase 5
MIN_SITE_POS = 10         # exclude hospitals with too few positives to train/average
N_UNSEEN_SITES = 15       # held out entirely -> unseen-hospital generalization test
TEST_FRAC = 0.15          # within federation hospitals: in-distribution stay holdout
VAL_FRAC = 0.10           # within federation hospitals: centralized early-stop holdout
SPLIT_SEED = 12345        # data split is fixed across model seeds

# Phase 5 care-unit federation (federated_full.md) for the side-by-side.
PHASE5 = {"MIMIC test": (0.212, 0.121), "MIMIC external (CVICU)": (0.290, 0.237)}


def _federation_split(part: pd.DataFrame) -> pd.DataFrame:
    """Assign each modeling stay a (site, role); role in train/val/indist_test/unseen_test."""
    m = part[part["label"].notna()][["stay_id", "site", "label"]].copy()
    m["label"] = m["label"].astype(int)
    g = m.groupby("site").agg(n=("stay_id", "size"), pos=("label", "sum"))
    qual = sorted(g[(g.n >= MIN_SITE_STAYS) & (g.pos >= MIN_SITE_POS)].index.tolist())
    if len(qual) <= N_UNSEEN_SITES + 1:
        raise SystemExit(f"only {len(qual)} qualifying sites — too few to federate.")

    rng = np.random.default_rng(SPLIT_SEED)
    qual = list(rng.permutation(qual))
    unseen = set(qual[:N_UNSEEN_SITES])
    fed = qual[N_UNSEEN_SITES:]

    srng = np.random.default_rng(SPLIT_SEED + 1)
    role = {}
    for s in fed:
        ids = srng.permutation(m[m.site == s]["stay_id"].to_numpy())
        n_test = max(1, int(round(len(ids) * TEST_FRAC)))
        n_val = max(1, int(round(len(ids) * VAL_FRAC)))
        for sid in ids[:n_test]:
            role[sid] = "indist_test"
        for sid in ids[n_test:n_test + n_val]:
            role[sid] = "val"
        for sid in ids[n_test + n_val:]:
            role[sid] = "train"
    for sid in m[m.site.isin(unseen)]["stay_id"]:
        role[sid] = "unseen_test"

    m = m[m.site.isin(set(fed) | unseen)].copy()
    m["role"] = m["stay_id"].map(role)
    m.attrs["fed_sites"] = list(fed)
    m.attrs["unseen_sites"] = sorted(unseen)
    m.attrs["n_all_neg_excluded"] = int(((g.n >= MIN_SITE_STAYS) & (g.pos == 0)).sum())
    return m


def _seed_cache(seed: int):
    """Per-seed checkpoint (mirrors eicu_robustness). FedAvg over 78 sites × 15
    rounds is ~1.5 h; this laptop idle-sleeps on battery, so caching each seed
    makes the run resumable. Under outputs/ (gitignored, DUA-safe)."""
    return PATHS.output_root / "cache" / f"eicu_federated_seed{seed}.json"


def _ds(df_e, ids, feat):
    return EpisodeSeqDataset(df_e[df_e["stay_id"].isin(ids)], feat)


def _train_federated(df_e, site_train_ids, feat, pw, rounds, local_epochs, lr, seed):
    import torch
    torch.manual_seed(seed)
    dev = device()
    site_ds = {s: _ds(df_e, ids, feat) for s, ids in site_train_ids.items()}
    site_n = {s: len(ids) for s, ids in site_train_ids.items()}
    g = GRUClassifier(len(feat)).to(dev)
    state = {k: v.detach().cpu() for k, v in g.state_dict().items()}
    for r in range(rounds):
        states, weights = [], []
        for s in site_train_ids:
            states.append(_local_update(state, site_ds[s], feat, pw, local_epochs, lr, dev))
            weights.append(site_n[s])
        state = _fedavg(states, weights)
        if (r + 1) % 5 == 0 or r == rounds - 1:
            log.info("    fedavg round %d/%d (%d sites)", r + 1, rounds, len(site_train_ids))
    g.load_state_dict(state)
    return g


def _score(model, df_e, ids, feat):
    y, s = predict_seq(model, _ds(df_e, ids, feat), return_truth=True)
    d = M.discrimination(y, s)
    return {"auprc": d.auprc, "auroc": d.auroc, "no_skill": float(np.mean(y)),
            "n_stays": int(df_e[df_e["stay_id"].isin(ids)]["stay_id"].nunique())}


def run(eicu_cfg: CohortConfig | None = None, seeds=(0, 1, 2), rounds=15,
        local_epochs=1, lr=1e-3) -> None:
    eicu_cfg = eicu_cfg or CohortConfig(mode="eicu")
    t0 = time.perf_counter()

    part = pd.read_parquet(partition_path(eicu_cfg))
    roles = _federation_split(part)
    fed_sites = roles.attrs["fed_sites"]
    df_e = load_hourly(eicu_cfg).merge(roles[["stay_id", "site", "role"]],
                                       on="stay_id", how="inner")
    feat = load_split(eicu_cfg, "external").feature_names

    train_ids = roles[roles.role == "train"]["stay_id"].to_numpy()
    val_ids = roles[roles.role == "val"]["stay_id"].to_numpy()
    indist_ids = roles[roles.role == "indist_test"]["stay_id"].to_numpy()
    unseen_ids = roles[roles.role == "unseen_test"]["stay_id"].to_numpy()
    site_train_ids = {s: roles[(roles.site == s) & (roles.role == "train")]["stay_id"].to_numpy()
                      for s in fed_sites}

    yt = df_e[df_e.role == "train"]["y"].to_numpy()
    pw = (len(yt) - int(yt.sum())) / max(int(yt.sum()), 1)
    log.info("eICU federation: %d federation hospitals + %d unseen | train=%d val=%d "
             "indist-test=%d unseen-test=%d stays | %d all-negative hospitals excluded",
             len(fed_sites), N_UNSEEN_SITES, len(train_ids), len(val_ids),
             len(indist_ids), len(unseen_ids), roles.attrs["n_all_neg_excluded"])

    # Signature guards the seed caches against reuse across a changed split/config.
    sig = (f"{len(train_ids)}-{len(val_ids)}-{len(indist_ids)}-{len(unseen_ids)}-"
           f"{len(feat)}-{len(fed_sites)}-{rounds}-{SPLIT_SEED}")

    def _compute_seed(seed: int) -> dict:
        log.info("seed %d: centralized (pooled) + federated (FedAvg over %d hospitals)",
                 seed, len(fed_sites))
        central = GRUClassifier(len(feat))
        train_seq(central, _ds(df_e, train_ids, feat), _ds(df_e, val_ids, feat),
                  pw, epochs=25, seed=seed)
        fed = _train_federated(df_e, site_train_ids, feat, pw, rounds, local_epochs, lr, seed)
        scores = {}
        for name, ids in (("in-distribution", indist_ids), ("unseen-hospital", unseen_ids)):
            scores[f"Centralized|{name}"] = _score(central, df_e, ids, feat)
            scores[f"Federated|{name}"] = _score(fed, df_e, ids, feat)
        return {"_sig": sig, "scores": scores}

    res: dict[tuple, list] = {}
    for seed in seeds:
        cache = _seed_cache(seed)
        payload = None
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("_sig") == sig:
                payload = cached
                log.info("seed %d: loaded cache %s (skip training)", seed, cache.name)
            else:
                log.info("seed %d: cache %s stale (sig mismatch) — recomputing", seed, cache.name)
        if payload is None:
            payload = _compute_seed(seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
        for k, sc in payload["scores"].items():
            model, name = k.split("|")
            res.setdefault((model, name), []).append(sc)
        log.info("seed %d done", seed)

    _write_report(roles, res, seeds, rounds)
    log.info("eICU federation done in %.1fs", time.perf_counter() - t0)


def _write_report(roles, res, seeds, rounds) -> None:
    def agg(model, testset, key):
        return float(np.mean([r[key] for r in res[(model, testset)]]))

    testsets = ["in-distribution", "unseen-hospital"]
    nsk = {t: agg("Centralized", t, "no_skill") for t in testsets}
    nst = {t: int(np.mean([r["n_stays"] for r in res[("Centralized", t)]])) for t in testsets}

    L = [f"# SENTINEL — Federated vs centralized across REAL eICU hospitals\n",
         f"_FedAvg over genuine institutions (`hospitalid`), {rounds} rounds, no data "
         f"pooling; mean over {len(seeds)} seed(s). Privacy cost Δ = Federated − Centralized "
         f"AUPRC. Sites = hospitals with ≥{MIN_SITE_STAYS} stays AND ≥{MIN_SITE_POS} positives "
         f"({len(roles.attrs['fed_sites'])} federation + {N_UNSEEN_SITES} held-out unseen); "
         f"**{roles.attrs['n_all_neg_excluded']} hospitals with ≥{MIN_SITE_STAYS} stays but "
         f"ZERO positives were excluded** (incomplete med/culture reporting — a real-federation "
         f"heterogeneity that simulated care-unit sites cannot show)._\n",
         "| Test set | Stays | No-skill | Centralized | Federated | Privacy cost (Δ) | "
         "Cen AUROC | Fed AUROC |", "|---|---|---|---|---|---|---|---|"]
    for t in testsets:
        ca, fa = agg("Centralized", t, "auprc"), agg("Federated", t, "auprc")
        cu, fu = agg("Centralized", t, "auroc"), agg("Federated", t, "auroc")
        L.append(f"| {t} ({len(roles.attrs['fed_sites']) if t=='in-distribution' else N_UNSEEN_SITES} "
                 f"hosp) | {nst[t]} | {nsk[t]:.3f} | {ca:.3f} | {fa:.3f} | {fa-ca:+.3f} | "
                 f"{cu:.3f} | {fu:.3f} |")

    L.append("\n## Privacy cost — simulated care units (Phase 5) vs real hospitals (Phase 9)\n")
    L.append("| Federation | #sites | Centralized | Federated | Privacy cost (Δ) |")
    L.append("|---|---|---|---|---|")
    for name, (c, f) in PHASE5.items():
        L.append(f"| {name} (care units) | ~5 | {c:.3f} | {f:.3f} | {f-c:+.3f} |")
    ca = agg("Centralized", "in-distribution", "auprc")
    fa = agg("Federated", "in-distribution", "auprc")
    L.append(f"| **eICU in-distribution (real hospitals)** | **{len(roles.attrs['fed_sites'])}** | "
             f"**{ca:.3f}** | **{fa:.3f}** | **{fa-ca:+.3f}** |")
    L.append("\n_Many small heterogeneous hospitals vs a few large care units: the privacy cost "
             "of never centralizing patient data, measured on a realistic federation. The "
             "unseen-hospital row is the generalization test — does a federated model transfer "
             "to an institution it never trained on. Note: the federation cohort (≥10 positives/"
             "site) is sepsis-reporting-enriched, so its no-skill (~0.05) exceeds the full "
             "external cohort's 0.036; the Centralized−Federated Δ is unaffected (identical "
             "cohort in both arms)._")

    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / "eicu_federated.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
