"""SENTINEL interpretability dashboard (Phase 7).

Replays one ICU stay hour-by-hour and shows WHY the team alerts: the six
per-organ risk estimates over time, the team risk (max-combine), the escalation
timeline at a chosen alarm threshold, and the true Sepsis-3 onset. The per-organ
decomposition is the point — you can see which organ system drove an alert.

Run:  streamlit run dashboard/app.py        (uses the dev feature tensor)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import sentinel  # noqa: F401  (applies KMP/runtime env before torch)
from sentinel.config import CohortConfig
from sentinel.constants import ORGAN_SYSTEMS
from sentinel.paths import PATHS

st.set_page_config(page_title="SENTINEL — organ-system early warning", layout="wide")
ORGAN_COLORS = {
    "respiratory": "#1f77b4", "coagulation": "#9467bd", "hepatic": "#8c564b",
    "cardiovascular": "#d62728", "neurologic": "#2ca02c", "renal": "#ff7f0e",
}
MODE = "dev"


@st.cache_resource(show_spinner="Training per-organ risk heads (one-time)…")
def _load():
    import json
    from sentinel.baselines.gru import train_organ_ensemble
    from sentinel.features.dataset import load_hourly, load_split

    cfg = CohortConfig(mode=MODE)
    df = load_hourly(cfg)
    with (PATHS.features_root / f"feature_manifest_{MODE}.json").open() as fh:
        manifest = json.load(fh)
    tr = load_split(cfg, "train", df=df)
    pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
    models = train_organ_ensemble(df, tr.manifest, pw, seed=0, epochs=20)
    return cfg, df, manifest, models


def _organ_risks(models, df, stay_id) -> dict:
    from sentinel.baselines.torch_seq import EpisodeSeqDataset, predict_seq
    one = df[df["stay_id"] == stay_id]
    out = {}
    for organ, (model, cols) in models.items():
        out[organ] = predict_seq(model, EpisodeSeqDataset(one, cols))
    return out


def main():
    feat_path = PATHS.features_root / f"hourly_{MODE}.parquet"
    st.title("SENTINEL — organ-system early warning")
    if not feat_path.exists():
        st.error(f"Features not built. Run `sentinel build-features --mode {MODE}` first.")
        return

    cfg, df, manifest, models = _load()
    st.caption("Per-organ supervised risk heads + max-combine. Each line is one "
               "organ agent's onset risk; the team alerts on the most-alarmed organ.")

    # --- pick a stay (default to a positive with a clear onset) ---
    ep = df.groupby("stay_id").agg(label=("label", "first"),
                                   onset=("t_to_onset", lambda s: s.iloc[0]),
                                   n=("hr", "size")).reset_index()
    pos = ep[(ep.label == 1) & (ep.n >= 12)].sort_values("n", ascending=False)
    c1, c2 = st.columns([1, 3])
    with c1:
        only_pos = st.checkbox("Septic stays only", value=True)
        pool = pos if only_pos else ep
        stay_id = st.selectbox("ICU stay", pool["stay_id"].tolist())
        thr = st.slider("Alarm threshold (team risk)", 0.0, 1.0, 0.5, 0.05)

    one = df[df["stay_id"] == stay_id].sort_values("hr")
    hrs = one["hr"].to_numpy()
    label = int(one["label"].iloc[0])
    lead0 = one["t_to_onset"].iloc[0]
    onset_hr = int(hrs[0] + lead0) if label == 1 and np.isfinite(lead0) else None
    risks = _organ_risks(models, df, stay_id)
    team = np.maximum.reduce([risks[o] for o in ORGAN_SYSTEMS])

    with c2:
        st.metric("Stay", f"{stay_id}",
                  f"{'SEPTIC — onset hr ' + str(onset_hr) if label == 1 else 'control (never Sepsis-3)'}")

    # --- timeline plot ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for o in ORGAN_SYSTEMS:
        ax.plot(hrs, risks[o], color=ORGAN_COLORS[o], alpha=0.55, lw=1.3, label=o)
    ax.plot(hrs, team, color="black", lw=2.4, label="TEAM (max)")
    ax.axhline(thr, ls="--", color="gray", lw=1)
    alarms = hrs[team >= thr]
    if alarms.size:
        ax.scatter(alarms, team[team >= thr], color="black", s=18, zorder=5)
        ax.annotate(f"first alarm hr {int(alarms.min())}", (alarms.min(), thr),
                    textcoords="offset points", xytext=(4, 8), fontsize=8)
    if onset_hr is not None:
        ax.axvline(onset_hr, color="red", lw=2)
        ax.annotate("Sepsis-3 onset", (onset_hr, 0.95), color="red", fontsize=9,
                    ha="right", rotation=90, va="top")
    ax.set_xlabel("hours since ICU admission"); ax.set_ylabel("onset risk")
    ax.set_ylim(0, 1); ax.legend(ncol=4, fontsize=8, loc="upper left")
    st.pyplot(fig)

    # --- per-hour attribution (which organ drives the alert) ---
    st.subheader("Per-organ attribution at hour t")
    t = st.slider("hour", int(hrs.min()), int(hrs.max()), int(hrs.min()))
    row = {o: float(risks[o][np.where(hrs == t)[0][0]]) for o in ORGAN_SYSTEMS}
    bar = pd.DataFrame({"organ": list(row), "risk": list(row.values())}).set_index("organ")
    st.bar_chart(bar, color="#d62728")
    top = max(row, key=row.get)
    msg = f"At hr {t}, **{top}** drives the team risk ({row[top]:.2f})."
    if onset_hr is not None:
        lead = onset_hr - t
        msg += f"  Onset is {'in ' + str(lead) + ' h' if lead > 0 else 'past'}."
    st.markdown(msg)
    st.caption("This per-organ attribution is the decomposition's value — a "
               "monolithic model gives one opaque score with no organ breakdown.")


if __name__ == "__main__":
    main()
