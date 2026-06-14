"""Early-warning escalation reward (clinical utility).

Incentives (all weights configurable):
  * correct escalation in the pre-onset window  -> positive, with a one-time
    lead-time bonus for the FIRST timely alert (earlier detection = more value);
  * missing the onset entirely                  -> large terminal penalty;
  * escalating on a control / too early         -> small false-alarm penalty;
  * each `escalate`                              -> small standing cost (alarm fatigue).

Actions: 0=maintain, 1=watch, 2=escalate-alert. `watch` earns/risks a fraction
of `escalate` (a hedged middle action). The reward is a shared team reward.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    r_detect: float = 1.0          # reward for a correct escalate in-window
    watch_frac: float = 0.4        # watch = this fraction of escalate (reward & risk)
    c_false_alarm: float = 0.10    # penalty per false-alarm hour (escalate)
    c_escalate: float = 0.02       # standing per-escalate cost (alarm fatigue)
    r_miss: float = 2.0            # terminal penalty for a missed onset
    b_lead: float = 0.05           # per-hour lead-time bonus on first timely alert
    alert_window_hours: int = 6    # the pre-onset window that counts as "timely"


# action constants
MAINTAIN, WATCH, ESCALATE = 0, 1, 2
_STRENGTH = {MAINTAIN: 0.0, WATCH: None, ESCALATE: 1.0}  # WATCH filled from cfg


def _strength(action: int, cfg: RewardConfig) -> float:
    if action == WATCH:
        return cfg.watch_frac
    return 1.0 if action == ESCALATE else 0.0


def in_window(label: int, lead_hours: float, cfg: RewardConfig) -> bool:
    """True if this hour is in a positive stay's [onset-W, onset) alert window."""
    return label == 1 and 1 <= lead_hours <= cfg.alert_window_hours


def step_reward(label: int, action: int, lead_hours: float, already_alerted: bool,
                cfg: RewardConfig) -> tuple[float, bool]:
    """Return (reward, fired_timely_now). `lead_hours` = onset - t (NaN/inf for controls)."""
    r = 0.0
    if action == ESCALATE:
        r -= cfg.c_escalate
    strength = _strength(action, cfg)
    timely = in_window(label, lead_hours, cfg)
    fired_now = False
    if timely:
        r += cfg.r_detect * strength
        if strength > 0 and not already_alerted:           # first timely alert
            r += cfg.b_lead * lead_hours                    # earlier => bigger bonus
            fired_now = True
    else:
        r -= cfg.c_false_alarm * strength                  # control or too-early
    return r, fired_now


def terminal_reward(label: int, alerted_in_window: bool, cfg: RewardConfig) -> float:
    """Large penalty if a true onset was never flagged during its window."""
    if label == 1 and not alerted_in_window:
        return -cfg.r_miss
    return 0.0
