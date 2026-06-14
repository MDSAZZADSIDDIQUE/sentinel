"""Unit tests for the early-warning escalation reward (synthetic episodes)."""
from __future__ import annotations

import math

from sentinel.env.reward import (ESCALATE, MAINTAIN, WATCH, RewardConfig,
                                  in_window, step_reward, terminal_reward)

C = RewardConfig()


def test_in_window():
    assert in_window(1, 3, C) and in_window(1, 1, C) and in_window(1, C.alert_window_hours, C)
    assert not in_window(1, 0, C)        # at/after onset is not "before"
    assert not in_window(1, C.alert_window_hours + 1, C)  # too early
    assert not in_window(0, 3, C)        # controls never in-window


def test_correct_timely_escalate_is_rewarded():
    r, fired = step_reward(label=1, action=ESCALATE, lead_hours=3, already_alerted=False, cfg=C)
    assert fired
    assert math.isclose(r, -C.c_escalate + C.r_detect + C.b_lead * 3)
    assert r > 0


def test_watch_is_a_hedged_middle_action():
    r_esc, _ = step_reward(1, ESCALATE, 3, True, C)   # already alerted -> no lead bonus
    r_watch, _ = step_reward(1, WATCH, 3, True, C)
    r_maint, _ = step_reward(1, MAINTAIN, 3, True, C)
    assert r_maint == 0
    assert 0 < r_watch < r_esc          # watch earns a fraction of escalate


def test_false_alarm_on_control_penalized():
    r, fired = step_reward(label=0, action=ESCALATE, lead_hours=float("inf"),
                           already_alerted=False, cfg=C)
    assert not fired
    assert math.isclose(r, -C.c_escalate - C.c_false_alarm)
    assert r < 0


def test_too_early_escalation_is_a_false_alarm():
    r, _ = step_reward(1, ESCALATE, lead_hours=C.alert_window_hours + 5,
                       already_alerted=False, cfg=C)
    assert math.isclose(r, -C.c_escalate - C.c_false_alarm)


def test_missed_onset_terminal_penalty():
    assert terminal_reward(1, alerted_in_window=False, cfg=C) == -C.r_miss
    assert terminal_reward(1, alerted_in_window=True, cfg=C) == 0.0
    assert terminal_reward(0, alerted_in_window=False, cfg=C) == 0.0


def test_earlier_detection_earns_more_lead_bonus():
    r_early, _ = step_reward(1, ESCALATE, lead_hours=6, already_alerted=False, cfg=C)
    r_late, _ = step_reward(1, ESCALATE, lead_hours=1, already_alerted=False, cfg=C)
    assert r_early > r_late      # more lead time = more clinical value


def test_maintain_is_neutral_everywhere():
    assert step_reward(1, MAINTAIN, 3, False, C)[0] == 0.0
    assert step_reward(0, MAINTAIN, float("inf"), False, C)[0] == 0.0
