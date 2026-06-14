"""Project-wide constants: organ systems, escalation actions, variable roles.

The six organ systems are aligned with the SOFA score and define the SENTINEL
agent team. Keep this list authoritative — env agents, feature grouping, and the
dashboard all key off it.
"""
from __future__ import annotations

# SOFA-aligned organ systems = the cooperative agent team.
ORGAN_SYSTEMS: tuple[str, ...] = (
    "respiratory",
    "coagulation",
    "hepatic",
    "cardiovascular",
    "neurologic",
    "renal",
)

# Context not owned by a single organ agent (shared / global features).
SHARED_GROUP = "shared"

# Joint escalation action space (per the central design — alerting, not dosing).
ESCALATION_ACTIONS: tuple[str, ...] = ("maintain", "watch", "escalate-alert")

# Raw-table provenance for resolved variables.
SOURCES: tuple[str, ...] = (
    "chartevents",
    "labevents",
    "inputevents",
    "outputevents",
)
