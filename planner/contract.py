"""Planner semantic contract shared by offline and online paths.

The planner owns only the semantic pair ``action + instruction``.  Runtime
behavior such as whether to inject VQAP codes is derived locally from the
normalized action so an external planner cannot change model-control policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


ACTION_WHITELIST = frozenset(
    {
        "approach",
        "grasp",
        "lift",
        "transfer",
        "place",
        "push",
        "pull",
        "press",
        "rotate",
        "slide",
        "insert",
        "hang",
        "wipe",
        "flip-open",
        "flip-close",
        "revolve-in",
        "revolve-out",
        "pose-adjust",
    }
)
POSE_ADJUST_ACTION = "pose-adjust"
PLANNER_SEMANTIC_CONTRACT_VERSION = "vqap_subtask_instruction_v2"
CodePolicy = Literal["normal", "always_off"]


class PlannerContractError(ValueError):
    """Planner output cannot be normalized without guessing its semantics."""


@dataclass(frozen=True)
class NormalizedSubtask:
    """The only semantic fields supplied by either planner path."""

    action: str
    instruction: str


def normalize_action(value: Any) -> str:
    """Return one canonical action label or fail instead of guessing a synonym."""

    action = str(value or "").strip().lower().replace("_", "-")
    if action not in ACTION_WHITELIST:
        raise PlannerContractError(
            f"Unsupported planner action={action!r}; expected one of {sorted(ACTION_WHITELIST)}."
        )
    return action


def normalize_instruction(value: Any) -> str:
    """Normalize a short one-line instruction while rejecting malformed output."""

    raw = str(value or "").strip()
    if not raw:
        raise PlannerContractError("Planner instruction must be non-empty.")
    if "\n" in raw or "\r" in raw:
        raise PlannerContractError("Planner instruction must be a single line.")
    instruction = " ".join(raw.split())
    if len(instruction) > 128:
        raise PlannerContractError("Planner instruction must contain at most 128 characters.")
    if len(instruction.split()) > 16:
        raise PlannerContractError("Planner instruction must contain at most 16 words.")
    return instruction


def normalize_planner_subtask(value: Mapping[str, Any]) -> NormalizedSubtask:
    """Normalize only the planner-owned semantic fields."""

    if not isinstance(value, Mapping):
        raise PlannerContractError("Planner subtask output must be an object.")
    return NormalizedSubtask(
        action=normalize_action(value.get("action")),
        instruction=normalize_instruction(value.get("instruction")),
    )


def derive_code_policy(action: str) -> CodePolicy:
    """Derive VQAP-code behavior solely from the canonical action.

    ``pose-adjust`` is not a stable atomic semantic class.  Disabling code
    injection is deterministic and avoids the ambiguous previous/next-neighbor
    mapping that would otherwise require hidden context.
    """

    return "always_off" if normalize_action(action) == POSE_ADJUST_ACTION else "normal"


def instruction_contract_text() -> str:
    """Prompt fragment shared by offline segmentation and future online planning."""

    return (
        f"Planner semantic contract ({PLANNER_SEMANTIC_CONTRACT_VERSION}):\n"
        "- Output a canonical action from the allowed action list.\n"
        "- Output one short, specific, single-line English instruction.\n"
        "- Do not output code_policy; it is derived by the runtime.\n"
        "- Do not include alternatives, explanations, frame numbers, or markdown."
    )
