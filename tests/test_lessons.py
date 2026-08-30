"""PHASE 6 -- the learning content must be complete and its checks must run."""
from __future__ import annotations

import pytest

from lessons.content import LESSONS

VALID_NODES = {f"F{i}" for i in range(1, 25)} | {"T", "LAB"}
_TEXT_FIELDS = (
    "title", "concept", "why", "mathematics", "worked_example", "code",
    "prediction", "interpretation", "misconception", "challenge", "solution_hint",
)


def test_lessons_are_numbered_contiguously_from_one():
    assert [les.n for les in LESSONS] == list(range(1, len(LESSONS) + 1))


@pytest.mark.parametrize("les", LESSONS, ids=lambda l: f"L{l.n}")
def test_every_field_is_present_and_nontrivial(les):
    assert les.node in VALID_NODES
    for field in _TEXT_FIELDS:
        value = getattr(les, field)
        assert isinstance(value, str) and len(value.strip()) >= 12, (les.n, field)


@pytest.mark.parametrize("les", LESSONS, ids=lambda l: f"L{l.n}")
def test_actual_result_callable_runs(les):
    out = les.actual()
    assert isinstance(out, str) and out.strip()


def test_progression_covers_the_spine():
    nodes = {les.node for les in LESSONS}
    # the load-bearing nodes must all appear as a lesson's primary node
    for required in ("F1", "F9", "F11", "F12", "F13", "F14", "F15", "F16",
                     "F18", "F19", "F20", "F21", "F24", "T", "LAB"):
        assert required in nodes, required


def test_challenge_and_solution_are_distinct():
    for les in LESSONS:
        assert les.challenge.strip() != les.solution_hint.strip()
