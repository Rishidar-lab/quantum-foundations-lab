"""Sequential lesson runner (PHASE 6).

    python -m lessons.run --list
    python -m lessons.run
    python -m lessons.run --lesson 5 [--run] [--challenge] [--reveal]
    python -m lessons.run --next
    python -m lessons.run --reset
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

from lessons.content import LESSONS, Lesson, lesson_by_number

PROGRESS = pathlib.Path(__file__).resolve().parent.parent / ".lessons_progress.json"
_WRAP = textwrap.TextWrapper(width=88, initial_indent="    ", subsequent_indent="    ")


def _load_progress() -> dict:
    try:
        return json.loads(PROGRESS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"completed": [], "current": 1}


def _save_progress(p: dict) -> None:
    try:
        PROGRESS.write_text(json.dumps(p, indent=2))
    except OSError:
        pass


def _para(text: str) -> str:
    return "\n".join(_WRAP.fill(line) for line in text.split("\n"))


def show_list() -> None:
    prog = _load_progress()
    done = set(prog["completed"])
    print("Lessons  (F1 -> F24 -> T -> labs)\n")
    for les in LESSONS:
        mark = "x" if les.n in done else ("*" if les.n == prog["current"] else " ")
        print(f"  [{mark}] {les.n:>2}. [{les.node:<4}] {les.title}")
    print(f"\n  {len(done)}/{len(LESSONS)} complete. Current: lesson {prog['current']}.")
    print("  python -m lessons.run --lesson N   |   --next to advance")


def show_lesson(les: Lesson, run: bool, challenge: bool, reveal: bool) -> None:
    print(f"\n=== Lesson {les.n}: {les.title}   [{les.node}] ===\n")
    fields = [
        ("1. CONCEPT", les.concept),
        ("2. WHY IT EXISTS", les.why),
        ("3. MATHEMATICS", les.mathematics),
        ("4. TINY WORKED EXAMPLE", les.worked_example),
        ("5. CODE", les.code),
        ("6. PREDICTION (before you run anything)", les.prediction),
    ]
    for head, body in fields:
        print(head)
        print(_para(body) + "\n")

    print("7. ACTUAL RESULT")
    if run:
        try:
            print(_para(les.actual()))
        except Exception as exc:  # pragma: no cover - defensive
            print(_para(f"(check raised {type(exc).__name__}: {exc})"))
    else:
        print(_para("(re-run with --run to execute the check and see the real output)"))
    print()

    for head, body in (("8. INTERPRETATION", les.interpretation),
                       ("9. COMMON MISCONCEPTION", les.misconception)):
        print(head)
        print(_para(body) + "\n")

    if challenge or reveal:
        print("10. CHALLENGE  (yours to solve)")
        print(_para(les.challenge) + "\n")
        if reveal:
            print("   -- solution hint (you asked) --")
            print(_para(les.solution_hint) + "\n")
        else:
            print(_para("(work it out; then --reveal for a hint)") + "\n")
    else:
        print("10. CHALLENGE: run with --challenge to see it (solutions stay hidden).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--lesson", type=int, default=None)
    ap.add_argument("--run", action="store_true", help="execute the lesson's check (field 7)")
    ap.add_argument("--challenge", action="store_true", help="show field 10")
    ap.add_argument("--reveal", action="store_true", help="also show the challenge solution hint")
    ap.add_argument("--next", action="store_true", help="mark current lesson done and advance")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        _save_progress({"completed": [], "current": 1})
        print("progress reset.")
        return 0

    prog = _load_progress()

    if args.next:
        cur = prog["current"]
        if cur not in prog["completed"]:
            prog["completed"].append(cur)
        prog["current"] = min(cur + 1, len(LESSONS))
        _save_progress(prog)
        print(f"lesson {cur} marked complete. next: lesson {prog['current']}.")
        return 0

    if args.list:
        show_list()
        return 0

    n = args.lesson if args.lesson is not None else prog["current"]
    try:
        les = lesson_by_number(n)
    except KeyError:
        print(f"no lesson {n} (have 1..{len(LESSONS)})")
        return 1

    show_lesson(les, run=args.run, challenge=args.challenge, reveal=args.reveal)
    if args.lesson is None:
        print(f"\n(this is your current lesson. `--run` to execute it, `--next` when done.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
