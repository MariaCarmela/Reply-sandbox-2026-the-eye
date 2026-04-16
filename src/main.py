
"""
main.py — Entry point for The Eye multi-agent training system.

Accepts a CLI argument to run one or all training levels, sharing a
single session ID across every level executed in the same invocation.
"""

import sys
import os
import logging
from dotenv import load_dotenv
from agents.llm_the_eye import TheEye, _generate_ulid


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_LEVELS: tuple[str, ...] = ("train1", "train2", "train3", "all")
BANNER_WIDTH: int = 60


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _banner(text: str, char: str = "=") -> str:
    """Return a centred banner string surrounded by a border of *char*."""
    border = char * BANNER_WIDTH
    return f"\n{border}\n{text.center(BANNER_WIDTH)}\n{border}"


def _print_session_id_block(session_id: str, label: str = "Session ID") -> None:
    """Print the session ID prominently so it is easy to spot in logs."""
    print(f"\n  {'─' * (BANNER_WIDTH - 4)}")
    print(f"  {label} : {session_id}")
    print(f"  {'─' * (BANNER_WIDTH - 4)}\n")


def _print_usage() -> None:
    """Print usage instructions to stdout."""
    print(
        "\nUsage:\n"
        "    python main.py <mode>\n\n"
        "Modes:\n"
        "    train1  — run training level 1 only\n"
        "    train2  — run training level 2 only\n"
        "    train3  — run training level 3 only\n"
        "    all     — run all three levels in sequence\n\n"
        "Example:\n"
        "    python main.py train2\n"
        "    python main.py all\n"
    )


def _print_summary_table(results: dict) -> None:
    """Print a formatted summary table of results per level."""
    col_level: int = 10
    col_status: int = 12
    col_detail: int = 36
    divider: str = "─" * (col_level + col_status + col_detail)

    header = (
        f"{'Level':<{col_level}}"
        f"{'Status':<{col_status}}"
        f"{'Detail':<{col_detail}}"
    )

    print(f"\n{divider}")
    print(header)
    print(divider)

    for level, outcome in sorted(results.items()):
        passed = outcome.get("status") == "OK"
        status_str = "✅ PASSED" if passed else "❌ FAILED"
        detail_str = str(outcome.get("error") or "completed")[:col_detail]
        print(
            f"{'train' + str(level):<{col_level}}"
            f"{status_str:<{col_status}}"
            f"{detail_str:<{col_detail}}"
        )

    print(divider)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_level(level_int: int, session_id: str) -> dict:
    """
    Instantiate TheEye for *level_int*, run it, and return a result dict.

    Returns
    -------
    dict
        ``{"status": "OK",     "error": None}``     on success, or
        ``{"status": "FAILED", "error": str(exc)}``  on failure.
    """
    try:
        logger.info("Starting level %d  (session=%s)", level_int, session_id)
        eye = TheEye(level=level_int)
        eye.run()
        logger.info("Level %d completed successfully.", level_int)
        return {"status": "OK", "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.error("Level %d raised an exception: %s", level_int, exc)
        return {"status": "FAILED", "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run requested levels, print summary, set exit code."""
    load_dotenv()

    # ── Argument validation ──────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("  ❌  No mode argument supplied.")
        _print_usage()
        sys.exit(1)

    mode: str = sys.argv[1].strip().lower()

    if mode not in VALID_LEVELS:
        print(f"  ❌  Unknown mode: '{mode}'")
        _print_usage()
        sys.exit(1)

    # ── Level map ────────────────────────────────────────────────────────────
    LEVEL_MAP: dict[str, list[int]] = {
        "train1": [1],
        "train2": [2],
        "train3": [3],
        "all":    [1, 2, 3],
    }
    levels: list[int] = LEVEL_MAP[mode]

    # ── Session ID ───────────────────────────────────────────────────────────
    session_id: str = _generate_ulid()

    # ── Banner ───────────────────────────────────────────────────────────────
    print(_banner("🔭  THE EYE  —  MULTI-AGENT SYSTEM"))
    _print_session_id_block(session_id)
    print(f"  Mode   : {mode}")
    print(f"  Levels : {levels}")

    # ── Run levels ───────────────────────────────────────────────────────────
    results: dict[int, dict] = {}

    for level in levels:
        divider = "─" * BANNER_WIDTH
        print(f"\n{divider}")
        print(f"  Running  train{level}  (level={level})")
        print(divider)

        results[level] = run_level(level, session_id)

        if results[level]["status"] == "OK":
            print(f"  ✅  Level {level} finished successfully.")
        else:
            print(f"  ❌  Level {level} failed: {results[level]['error']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_banner("SUMMARY"))
    _print_summary_table(results)
    _print_session_id_block(session_id, label="Session ID")

    # ── Exit code ─────────────────────────────────────────────────────────────
    any_failed: bool = any(r["status"] != "OK" for r in results.values())

    if any_failed:
        print("  ⚠️   One or more levels FAILED. Exiting with code 1.\n")
        sys.exit(1)

    print("  🎉  All levels completed successfully. Exiting with code 0.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
