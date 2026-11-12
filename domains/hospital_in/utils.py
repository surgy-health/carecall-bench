"""Paths and fixed constants for the hospital_in domain."""

from datetime import datetime
from pathlib import Path

DOMAIN_NAME = "hospital_in"

DOMAIN_DIR = Path(__file__).parent
DATA_DIR = DOMAIN_DIR / "data"
DB_PATH = DATA_DIR / "db.json"
SCHEMA_PATH = DATA_DIR / "schema.sql"
POLICY_PATH = DATA_DIR / "policy.md"
TASKS_PATH = DATA_DIR / "tasks.json"
# Tasks written by taskgen.py (split "generated"); the hand-written tasks stay in TASKS_PATH.
GENERATED_TASKS_PATH = DATA_DIR / "tasks_generated.json"

# The scenario clock. Every task runs "now", so tools never read the wall clock
# (tau2 replays write tools during grading and needs identical outputs).
NOW = datetime(2026, 11, 16, 9, 30)  # Monday, IST

# OPD calendar window in the seed: one week of history, two weeks ahead.
CALENDAR_START = datetime(2026, 11, 9)
CALENDAR_END = datetime(2026, 11, 28)  # inclusive
