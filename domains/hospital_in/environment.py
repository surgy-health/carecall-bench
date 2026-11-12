"""tau2 environment, tasks and registration for the hospital_in domain."""

import json
from functools import lru_cache
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment

from domains.hospital_in.data_model import HospitalDB
from domains.hospital_in.tools import HospitalTools
from domains.hospital_in.utils import (
    DB_PATH,
    DOMAIN_NAME,
    GENERATED_TASKS_PATH,
    POLICY_PATH,
    TASKS_PATH,
)

# Hand-written tasks first; a split name is looked up in each file's split_<stem>.json.
TASK_FILES = (TASKS_PATH, GENERATED_TASKS_PATH)


@lru_cache(maxsize=1)
def _seed_db() -> HospitalDB:
    return HospitalDB.load(str(DB_PATH))


@lru_cache(maxsize=1)
def _seed_json() -> str:
    return _seed_db().model_dump_json()


def get_environment(
    db: HospitalDB | None = None,
    solo_mode: bool = False,
    policy: str | None = None,
) -> Environment:
    if solo_mode:
        raise ValueError("Solo mode not supported for hospital_in")
    # A fresh seed copy is re-parsed from cached JSON: same DB (and hash) as a deep copy,
    # about 6x faster, which matters because grading builds three environments per run.
    db = HospitalDB.model_validate_json(_seed_json()) if db is None else db.model_copy(deep=True)
    if policy is None:
        policy = POLICY_PATH.read_text(encoding="utf-8")
    return Environment(domain_name=DOMAIN_NAME, policy=policy, tools=HospitalTools(db))


# Rules every simulated caller gets, appended to each task's own script by get_tasks (so
# text, cascade and realtime runs all see them; tasks.json keeps only the per-task script).
# Pilot2 (runs/text/pilot2-nova-lite): the caller said "Yes" to the agent's read-back and
# ended the call in the same message, before the agent made the change (reschedule-001-en,
# routing-004-en), so the change was never made and the run scored 0.
# gen1-sonnet46: callers accepted slots that broke their own constraints, and some spoke as
# the hospital, announcing a booking/appointment ID themselves (last two sentences).
CALLER_RULES = (
    "Never end the call in the same message where you answer a question or say yes. "
    "End the call (###STOP###) only after the agent confirms the change is done (e.g. gives "
    "an appointment ID or says help is arranged). Use only the names, numbers and details in "
    "your scenario; never invent doctors, IDs or numbers; give your UHID when asked if your "
    "scenario has one. If the agent says you are being transferred, reply only ###TRANSFER###. "
    "If your task instructions say when to end the call, end it then, but never in the same "
    "message where you answer a question or say yes. "
    "If the agent offers a slot, doctor or day that does not meet what your task asks for "
    "(time, day, language, department), politely decline and repeat what you need. "
    "Never speak as the hospital or announce a booking or appointment ID yourself."
)
CALLER_RULES_BLOCK = "\n\nRules for every call:\n" + CALLER_RULES


def with_caller_rules(script: str) -> str:
    """A task's caller script with CALLER_RULES appended once."""
    return script if script.endswith(CALLER_RULES_BLOCK) else script + CALLER_RULES_BLOCK


def _tasks_file_for_split(task_split_name: str | None) -> Path:
    """The task file whose split file defines `task_split_name` (hand-written tasks if None)."""
    if task_split_name is not None:
        for path in TASK_FILES:
            if path.exists() and task_split_name in get_tasks_split(path):
                return path
    return TASKS_PATH


def get_tasks(task_split_name: str | None = "base", tasks_path: Path | None = None) -> list[Task]:
    """Tasks in a split ('base', 'generated', ...); None = every task in the file.

    Without tasks_path, the file is the one that defines the split (None: the hand-written
    tasks.json), so tau2's registry and the runner reach generated tasks by split name.
    """
    if tasks_path is None:
        tasks_path = _tasks_file_for_split(task_split_name)
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    for t in raw:
        instructions = t["user_scenario"]["instructions"]
        instructions["task_instructions"] = with_caller_rules(instructions["task_instructions"])
    tasks = [Task.model_validate(t) for t in raw]
    if task_split_name is None:
        return tasks
    splits = get_tasks_split(tasks_path)
    if task_split_name not in splits:
        raise ValueError(f"Invalid task split {task_split_name!r}; valid: {sorted(splits)}")
    wanted = set(splits[task_split_name])
    return [t for t in tasks if t.id in wanted]


def get_tasks_split(tasks_path: Path | None = None) -> dict[str, list[str]]:
    """Splits of one task file, or (no path) of every task file that exists, merged."""
    if tasks_path is not None:
        split_file = tasks_path.parent / f"split_{tasks_path.stem}.json"
        return json.loads(split_file.read_text(encoding="utf-8"))
    merged: dict[str, list[str]] = {}
    for path in TASK_FILES:
        if not path.exists():
            continue
        for name, ids in get_tasks_split(path).items():
            if name in merged:
                raise ValueError(f"Split {name!r} is defined in more than one task file")
            merged[name] = ids
    return merged


def register() -> None:
    """Register hospital_in with tau2's global registry (idempotent).

    tau2 has no plugin hook, so anything that runs tau2 on this domain (our CLI
    wrapper, tests) must call this first.
    """
    from tau2.registry import registry

    if DOMAIN_NAME not in registry.get_domains():
        registry.register_domain(get_environment, DOMAIN_NAME)
    if DOMAIN_NAME not in registry.get_task_sets():
        registry.register_tasks(get_tasks, DOMAIN_NAME, get_task_splits=get_tasks_split)
