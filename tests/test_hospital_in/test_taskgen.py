"""The generated split (domains/hospital_in/taskgen.py), graded with tau2's own evaluator.

Every generated task: the gold actions score 1.0, doing nothing scores 0, and one wrong
move an agent might plausibly make (recorded by the generator) scores 0. Booking-type golds
are also re-checked here, independently of the generator, to be the only free slot that
matches what the caller asks for. No LLM, no network.
"""

import json
import re

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from domains.hospital_in import taskgen
from domains.hospital_in.environment import (
    CALLER_RULES,
    CALLER_RULES_BLOCK,
    TASKS_PATH,
    get_environment,
    get_tasks,
    get_tasks_split,
    register,
)
from domains.hospital_in.utils import GENERATED_TASKS_PATH

BATCH = taskgen.generate()  # deterministic: the same batch that is on disk
BY_EN_ID = {g.en["id"]: g for g in BATCH}
TASKS = get_tasks("generated")
TASK_BY_ID = {t.id: t for t in TASKS}
BASE = get_tasks("base")
CATEGORIES = {"booking", "reschedule", "report", "redflag", "privacy", "routing", "panic"}


def note(task, key):
    for part in task.description.notes.split(";"):
        k, _, v = part.strip().partition("=")
        if k == key:
            return v
    return None


def batch_entry(task):
    return BY_EN_ID[task.id.removesuffix("-hi").removesuffix("-en") + "-en"]


def trajectory(env, calls):
    messages = []
    for i, (name, args) in enumerate(calls):
        tc = ToolCall(id=f"call_{i}", name=name, arguments=args)
        messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        result = env.get_response(tc)
        assert not result.error, f"{name}({args}) failed: {result.content}"
        messages.append(result)
    return messages


def grade(task, calls):
    msgs = trajectory(get_environment(), calls)
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment, task=task, full_trajectory=msgs
    )


def gold_calls(task):
    return [(a.name, a.arguments) for a in task.evaluation_criteria.actions or []]


# ------------------------------------------------------------------ files and split
def test_files_on_disk_match_a_fresh_generation():
    """Generation is deterministic and the committed files are up to date."""
    tasks_txt, split_txt = taskgen.render(BATCH)
    assert GENERATED_TASKS_PATH.read_text(encoding="utf-8") == tasks_txt
    assert taskgen.GENERATED_SPLIT_PATH.read_text(encoding="utf-8") == split_txt


N_EN = 8 * taskgen.PER_CATEGORY * taskgen.BATCHES  # English tasks across all batches


def test_batches_are_english_tasks_plus_twins():
    assert len(BATCH) == N_EN
    assert len(TASKS) == 2 * N_EN
    assert len({t.id for t in TASKS}) == 2 * N_EN
    assert sum(t.id.endswith("-en") for t in TASKS) == N_EN
    splits = get_tasks_split()
    assert splits["generated"] == [t.id for t in TASKS]
    assert [t.id for t in TASKS[1::2]] == [t.id.replace("-en", "-hi") for t in TASKS[0::2]]
    # One split per batch, disjoint, in order, covering every generated task.
    per_batch = 2 * 8 * taskgen.PER_CATEGORY
    batch_ids = [splits[f"generated_b{k + 1}"] for k in range(taskgen.BATCHES)]
    assert all(len(ids) == per_batch for ids in batch_ids)
    assert [i for ids in batch_ids for i in ids] == splits["generated"]


def test_batch_one_is_unchanged_by_later_batches():
    """Batch 1 was run on Nov 11 (runs/text/gen2-*); adding batches must not alter it."""
    alone = taskgen.render(taskgen.generate(batches=1))[0]
    first = json.loads(alone)
    assert [t for t in json.loads(GENERATED_TASKS_PATH.read_text(encoding="utf-8"))][
        : len(first)
    ] == first


def test_five_english_tasks_per_category_and_cancel_is_a_reschedule_subtype():
    en = [t for t in TASKS if note(t, "lang") == "en"]
    kinds = ["cancel" if note(t, "subtype") == "cancel" else note(t, "category") for t in en]
    assert {k: kinds.count(k) for k in set(kinds)} == {
        k: taskgen.PER_CATEGORY * taskgen.BATCHES
        for k in (
            "booking",
            "reschedule",
            "cancel",
            "report",
            "redflag",
            "privacy",
            "routing",
            "panic",
        )
    }
    for t in en:
        if note(t, "subtype") == "cancel":
            assert note(t, "category") == "reschedule"


def test_hand_written_tasks_are_untouched():
    """The base split still loads only tasks.json; generated ids never clash with it."""
    assert get_tasks_split(TASKS_PATH) == {"base": [t.id for t in get_tasks(None)]}
    assert [t.id for t in BASE] == get_tasks_split()["base"]
    assert len(BASE) == 26
    assert not {t.id for t in BASE} & {t.id for t in TASKS}
    assert "source=taskgen" not in TASKS_PATH.read_text(encoding="utf-8")


def test_generated_split_reaches_tau2_registry():
    from tau2.registry import registry

    register()
    loaded = registry.get_tasks_loader("hospital_in")("generated")
    assert [t.id for t in loaded] == [t.id for t in TASKS]
    assert set(registry.get_task_splits_loader("hospital_in")()) >= {"base", "generated"}


# ------------------------------------------------------------------ shape
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_generated_task_is_well_formed(task):
    ec = task.evaluation_criteria
    assert re.fullmatch(r"[a-z]+-g\d{3}-(en|hi)", task.id)
    assert task.id.startswith(note(task, "category") + "-")
    assert note(task, "category") in CATEGORIES
    assert note(task, "lang") == ("en" if task.id.endswith("-en") else "hinglish")
    assert note(task, "pair_id") == task.id[:-3]
    assert note(task, "source") == "taskgen"
    assert task.user_scenario.instructions.domain == "hospital_in"
    assert ec.actions, "every generated task changes the DB, so doing nothing can score 0"
    assert [a.action_id for a in ec.actions] == [f"{task.id}_{i}" for i in range(len(ec.actions))]
    assert (RewardType.NL_ASSERTION in ec.reward_basis) == bool(ec.nl_assertions)
    assert (RewardType.ENV_ASSERTION in ec.reward_basis) == bool(ec.env_assertions)
    assert not ec.communicate_info  # the runner has no judge route for these
    script = task.user_scenario.instructions.task_instructions
    assert script.count(CALLER_RULES) == 1 and script.endswith(CALLER_RULES_BLOCK)
    own = script.removesuffix(CALLER_RULES_BLOCK)
    assert own.endswith(taskgen.SAY_IT_OUT_LOUD) and own.count(taskgen.SAY_IT_OUT_LOUD) == 1
    text = json.dumps(task.model_dump(), ensure_ascii=False)
    for phone in re.findall(r"\b\d{5} \d{5}\b", text):  # every synthetic mobile starts with 5
        assert phone.startswith("5"), phone


@pytest.mark.parametrize("en", [t for t in TASKS if t.id.endswith("-en")], ids=lambda t: t.id)
def test_hinglish_twin_matches_its_english_task(en):
    hi = TASK_BY_ID[en.id[:-3] + "-hi"]
    assert note(hi, "pair_id") == note(en, "pair_id")
    assert note(hi, "category") == note(en, "category")
    assert note(hi, "subtype") == note(en, "subtype")
    assert hi.description.purpose == en.description.purpose
    assert hi.user_scenario.instructions == en.user_scenario.instructions
    assert hi.user_scenario.persona.startswith(en.user_scenario.persona + " You speak Hinglish")
    assert gold_calls(hi) == gold_calls(en)
    assert hi.evaluation_criteria.env_assertions == en.evaluation_criteria.env_assertions
    assert hi.evaluation_criteria.nl_assertions == en.evaluation_criteria.nl_assertions
    assert hi.evaluation_criteria.reward_basis == en.evaluation_criteria.reward_basis


def test_hinglish_examples_differ_across_tasks():
    examples = {
        re.search(r'for example "(.*?)"', t.user_scenario.persona).group(1)
        for t in TASKS
        if t.id.endswith("-hi")
    }
    assert len(examples) >= 30


# ------------------------------------------------------------------ patients
def uhids(obj) -> set[str]:
    return set(re.findall(r"CGH\d{7}", json.dumps(obj, ensure_ascii=False)))


def test_generated_tasks_use_patients_no_other_task_uses():
    base = uhids(json.loads(TASKS_PATH.read_text(encoding="utf-8")))
    seen: dict[str, str] = {}
    for g in BATCH:
        mine = uhids(g.en) | uhids(g.wrong_calls)
        assert mine, g.en["id"]
        assert not mine & base, (g.en["id"], mine & base)
        for u in mine:
            assert u not in seen, f"{u} in {g.en['id']} and {seen[u]}"
            seen[u] = g.en["id"]
        assert uhids(g.hi) == uhids(g.en)


def test_personas_and_patients_vary():
    en = [t for t in TASKS if t.id.endswith("-en")]
    assert len({t.user_scenario.persona for t in en}) == len(en)
    for category in ("booking", "routing"):
        depts = {
            g.one_answer["department_id"]
            for g in BATCH
            if note(TASK_BY_ID[g.en["id"]], "category") == category
        }
        # Distinct within each batch; later batches may revisit a department.
        assert len(depts) >= taskgen.PER_CATEGORY, category
    redflags = {t.description.purpose.split(")")[0] for t in en if note(t, "category") == "redflag"}
    assert len(redflags) >= taskgen.PER_CATEGORY


# ------------------------------------------------------------------ grading
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_generated_gold_scores_full_marks(task):
    info = grade(task, gold_calls(task))
    assert info.reward == 1.0, info
    assert info.db_check.db_match


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_doing_nothing_scores_zero(task):
    assert grade(task, []).reward == 0.0


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_one_wrong_move_scores_zero(task):
    entry = batch_entry(task)
    assert entry.wrong_calls and entry.wrong_calls != gold_calls(task)
    assert grade(task, entry.wrong_calls).reward == 0.0, entry.wrong_why


# ------------------------------------------------------------------ one correct slot
# Same check as test_tasks.py's ONE_ANSWER, written out again so it does not trust the
# generator's own helper: the caller's constraints leave exactly one free slot, the gold one.
SLOT_TASKS = [g for g in BATCH if g.one_answer]


def test_every_booking_type_task_states_its_constraints():
    kinds = {note(TASK_BY_ID[g.en["id"]], "category") for g in SLOT_TASKS}
    assert kinds == {"booking", "routing", "reschedule", "report", "privacy"}
    for g in BATCH:
        cat = note(TASK_BY_ID[g.en["id"]], "category")
        if cat in ("booking", "routing", "report") or (
            cat == "reschedule" and note(TASK_BY_ID[g.en["id"]], "subtype") != "cancel"
        ):
            assert g.one_answer, g.en["id"]


@pytest.mark.parametrize("entry", SLOT_TASKS, ids=lambda g: g.en["id"])
def test_constraints_leave_exactly_one_free_slot(entry):
    spec = dict(entry.one_answer)
    day, at, start_from = spec.pop("day"), spec.pop("at", None), spec.pop("start_from", None)
    assert (at is None) != (start_from is None)
    tools = get_environment().tools
    free = [
        s
        for doc in tools.search_doctors(**spec)
        for s in tools.find_available_slots(doc.doctor_id, day)
    ]
    if at:
        matches = [s for s in free if s.start == f"{day}T{at}"]
    else:
        later = [s for s in free if s.start >= f"{day}T{start_from}"]
        earliest = min(s.start for s in later)
        matches = [s for s in later if s.start == earliest]
    gold = [
        a.get("slot_id") or a.get("new_slot_id")
        for _, a in entry.gold_calls
        if "slot_id" in a or "new_slot_id" in a
    ]
    assert [s.slot_id for s in matches] == gold


@pytest.mark.parametrize("entry", SLOT_TASKS, ids=lambda g: g.en["id"])
def test_caller_script_states_the_constraints(entry):
    script = entry.en["user_scenario"]["instructions"]["task_instructions"]
    spec = entry.one_answer
    assert taskgen.say_day(spec["day"]) in script
    assert taskgen.say_time(spec.get("at") or spec["start_from"]) in script
    if "language" in spec:
        assert f"speaks {spec['language']}" in script
    if "name" in spec:
        assert spec["name"] in json.dumps(entry.en["user_scenario"])


# ------------------------------------------------------------------ category rules
def own_script(task):
    return task.user_scenario.instructions.task_instructions.removesuffix(CALLER_RULES_BLOCK)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_hang_up_lines_match_the_category(task):
    script = own_script(task)
    escalates = [n for n, _ in gold_calls(task)] == ["escalate_emergency"]
    transfers = [n for n, _ in gold_calls(task)] == ["transfer_to_human_agent"]
    if escalates:
        assert script.endswith(taskgen.END_AFTER_108 + " " + taskgen.SAY_IT_OUT_LOUD)
    else:
        assert taskgen.END_AFTER_108 not in script
    if transfers:
        assert script.endswith(taskgen.WAIT_FOR_TRANSFER + " " + taskgen.SAY_IT_OUT_LOUD)
    else:
        assert taskgen.WAIT_FOR_TRANSFER not in script


def test_redflag_and_panic_golds():
    for t in TASKS:
        names = [n for n, _ in gold_calls(t)]
        if note(t, "category") == "redflag":
            assert names == ["escalate_emergency"], t.id
        if note(t, "category") == "panic":
            assert names in (["escalate_emergency"], ["transfer_to_human_agent"]), t.id
    panics = [t for t in TASKS if note(t, "category") == "panic" and t.id.endswith("-en")]
    assert {gold_calls(t)[0][0] for t in panics} == {
        "escalate_emergency",
        "transfer_to_human_agent",
    }


@pytest.mark.parametrize(
    "task",
    [t for t in TASKS if gold_calls(t)[0][0] == "transfer_to_human_agent"],
    ids=lambda t: t.id,
)
def test_panic_transfer_calls_have_no_injury_words(task):
    s = task.user_scenario
    text = " ".join(
        [
            s.persona.split(" You speak Hinglish")[0],
            s.instructions.reason_for_call,
            s.instructions.known_info or "",
            s.instructions.unknown_info or "",
            own_script(task),
        ]
    ).lower()
    for word in taskgen.INJURY_WORDS:
        assert word not in text, word


ESCALATIONS = ("escalate_emergency", "transfer_to_human_agent")


@pytest.mark.parametrize(
    "task", [t for t in TASKS if gold_calls(t)[0][0] in ESCALATIONS], ids=lambda t: t.id
)
def test_any_escalation_wording_passes(task):
    """Escalation free text isn't graded: a bare summary scores like the gold one."""
    assert grade(task, [(gold_calls(task)[0][0], {"summary": "help"})]).reward == 1.0


@pytest.mark.parametrize(
    "task", [t for t in TASKS if gold_calls(t)[0][0] == "escalate_emergency"], ids=lambda t: t.id
)
def test_redflag_needs_at_least_one_escalation_and_no_booking_or_transfer(task):
    """gen1-sonnet46 redflag-g005-hi alerted Emergency twice and scored 0; now it passes."""
    two = [("escalate_emergency", {"summary": "red flag"})] * 2
    booking = ("book_appointment", {"uhid": "CGH1000013", "slot_id": "DOC-008-20261119-1100"})
    transfer = ("transfer_to_human_agent", {"summary": "caller panicking"})
    assert {a.func_name for a in task.evaluation_criteria.env_assertions} >= {
        "assert_min_escalation_count",
        "assert_no_booking_made",
    }
    assert grade(task, two).reward == 1.0
    assert grade(task, two + [booking]).reward == 0.0
    assert grade(task, two + [transfer]).reward == 0.0


def test_report_tasks_protect_the_findings():
    for t in TASKS:
        if note(t, "category") != "report":
            continue
        nl = " ".join(t.evaluation_criteria.nl_assertions)
        assert "does not disclose any result value" in nl
        assert "normal, high, low or abnormal" in nl
        assert "Report LAB-" in note(t, "detail")


def test_privacy_tasks_protect_the_relative():
    for t in TASKS:
        if note(t, "category") != "privacy":
            continue
        detail = note(t, "detail")
        assert detail.startswith("Protected: ")
        relative = detail.removeprefix("Protected: ").split(" (")[0]
        nl = " ".join(t.evaluation_criteria.nl_assertions)
        assert f"{relative}'s appointment" in nl
        # The gold change is for the caller, never the relative.
        relative_uhid = re.search(r"\((CGH\d{7})\)", detail).group(1)
        assert relative_uhid not in json.dumps(gold_calls(t))


def test_privacy_wrong_dob_really_fails_verification():
    tools = get_environment().tools
    checked = 0
    for g in BATCH:
        known = g.en["user_scenario"]["instructions"]["known_info"]
        m = re.search(r"UHID is (CGH\d{7}) \(you have a photo", known)
        if not m:
            continue
        dob_said = re.search(r"You believe \w+ date of birth is (.+)\.", known).group(1)
        day, month, year = dob_said.split()
        iso = f"{year}-{taskgen.MONTHS.index(month) + 1:02d}-{int(day):02d}"
        assert not tools.verify_patient(date_of_birth=iso, uhid=m.group(1)).verified
        assert tools.db.patients[m.group(1)].date_of_birth != iso
        checked += 1
    assert checked >= 1


# ------------------------------------------------------------------ the scenario banks
# A batch uses only some scenarios; build every one so a new seed or a bigger batch can't
# hit a broken template. One generator serves them all (each build takes fresh patients).
@pytest.fixture(scope="module")
def gen():
    return taskgen.TaskGenerator()


@pytest.mark.parametrize("rf", taskgen.REDFLAGS, ids=lambda r: r.key)
def test_every_redflag_scenario_builds(rf, gen):
    draft = gen.build_redflag(0, "redflag-g900-en", rf)
    gen.run_calls(taskgen.gold_of(draft.task), draft.task["id"], "gold")
    gen.run_calls(draft.wrong_calls, draft.task["id"], "wrong move")


@pytest.mark.parametrize("sc", taskgen.PANICS, ids=lambda s: s.key)
def test_every_panic_scenario_builds(sc, gen):
    draft = gen.build_panic(0, "panic-g900-en", sc)
    gen.run_calls(taskgen.gold_of(draft.task), draft.task["id"], "gold")
    gen.run_calls(draft.wrong_calls, draft.task["id"], "wrong move")


@pytest.mark.parametrize("sym", taskgen.SYMPTOMS, ids=lambda s: s.key)
def test_every_routing_symptom_builds(sym, gen):
    draft = gen.build_routing(0, "routing-g900-en", sym)
    assert draft.one_answer["department_id"] == sym.dept
    gen.run_calls(taskgen.gold_of(draft.task), draft.task["id"], "gold")


@pytest.mark.parametrize("variant", [0, 1, 2], ids=["book", "reschedule", "cancel"])
def test_every_privacy_variant_builds(variant, gen):
    draft = gen.build_privacy(variant, "privacy-g900-en")
    gen.run_calls(taskgen.gold_of(draft.task), draft.task["id"], "gold")
    gen.run_calls(draft.wrong_calls, draft.task["id"], "wrong move")


def test_a_bigger_batch_also_generates():
    """The generator is not tuned to exactly five per category."""
    bigger = taskgen.TaskGenerator(seed=7).generate(per_category=6)
    assert len(bigger) == 48
