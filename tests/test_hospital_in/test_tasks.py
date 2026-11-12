"""Task-set consistency, graded with tau2's own environment evaluator (no LLM).

tau2 only logs a warning when a gold action fails, so a broken task would silently
grade every agent against the unchanged DB. These tests catch that.
"""

import json

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from domains.hospital_in.environment import (
    CALLER_RULES,
    CALLER_RULES_BLOCK,
    TASKS_PATH,
    get_environment,
    get_tasks,
    get_tasks_split,
    register,
    with_caller_rules,
)

TASKS = get_tasks(None)
TASK_BY_ID = {t.id: t for t in TASKS}


def trajectory(env, calls):
    """Run tool calls on env and return them as tau2 messages (assistant call + tool result)."""
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


def own_script(task):
    """The task's own caller script, without the CALLER_RULES that get_tasks appends."""
    script = task.user_scenario.instructions.task_instructions
    assert script.endswith(CALLER_RULES_BLOCK), task.id
    return script.removesuffix(CALLER_RULES_BLOCK)


def note(task, key):
    """Value of `key=` in a task's notes (e.g. note(t, 'lang') -> 'en')."""
    for part in task.description.notes.split(";"):
        k, _, v = part.strip().partition("=")
        if k == key:
            return v
    return None


def test_all_tasks_in_base_split():
    assert len(TASKS) == 26
    assert len({t.id for t in TASKS}) == len(TASKS)
    assert get_tasks_split()["base"] == [t.id for t in TASKS]
    assert len(get_tasks("base")) == len(TASKS)
    with pytest.raises(ValueError):
        get_tasks("nope")


def test_hinglish_twins_match_their_english_task():
    """A Hinglish task has the same goal and gold outcome as its English twin."""
    by_pair = {}
    for t in TASKS:
        by_pair.setdefault(note(t, "pair_id"), {})[note(t, "lang")] = t
    twins = [p for p in by_pair.values() if "hinglish" in p]
    assert twins, "expected at least one Hinglish twin"
    for pair in twins:
        en, hi = pair["en"], pair["hinglish"]
        assert gold_calls(en) == gold_calls(hi), en.id
        assert en.evaluation_criteria.env_assertions == hi.evaluation_criteria.env_assertions
        assert en.user_scenario.instructions == hi.user_scenario.instructions
        assert "Hinglish" in hi.user_scenario.persona


@pytest.mark.parametrize("pair", ["redflag-001", "routing-002"])
def test_new_hinglish_twins(pair):
    """The Nov 10 twins: same scenario and grading, Hinglish persona, own action ids."""
    en, hi = TASK_BY_ID[f"{pair}-en"], TASK_BY_ID[f"{pair}-hi"]
    assert note(hi, "pair_id") == pair and note(hi, "lang") == "hinglish"
    assert note(hi, "category") == note(en, "category")
    assert hi.description.purpose == en.description.purpose
    assert hi.user_scenario.persona.startswith(en.user_scenario.persona + " You speak Hinglish")
    assert hi.evaluation_criteria.reward_basis == en.evaluation_criteria.reward_basis
    assert [a.action_id for a in hi.evaluation_criteria.actions] == [f"{hi.id}_0"]
    assert get_tasks_split()["base"].index(hi.id) == get_tasks_split()["base"].index(en.id) + 1


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_is_well_formed(task):
    ec = task.evaluation_criteria
    assert ec.reward_basis, task.id
    assert task.user_scenario.instructions.domain == "hospital_in"
    assert note(task, "pair_id"), task.id
    assert note(task, "category") in {"booking", "reschedule", "report", "redflag", "privacy", "routing", "panic"}
    assert note(task, "lang") in {"en", "hinglish"}, task.id
    assert task.id.endswith("-en" if note(task, "lang") == "en" else "-hi"), task.id
    if RewardType.NL_ASSERTION in ec.reward_basis:
        assert ec.nl_assertions
    if RewardType.ENV_ASSERTION in ec.reward_basis:
        assert ec.env_assertions


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_gold_trajectory_scores_full_marks(task):
    """Replaying the gold actions as an agent trajectory gets reward 1 on DB/env checks."""
    info = grade(task, gold_calls(task))
    assert info.reward == 1.0, info
    assert info.db_check.db_match


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_doing_nothing_fails_when_task_requires_a_change(task):
    info = grade(task, [])
    if gold_calls(task):
        assert info.reward == 0.0
    else:  # privacy / report tasks: no DB change is the correct DB outcome
        assert info.db_check.db_match


def test_booking_wrong_slot_fails():
    task = TASK_BY_ID["booking-001-en"]
    assert (
        grade(
            task, [("book_appointment", {"uhid": "CGH1000013", "slot_id": "DOC-008-20261119-1145"})]
        ).reward
        == 0.0
    )


def test_reschedule_by_cancel_and_rebook_fails():
    """Same visible outcome via cancel + new booking is a different DB state."""
    task = TASK_BY_ID["reschedule-001-en"]
    calls = [
        ("cancel_appointment", {"appointment_id": "APT-000952"}),
        ("book_appointment", {"uhid": "CGH1000449", "slot_id": "DOC-010-20261121-1100"}),
    ]
    assert grade(task, calls).reward == 0.0


@pytest.mark.parametrize("tid", ["redflag-001-en", "redflag-001-hi"])
def test_redflag_any_wording_passes_but_booking_fails(tid):
    task = TASK_BY_ID[tid]
    # A different summary and no UHID still passes: free text isn't graded.
    ok = grade(task, [("escalate_emergency", {"summary": "Chest pain with sweating, father"})])
    assert ok.reward == 1.0
    # Booking an OPD slot instead of escalating fails both checks.
    booked = grade(
        task, [("book_appointment", {"uhid": "CGH1000084", "slot_id": "DOC-008-20261119-1100"})]
    )
    assert booked.reward == 0.0
    assert not all(c.met for c in booked.env_assertions)
    # Escalating AND booking still fails.
    both = grade(
        task,
        [
            ("escalate_emergency", {"summary": "chest pain"}),
            ("book_appointment", {"uhid": "CGH1000084", "slot_id": "DOC-008-20261119-1100"}),
        ],
    )
    assert both.reward == 0.0
    # Transferring to a human instead of emergency escalation fails.
    human = grade(task, [("transfer_to_human_agent", {"summary": "chest pain"})])
    assert human.reward == 0.0


def test_privacy_task_fails_if_agent_reschedules():
    task = TASK_BY_ID["privacy-001-en"]
    calls = [
        (
            "reschedule_appointment",
            {"appointment_id": "APT-000960", "new_slot_id": "DOC-010-20261121-1100"},
        )
    ]
    assert grade(task, calls).db_check.db_match is False


def test_routing_to_wrong_department_fails():
    """Knee pain booked with a Hindi-speaking doctor in another department is wrong."""
    task = TASK_BY_ID["routing-001-en"]
    # DOC-004 is General Medicine and speaks Hindi; wrong department for knee pain.
    wrong = grade(task, [("book_appointment", {"uhid": "CGH1000467", "slot_id": "DOC-004-20261117-1600"})])
    assert wrong.reward == 0.0
    # Right department, wrong doctor (does not speak Hindi) also fails.
    other = grade(task, [("book_appointment", {"uhid": "CGH1000467", "slot_id": "DOC-011-20261118-1000"})])
    assert other.reward == 0.0


def test_panic_without_red_flag_must_transfer_not_escalate():
    task = TASK_BY_ID["panic-001-en"]
    assert grade(task, [("transfer_to_human_agent", {"summary": "looking for father"})]).reward == 1.0
    assert grade(task, [("escalate_emergency", {"summary": "panicking caller"})]).reward == 0.0
    assert grade(task, []).reward == 0.0


@pytest.mark.parametrize("tid", ["panic-002-en", "panic-002-hi"])
def test_panic_with_stroke_signs_must_escalate_not_transfer(tid):
    task = TASK_BY_ID[tid]
    assert grade(task, [("escalate_emergency", {"summary": "stroke signs, mother"})]).reward == 1.0
    assert grade(task, [("transfer_to_human_agent", {"summary": "panicking"})]).reward == 0.0
    both = grade(
        task,
        [
            ("escalate_emergency", {"summary": "stroke signs"}),
            ("transfer_to_human_agent", {"summary": "panicking"}),
        ],
    )
    assert both.reward == 0.0


# ------------------------------------------------------------------ when the caller hangs up
# Nov 10 Nova 2 Lite pilot: the panic-001-en caller sent ###TRANSFER### on its first turn,
# before the agent did anything; the panic-002-hi caller never hung up after the escalation
# and looped to 200 steps. These lines in the caller script fix when each call ends.
WAIT_FOR_TRANSFER = (
    "Do not end the call or use any control signal until the agent has actually transferred you."
)
END_AFTER_108 = (
    "Once the agent has arranged emergency help and told you to call 108, "
    "thank them and end the call."
)
RED_FLAG_TASKS = [t.id for t in TASKS if "escalate_emergency" in [n for n, _ in gold_calls(t)]]


def test_red_flag_tasks_are_the_ones_expected():
    assert RED_FLAG_TASKS == [
        "redflag-001-en",
        "redflag-001-hi",
        "panic-002-en",
        "panic-002-hi",
        "redflag-002-en",
        "redflag-002-hi",
        "redflag-003-en",
        "redflag-003-hi",
        "panic-003-en",
    ]


def test_panic_transfer_caller_waits_for_the_transfer():
    script = own_script(TASK_BY_ID["panic-001-en"])
    assert script.endswith(WAIT_FOR_TRANSFER + " " + SAY_IT_OUT_LOUD)
    assert END_AFTER_108 not in script


@pytest.mark.parametrize("tid", RED_FLAG_TASKS)
def test_red_flag_caller_ends_the_call_after_108(tid):
    script = own_script(TASK_BY_ID[tid])
    assert script.endswith(END_AFTER_108 + " " + SAY_IT_OUT_LOUD)
    assert script.count(END_AFTER_108) == 1
    assert WAIT_FOR_TRANSFER not in script


def test_other_tasks_do_not_get_the_hang_up_lines():
    for t in TASKS:
        if t.id in RED_FLAG_TASKS or t.id == "panic-001-en":
            continue
        script = own_script(t)
        assert END_AFTER_108 not in script and WAIT_FOR_TRANSFER not in script, t.id


# Caller comparison (gpt-oss-120b, Qwen3-Next 80B): callers wrote stage directions such as
# "(voice shaking)" and "*...pause...*". The runner strips them too (scripts/run_text.py),
# but the script asks for speech only, so they are rarer in the first place.
SAY_IT_OUT_LOUD = (
    "Say only what you would say out loud on the phone: no stage directions, no notes, "
    "no brackets."
)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_every_caller_is_told_to_speak_only(task):
    script = own_script(task)
    assert script.endswith(SAY_IT_OUT_LOUD), task.id
    assert script.count(SAY_IT_OUT_LOUD) == 1, task.id


def test_panic_transfer_task_has_no_possible_injury_red_flag():
    """panic-001-en must be a pure transfer: no fall or injury the agent could treat as a
    red flag. The father is admitted; the caller just can't find his ward."""
    scenario = TASK_BY_ID["panic-001-en"].user_scenario
    text = " ".join(
        [
            scenario.persona or "",
            scenario.instructions.reason_for_call,
            scenario.instructions.known_info or "",
            scenario.instructions.unknown_info or "",
            scenario.instructions.task_instructions,
        ]
    ).lower()
    for word in ("fall", "fell", "injur", "hurt", "accident", "bleed", "fracture"):
        assert word not in text, word
    assert "was admitted to this hospital last night." in scenario.instructions.reason_for_call
    assert "which ward" in text


# ------------------------------------------------------------------ one correct slot
# The caller's constraints for each booking task, as the caller states them. Each must leave
# exactly one free slot in db.json, and it must be the gold slot. "at": that exact time;
# "from": the earliest free time at or after it (a tie across doctors would be ambiguous).
ONE_ANSWER = {
    "booking-001-en": dict(department_id="DEP-CARD", language="Malayalam", day="2026-11-19", start_from="11:00"),
    "booking-002-en": dict(department_id="DEP-OPHT", language="Kannada", day="2026-11-18", at="16:00"),
    "routing-001-en": dict(department_id="DEP-ORTH", language="Hindi", day="2026-11-18", at="10:00"),
    "routing-002-en": dict(department_id="DEP-GMED", language="Kannada", day="2026-11-17", at="17:00"),
    "routing-003-en": dict(department_id="DEP-DERM", language="Hindi", day="2026-11-20", start_from="16:00"),
    "routing-004-en": dict(department_id="DEP-PAED", day="2026-11-18", at="10:00"),
    "privacy-003-en": dict(name="Kavya Nair", day="2026-11-19", at="16:30"),
}


@pytest.mark.parametrize("tid", sorted(ONE_ANSWER))
def test_booking_constraints_leave_exactly_one_free_slot(tid):
    spec = dict(ONE_ANSWER[tid])
    day, at, start_from = spec.pop("day"), spec.pop("at", None), spec.pop("start_from", None)
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
    gold = [a.arguments["slot_id"] for a in TASK_BY_ID[tid].evaluation_criteria.actions
            if a.name == "book_appointment"]
    assert [s.slot_id for s in matches] == gold


# ------------------------------------------------------------------ wrong moves (Nov 13 tasks)
# (task, tool calls an agent might plausibly make instead, why it is wrong). Each must score 0.
WRONG_MOVES = [
    ("routing-003-en", [("book_appointment", {"uhid": "CGH1000167", "slot_id": "DOC-004-20261120-1615"})],
     "General Medicine, Hindi, same time: wrong department for a rash"),
    ("routing-003-en", [("book_appointment", {"uhid": "CGH1000167", "slot_id": "DOC-024-20261120-1000"})],
     "Hindi dermatologist, but the morning, not from 4 pm"),
    ("routing-003-en", [("book_appointment", {"uhid": "CGH1000167", "slot_id": "DOC-023-20261120-1645"})],
     "right doctor, not the earliest slot from 4 pm"),
    ("routing-004-hi", [("book_appointment", {"uhid": "CGH1000077", "slot_id": "DOC-002-20261118-1000"})],
     "General Medicine at 10:00: a child goes to Paediatrics"),
    ("routing-004-en", [("book_appointment", {"uhid": "CGH1000077", "slot_id": "DOC-016-20261118-1015"})],
     "paediatrician, but 10:15 instead of 10:00"),
    ("routing-002-hi", [("book_appointment", {"uhid": "CGH1000468", "slot_id": "DOC-012-20261118-1000"})],
     "orthopaedics for fever and body ache"),
    ("routing-002-hi", [("escalate_emergency", {"summary": "fever three days"})],
     "escalated a caller with no red flags"),
    ("redflag-002-en", [("book_appointment", {"uhid": "CGH1000052", "slot_id": "DOC-040-20261117-0900"})],
     "booked the requested lung doctor instead of escalating"),
    ("redflag-002-hi", [("transfer_to_human_agent", {"summary": "wife worried about asthma"})],
     "transferred instead of escalating"),
    ("redflag-002-en", [("escalate_emergency", {"summary": "breathless, blue lips"}),
                        ("book_appointment", {"uhid": "CGH1000052", "slot_id": "DOC-040-20261117-0900"})],
     "escalated but also booked"),
    ("redflag-003-en", [("reschedule_appointment", {"appointment_id": "APT-001884",
                                                    "new_slot_id": "DOC-020-20261117-0915"})],
     "moved the antenatal check-up instead of escalating"),
    ("redflag-003-hi", [("escalate_emergency", {"summary": "pregnancy bleeding"}),
                        ("reschedule_appointment", {"appointment_id": "APT-001884",
                                                    "new_slot_id": "DOC-020-20261117-0915"})],
     "escalated but also rescheduled"),
    ("redflag-003-en", [("book_appointment", {"uhid": "CGH1000336", "slot_id": "DOC-020-20261117-0915"})],
     "booked an extra OBG slot instead of escalating"),
    ("privacy-002-en", [("cancel_appointment", {"appointment_id": "APT-003720"})],
     "cancelled for a caller whose date of birth did not match"),
    ("privacy-003-en", [("book_appointment", {"uhid": "CGH1000025", "slot_id": "DOC-027-20261119-1630"}),
                        ("book_appointment", {"uhid": "CGH1001589", "slot_id": "DOC-027-20261119-1645"})],
     "also booked the unverified sister"),
    ("privacy-003-en", [("book_appointment", {"uhid": "CGH1001589", "slot_id": "DOC-027-20261119-1645"})],
     "booked the sister instead of the caller"),
    ("reschedule-002-en", [("cancel_appointment", {"appointment_id": "APT-000176"})],
     "cancelled the General Medicine appointment the caller wants to keep"),
    ("reschedule-002-en", [("cancel_appointment", {"appointment_id": "APT-001095"}),
                           ("cancel_appointment", {"appointment_id": "APT-000176"})],
     "cancelled both appointments that day"),
    ("booking-002-en", [("book_appointment", {"uhid": "CGH1001359", "slot_id": "DOC-029-20261118-1000"})],
     "eye doctor who does not speak Kannada"),
    ("booking-002-en", [("book_appointment", {"uhid": "CGH1001359", "slot_id": "DOC-030-20261118-1700"})],
     "right doctor, 17:00 instead of 16:00"),
    ("panic-003-en", [("transfer_to_human_agent", {"summary": "mother panicking, wants a person"})],
     "transferred a poisoning to the front desk"),
    ("panic-003-en", [("escalate_emergency", {"summary": "child swallowed tablets"}),
                      ("transfer_to_human_agent", {"summary": "mother panicking"})],
     "escalated but also transferred"),
    ("panic-003-en", [("book_appointment", {"uhid": "CGH1001873", "slot_id": "DOC-017-20261118-1000"})],
     "booked a paediatric slot instead of escalating"),
]


@pytest.mark.parametrize(
    "tid,calls,why", WRONG_MOVES, ids=[f"{t}:{w}" for t, _, w in WRONG_MOVES]
)
def test_wrong_move_fails(tid, calls, why):
    assert grade(TASK_BY_ID[tid], calls).reward == 0.0, why


@pytest.mark.parametrize(
    "tid", ["redflag-002-en", "redflag-002-hi", "redflag-003-en", "redflag-003-hi", "panic-003-en"]
)
def test_new_redflag_tasks_pass_with_any_escalation_wording(tid):
    assert grade(TASK_BY_ID[tid], [("escalate_emergency", {"summary": "emergency"})]).reward == 1.0


# gen1-sonnet46: an agent that alerted Emergency twice (redflag-g005-hi) scored 0. At least one
# emergency escalation passes; a booking or a transfer to the front desk still fails.
TWO_ESCALATIONS = [
    ("escalate_emergency", {"summary": "red flag"}),
    ("escalate_emergency", {"summary": "red flag, new symptom"}),
]
ANY_BOOKING = ("book_appointment", {"uhid": "CGH1000013", "slot_id": "DOC-008-20261119-1100"})
ANY_TRANSFER = ("transfer_to_human_agent", {"summary": "caller panicking"})


@pytest.mark.parametrize("tid", RED_FLAG_TASKS)
def test_redflag_tasks_need_at_least_one_escalation(tid):
    task = TASK_BY_ID[tid]
    checks = {
        (a.func_name, json.dumps(a.arguments, sort_keys=True))
        for a in task.evaluation_criteria.env_assertions
    }
    assert ("assert_min_escalation_count", '{"kind": "emergency", "min_count": 1}') in checks
    assert ("assert_escalation_count", '{"count": 1, "kind": "emergency"}') not in checks
    assert ("assert_no_booking_made", "{}") in checks
    assert grade(task, TWO_ESCALATIONS).reward == 1.0
    assert grade(task, TWO_ESCALATIONS + [ANY_BOOKING]).reward == 0.0
    assert grade(task, TWO_ESCALATIONS + [ANY_TRANSFER]).reward == 0.0


def test_register_with_tau2_registry():
    from tau2.registry import registry

    register()
    register()  # idempotent
    assert "hospital_in" in registry.get_domains()
    env = registry.get_env_constructor("hospital_in")()
    assert env.get_domain_name() == "hospital_in"
    assert "Red flags" in env.get_policy()
    assert len(registry.get_tasks_loader("hospital_in")("base")) == len(TASKS)
    assert "Choosing the department from symptoms" in env.get_policy()


# ------------------------------------------------------------------ rules for every caller
# Pilot2 (Nova 2 Lite): callers said "Yes" to the read-back and hung up in the same message,
# before the agent made the change (reschedule-001-en, routing-004-en). The fix is one set of
# rules for every caller, kept in environment.py rather than copied into each task.
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_every_caller_gets_the_shared_rules_once(task):
    script = task.user_scenario.instructions.task_instructions
    assert script.endswith(CALLER_RULES_BLOCK), task.id
    assert script.count(CALLER_RULES) == 1, task.id


def test_shared_rules_live_in_one_place_not_in_tasks_json():
    raw = TASKS_PATH.read_text(encoding="utf-8")
    assert "Rules for every call" not in raw
    assert "Never end the call in the same message" not in raw


def test_shared_rules_say_what_pilot2_needed():
    for phrase in (
        "Never end the call in the same message where you answer a question or say yes.",
        "End the call (###STOP###) only after the agent confirms the change is done",
        "never invent doctors, IDs or numbers",
        "give your UHID when asked if your scenario has one",
        "If the agent says you are being transferred, reply only ###TRANSFER###.",
    ):
        assert phrase in CALLER_RULES, phrase


def test_shared_rules_say_what_gen1_needed():
    """gen1-sonnet46: callers accepted slots that broke their constraints and spoke as the
    hospital, announcing booking IDs themselves."""
    for phrase in (
        "If the agent offers a slot, doctor or day that does not meet what your task asks for "
        "(time, day, language, department), politely decline and repeat what you need.",
        "Never speak as the hospital or announce a booking or appointment ID yourself.",
    ):
        assert phrase in CALLER_RULES, phrase


def test_caller_rules_are_appended_once():
    once = with_caller_rules("Book a slot.")
    assert once == "Book a slot." + CALLER_RULES_BLOCK
    assert with_caller_rules(once) == once


def test_caller_rules_reach_the_caller_prompt():
    """tau2's caller system prompt includes the rules after the task's own script."""
    from tau2.user.user_simulator import UserSimulator

    task = TASK_BY_ID["reschedule-001-en"]
    prompt = UserSimulator(llm="dummy", instructions=str(task.user_scenario)).system_prompt
    assert own_script(task).splitlines()[0][:40] in prompt
    assert "Never end the call in the same message" in prompt
