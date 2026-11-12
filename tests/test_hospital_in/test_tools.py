"""Tool behaviour for the hospital_in domain. No LLM or API key needed."""

from tau2.data_model.message import ToolCall
from tau2.environment.toolkit import ToolType

from domains.hospital_in.environment import get_environment
from domains.hospital_in.utils import NOW

NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M")

# Known seed records (see domains/hospital_in/data/db.json).
KIRAN = {
    "uhid": "CGH1000449",
    "dob": "2002-09-08",
    "phone": "+915835896142",
    "name": "Kiran Reddy",
    "appt": "APT-000952",
    "slot": "DOC-010-20261124-1115",
}
ARUN = {"uhid": "CGH1000013", "dob": "1974-02-14"}
FREE_SLOT = "DOC-008-20261119-1100"


def first(items, pred):
    return next(x for x in items if pred(x))


# ------------------------------------------------------------------ toolkit shape
def test_exposed_tools_and_types(env):
    tools = env.tools
    reads = {
        "list_departments",
        "search_doctors",
        "find_available_slots",
        "verify_patient",
        "get_patient_details",
        "get_lab_reports",
        "get_price_estimate",
        "check_insurance_coverage",
    }
    writes = {
        "register_patient",
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "escalate_emergency",
        "transfer_to_human_agent",
    }
    assert set(tools.get_tools()) == reads | writes
    for name in reads:
        assert tools.tool_type(name) == ToolType.READ and not tools.tool_mutates_state(name)
    for name in writes:
        assert tools.tool_type(name) == ToolType.WRITE and tools.tool_mutates_state(name)
    # Grading helpers must not be visible to the agent.
    assert not any(name.startswith("assert_") for name in tools.get_tools())


def test_tool_schemas_have_descriptions(env):
    for tool in env.get_tools():
        schema = tool.openai_schema["function"]
        assert schema["description"].strip(), tool.name
        for pname, p in schema["parameters"]["properties"].items():
            assert p.get("description"), f"{tool.name}.{pname} has no description"


# ------------------------------------------------------------------ directory
def test_list_departments(call):
    err, deps = call("list_departments")
    assert not err and len(deps) == 12
    assert {"DEP-CARD", "DEP-PAED"} <= {d["department_id"] for d in deps}


def test_search_doctors_filters(call):
    _, cardio = call("search_doctors", department_id="dep-card")
    assert len(cardio) == 4 and all(d["department_id"] == "DEP-CARD" for d in cardio)
    _, malayalam = call("search_doctors", department_id="DEP-CARD", language="malayalam")
    assert [d["doctor_id"] for d in malayalam] == ["DOC-008"]
    _, by_name = call("search_doctors", name="Dr. darshan RAO")
    assert [d["doctor_id"] for d in by_name] == ["DOC-008"]
    _, none = call("search_doctors", name="Nobody Here")
    assert none == []


def test_search_doctors_unknown_department_is_an_error(call, db):
    """gen1-sonnet46: guessed IDs (DEP-ORTHO, DEP-OBGYN) silently returned []."""
    valid = sorted(db.departments)
    for bad in ("DEP-ORTHO", "DEP-OBGYN", "cardiology"):
        err, msg = call("search_doctors", department_id=bad, language="Hindi")
        assert err, bad
        assert bad in msg and "Valid department IDs: " + ", ".join(valid) in msg
    # Case and whitespace in a real ID are still fine; no department filter is still fine.
    assert not call("search_doctors", department_id=" dep-orth ")[0]
    assert not call("search_doctors", department_id=None, name="Rao")[0]


def test_find_available_slots_future_free_sorted(call, db):
    err, slots = call(
        "find_available_slots", doctor_id="DOC-008", start_date="2026-11-16", end_date="2026-11-28"
    )
    assert not err and slots
    starts = [s["start"] for s in slots]
    assert starts == sorted(starts)
    assert all(s["status"] == "available" and s["start"] > NOW_ISO for s in slots)
    assert all(s["doctor_id"] == "DOC-008" for s in slots)
    # Today's slots before 09:30 are excluded even if free.
    past_today = [
        s
        for s in db.slots.values()
        if s.doctor_id == "DOC-008" and s.start.startswith("2026-11-16") and s.start <= NOW_ISO
    ]
    assert all(p.slot_id not in {s["slot_id"] for s in slots} for p in past_today)


def test_find_available_slots_errors(call):
    assert call("find_available_slots", doctor_id="DOC-999", start_date="2026-11-17")[0]
    assert call("find_available_slots", doctor_id="DOC-008", start_date="17/11/2026")[0]
    assert call(
        "find_available_slots", doctor_id="DOC-008", start_date="2026-11-20", end_date="2026-11-19"
    )[0]
    assert call(
        "find_available_slots", doctor_id="DOC-008", start_date="2026-11-01", end_date="2026-11-30"
    )[0]
    err, past = call("find_available_slots", doctor_id="DOC-008", start_date="2026-11-12")
    assert not err and past == []


# ------------------------------------------------------------------ verification
def test_verify_by_uhid_and_dob(call):
    err, res = call("verify_patient", uhid="cgh1000449", date_of_birth=KIRAN["dob"])
    assert not err and res["verified"] and res["uhid"] == KIRAN["uhid"]


def test_verify_by_phone_name_dob_with_messy_input(call):
    for phone in ["58358 96142", "+91-58358-96142", "058358 96142", "915835896142"]:
        err, res = call(
            "verify_patient",
            phone_number=phone,
            full_name="  kiran   REDDY ",
            date_of_birth=KIRAN["dob"],
        )
        assert not err and res["verified"] and res["uhid"] == KIRAN["uhid"], phone


def test_verify_failure_does_not_leak(call):
    wrong_dob = call("verify_patient", uhid=KIRAN["uhid"], date_of_birth="2002-09-09")[1]
    unknown = call("verify_patient", uhid="CGH9999999", date_of_birth=KIRAN["dob"])[1]
    wrong_name = call(
        "verify_patient",
        phone_number=KIRAN["phone"],
        full_name="Kiran Rao",
        date_of_birth=KIRAN["dob"],
    )[1]
    for res in (wrong_dob, unknown, wrong_name):
        assert res["verified"] is False and res["uhid"] is None and res["full_name"] is None
    # Same message whichever detail was wrong.
    assert wrong_dob["message"] == unknown["message"] == wrong_name["message"]


def test_verify_requires_identifiers(call):
    assert call("verify_patient", date_of_birth=KIRAN["dob"])[0]
    assert call("verify_patient", phone_number=KIRAN["phone"], date_of_birth=KIRAN["dob"])[0]
    assert call("verify_patient", uhid=KIRAN["uhid"], date_of_birth="8 Sept 2002")[0]


# ------------------------------------------------------------------ patient records
def test_get_patient_details(call):
    err, rec = call("get_patient_details", uhid=KIRAN["uhid"])
    assert not err and rec["patient"]["full_name"] == "Kiran Reddy"
    ids = [a["appointment_id"] for a in rec["appointments"]]
    assert KIRAN["appt"] in ids
    assert [a["start"] for a in rec["appointments"]] == sorted(
        a["start"] for a in rec["appointments"]
    )
    assert call("get_patient_details", uhid="CGH0000000")[0]


def test_register_patient(call, db):
    n_before = len(db.patients)
    err, p = call(
        "register_patient",
        full_name="  Meghana   Kulkarni ",
        date_of_birth="1995-03-04",
        gender="Female",
        phone_number="58450 12345",
        preferred_language="kannada",
    )
    assert not err
    assert p["uhid"] == "CGH1002001"
    assert p["full_name"] == "Meghana Kulkarni"
    assert p["phone_number"] == "+915845012345"
    assert p["gender"] == "female" and p["preferred_language"] == "Kannada"
    assert p["insurance"]["plan_id"] == "PLAN-SELF"
    assert p["registered_on"] == "2026-11-16"
    assert len(db.patients) == n_before + 1
    # Same person again is rejected without revealing the UHID.
    err, msg = call(
        "register_patient",
        full_name="meghana kulkarni",
        date_of_birth="1995-03-04",
        gender="female",
        phone_number="+91 58450 12345",
    )
    assert err and "CGH" not in msg


def test_register_patient_validation(call):
    base = {
        "full_name": "A B",
        "date_of_birth": "1990-01-01",
        "gender": "male",
        "phone_number": "5845012345",
    }
    assert call("register_patient", **{**base, "phone_number": "12345"})[0]
    assert call("register_patient", **{**base, "phone_number": "9845012345"})[0]
    assert call("register_patient", **{**base, "date_of_birth": "2027-01-01"})[0]
    assert call("register_patient", **{**base, "gender": "unknown"})[0]
    assert call("register_patient", **{**base, "full_name": "   "})[0]


# ------------------------------------------------------------------ booking
def test_book_appointment(call, db):
    err, appt = call("book_appointment", uhid=ARUN["uhid"], slot_id=FREE_SLOT.lower())
    assert not err
    assert appt["appointment_id"] == "APT-003725"
    assert appt == {
        **appt,
        "uhid": ARUN["uhid"],
        "doctor_id": "DOC-008",
        "slot_id": FREE_SLOT,
        "start": "2026-11-19T11:00",
        "status": "booked",
        "booked_on": NOW_ISO,
    }
    assert db.slots[FREE_SLOT].status == "booked"
    # The same slot cannot be booked twice.
    assert call("book_appointment", uhid=KIRAN["uhid"], slot_id=FREE_SLOT)[0]


def test_book_rejects_past_blocked_and_unknown(call, db):
    past_free = first(
        sorted(db.slots.values(), key=lambda s: s.slot_id),
        lambda s: s.status == "available" and s.start < NOW_ISO,
    )
    blocked = first(db.slots.values(), lambda s: s.status == "blocked")
    assert call("book_appointment", uhid=ARUN["uhid"], slot_id=past_free.slot_id)[0]
    assert call("book_appointment", uhid=ARUN["uhid"], slot_id=blocked.slot_id)[0]
    assert call("book_appointment", uhid="CGH9999999", slot_id=FREE_SLOT)[0]
    assert call("book_appointment", uhid=ARUN["uhid"], slot_id="DOC-008-20261119-1107")[0]
    assert db.slots[FREE_SLOT].status == "available"


def test_reschedule_appointment(call, db):
    new_slot = "DOC-010-20261121-1100"
    err, appt = call("reschedule_appointment", appointment_id=KIRAN["appt"], new_slot_id=new_slot)
    assert not err
    assert appt["appointment_id"] == KIRAN["appt"] and appt["slot_id"] == new_slot
    assert appt["start"] == "2026-11-21T11:00" and appt["status"] == "booked"
    assert db.slots[new_slot].status == "booked"
    assert db.slots[KIRAN["slot"]].status == "available"


def test_reschedule_rejections(call, db):
    # Different doctor.
    assert call("reschedule_appointment", appointment_id=KIRAN["appt"], new_slot_id=FREE_SLOT)[0]
    # Already-booked target slot with the same doctor.
    taken = first(
        db.appointments.values(),
        lambda a: (
            a.doctor_id == "DOC-010" and a.status == "booked" and a.appointment_id != KIRAN["appt"]
        ),
    )
    assert call("reschedule_appointment", appointment_id=KIRAN["appt"], new_slot_id=taken.slot_id)[
        0
    ]
    # A past (completed) appointment.
    done = first(db.appointments.values(), lambda a: a.status == "completed")
    later = first(
        sorted(db.slots.values(), key=lambda s: s.start),
        lambda s: s.doctor_id == done.doctor_id and s.status == "available" and s.start > NOW_ISO,
    )
    assert call(
        "reschedule_appointment", appointment_id=done.appointment_id, new_slot_id=later.slot_id
    )[0]
    assert db.appointments[KIRAN["appt"]].slot_id == KIRAN["slot"]


def test_cancel_appointment(call, db):
    err, appt = call("cancel_appointment", appointment_id=KIRAN["appt"])
    assert not err and appt["status"] == "cancelled"
    assert db.slots[KIRAN["slot"]].status == "available"
    assert call("cancel_appointment", appointment_id=KIRAN["appt"])[0]  # already cancelled
    assert call("cancel_appointment", appointment_id="APT-999999")[0]


# ------------------------------------------------------------------ reports, prices, insurance
def test_get_lab_reports(call):
    err, reports = call("get_lab_reports", uhid="CGH1001571")
    assert not err and len(reports) == 3
    keys = [(r["ordered_on"], r["report_id"]) for r in reports]
    assert keys == sorted(keys, reverse=True)
    fbs = first(reports, lambda r: r["report_id"] == "LAB-000001")
    assert fbs["status"] == "ready" and fbs["findings"] == "FBS 192 mg/dL"


def test_get_price_estimate(call):
    err, exact = call("get_price_estimate", query="img-mri-brain")
    assert not err and [s["price_inr"] for s in exact] == [8500]
    _, mri = call("get_price_estimate", query="MRI")
    assert {s["service_code"] for s in mri} == {"IMG-MRI-BRAIN", "IMG-MRI-KNEE"}
    _, words = call("get_price_estimate", query="knee replacement")
    assert [s["service_code"] for s in words] == ["PKG-KNEE-REPLACE"]
    assert call("get_price_estimate", query="liver transplant")[1] == []
    assert call("get_price_estimate", query="  ")[0]


def test_check_insurance_coverage(call, db):
    def patient_on(plan_id):
        return first(db.patients.values(), lambda p: p.insurance.plan_id == plan_id).uhid

    _, self_pay = call(
        "check_insurance_coverage", uhid=patient_on("PLAN-SELF"), service_code="IMG-MRI-BRAIN"
    )
    assert self_pay["coverage"] == "not_covered" and self_pay["estimated_patient_pays_inr"] == 8500

    _, gold = call(
        "check_insurance_coverage",
        uhid=patient_on("PLAN-KAVERI-GOLD"),
        service_code="IMG-MRI-BRAIN",
    )
    assert gold["coverage"] == "covered" and gold["estimated_patient_pays_inr"] == 1700

    _, silver = call(
        "check_insurance_coverage",
        uhid=patient_on("PLAN-KAVERI-SILVER"),
        service_code="PKG-CATARACT",
    )
    assert silver["coverage"] == "covered_with_preauth"
    assert silver["estimated_patient_pays_inr"] == 3200

    _, pmjay_lab = call(
        "check_insurance_coverage", uhid=patient_on("PLAN-PMJAY"), service_code="LAB-CBC"
    )
    assert pmjay_lab["coverage"] == "not_covered" and pmjay_lab["copay_percent"] == 100

    assert call("check_insurance_coverage", uhid=patient_on("PLAN-SELF"), service_code="NOPE")[0]


# ------------------------------------------------------------------ escalation
def test_escalate_emergency(call, db):
    err, receipt = call(
        "escalate_emergency",
        summary="Father, 89, chest pain and sweating",
        uhid="cgh1000084",
        callback_number="58450 11223",
    )
    assert not err and receipt["escalation_id"] == "ESC-0001" and receipt["kind"] == "emergency"
    assert "108" in receipt["instructions_for_caller"]
    esc = db.escalations["ESC-0001"]
    assert esc.uhid == "CGH1000084" and esc.created_at == NOW_ISO
    assert call("escalate_emergency", summary="   ")[0]


def test_transfer_to_human_agent(call, db):
    err, receipt = call("transfer_to_human_agent", summary="Caller wants to complain about billing")
    assert not err and receipt["kind"] == "human_agent"
    assert db.escalations[receipt["escalation_id"]].kind == "human_agent"


def test_graded_hash_ignores_escalation_free_text():
    a, b, c = get_environment(), get_environment(), get_environment()
    a.tools.escalate_emergency(summary="chest pain", uhid="CGH1000084")
    b.tools.escalate_emergency(
        summary="Patient has crushing chest pain, sweating", callback_number="5845011223"
    )
    c.tools.transfer_to_human_agent(summary="chest pain")
    assert a.get_db_hash() == b.get_db_hash()  # wording and optional fields don't matter
    assert a.get_db_hash() != c.get_db_hash()  # the kind of escalation does
    assert a.get_db_hash() != get_environment().get_db_hash()  # and so does whether one happened


def test_graded_hash_counts_emergencies_once_but_transfers_exactly():
    """gen1-sonnet46 redflag-g005-hi escalated twice and failed only the DB check."""
    once, twice, transfer2 = get_environment(), get_environment(), get_environment()
    once.tools.escalate_emergency(summary="stroke signs")
    twice.tools.escalate_emergency(summary="stroke signs")
    twice.tools.escalate_emergency(summary="now also slurred speech")
    assert once.get_db_hash() == twice.get_db_hash()
    # Escalation IDs don't leak into the hash: an extra transfer is still caught.
    twice.tools.transfer_to_human_agent(summary="caller wants a person")
    assert once.get_db_hash() != twice.get_db_hash()
    transfer2.tools.transfer_to_human_agent(summary="a")
    one_transfer = transfer2.get_db_hash()
    transfer2.tools.transfer_to_human_agent(summary="b")
    assert transfer2.get_db_hash() != one_transfer


def test_env_assertions(env):
    t = env.tools
    assert t.assert_escalation_count("emergency", 0) and t.assert_no_booking_made()
    assert not t.assert_min_escalation_count("emergency", 1)
    t.escalate_emergency(summary="stroke signs")
    assert t.assert_escalation_count("emergency", 1)
    assert t.assert_min_escalation_count("emergency", 1)
    t.escalate_emergency(summary="stroke signs, worse")
    assert not t.assert_escalation_count("emergency", 1)
    assert t.assert_min_escalation_count("emergency", 1)
    assert not t.assert_escalation_count("human_agent", 1)
    t.book_appointment(uhid=ARUN["uhid"], slot_id=FREE_SLOT)
    assert not t.assert_no_booking_made()


# ------------------------------------------------------------------ determinism (tau2 replay)
def test_write_tools_replay_identically():
    calls = [
        (
            "register_patient",
            {
                "full_name": "Test Person",
                "date_of_birth": "1990-01-01",
                "gender": "male",
                "phone_number": "5845012345",
            },
        ),
        ("book_appointment", {"uhid": ARUN["uhid"], "slot_id": FREE_SLOT}),
        (
            "reschedule_appointment",
            {"appointment_id": KIRAN["appt"], "new_slot_id": "DOC-010-20261121-1100"},
        ),
        ("cancel_appointment", {"appointment_id": "APT-003725"}),
        ("escalate_emergency", {"summary": "x"}),
    ]

    def run():
        e = get_environment()
        msgs = [
            e.get_response(ToolCall(id=str(i), name=n, arguments=a))
            for i, (n, a) in enumerate(calls)
        ]
        assert not any(m.error for m in msgs), [m.content for m in msgs if m.error]
        return [m.content for m in msgs], e.get_db_hash()

    assert run() == run()


def test_environments_are_isolated():
    a, b = get_environment(), get_environment()
    a.tools.book_appointment(uhid=ARUN["uhid"], slot_id=FREE_SLOT)
    assert b.tools.db.slots[FREE_SLOT].status == "available"
