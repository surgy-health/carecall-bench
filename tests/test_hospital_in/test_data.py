"""Seed data integrity and the SQLite schema round-trip."""

import sqlite3
from collections import Counter
from datetime import date

import pytest

from domains.hospital_in.data_model import HospitalDB
from domains.hospital_in.seed import build_db, dump_json
from domains.hospital_in.sqlite_store import from_sqlite, read_from, to_sqlite, write_into
from domains.hospital_in.utils import DB_PATH, NOW

NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M")


@pytest.fixture(scope="module")
def seed() -> HospitalDB:
    return HospitalDB.load(str(DB_PATH))


def test_db_json_matches_generator(seed):
    """db.json is exactly what seed.py produces (so tasks and seed can't drift apart)."""
    assert DB_PATH.read_text(encoding="utf-8") == dump_json(build_db())


def test_sizes(seed):
    s = seed.get_statistics()
    assert s["num_departments"] == 12 and s["num_doctors"] == 40 and s["num_patients"] == 2000
    assert s["num_slots"] > 5000 and s["num_appointments"] > 3000 and s["num_lab_reports"] > 300


def test_sqlite_round_trip(seed, tmp_path):
    path = tmp_path / "hospital.sqlite"
    to_sqlite(seed, path)
    back = from_sqlite(path)
    assert back == seed
    assert back.get_hash() == seed.get_hash()


def test_sqlite_constraints_reject_bad_rows(seed):
    conn = sqlite3.connect(":memory:")
    write_into(conn, seed)
    booked = next(a for a in seed.appointments.values() if a.status == "booked")
    bad_rows = [
        # Second active appointment in the same slot.
        (
            "INSERT INTO appointments VALUES ('APT-999999', ?, ?, ?, ?, 'booked', ?)",
            (booked.uhid, booked.doctor_id, booked.slot_id, booked.start, NOW_ISO),
        ),
        # Unknown patient.
        (
            "INSERT INTO appointments VALUES ('APT-999998', 'CGH0000000', 'DOC-001', ?, ?, 'booked', ?)",
            (booked.slot_id, booked.start, NOW_ISO),
        ),
        # Bad phone number format.
        ("UPDATE patients SET phone_number = '12345' WHERE uhid = ?", (booked.uhid,)),
        # Unknown slot status.
        ("UPDATE slots SET status = 'maybe' WHERE slot_id = ?", (booked.slot_id,)),
    ]
    for sql, params in bad_rows:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, params)
    # A cancelled appointment may share a slot with an active one.
    conn.execute(
        "INSERT INTO appointments VALUES ('APT-999997', ?, ?, ?, ?, 'cancelled', ?)",
        (booked.uhid, booked.doctor_id, booked.slot_id, booked.start, NOW_ISO),
    )
    assert read_from(conn).appointments["APT-999997"].status == "cancelled"


def test_slot_and_appointment_consistency(seed):
    active = Counter(a.slot_id for a in seed.appointments.values() if a.status != "cancelled")
    assert max(active.values()) == 1
    for slot in seed.slots.values():
        assert (slot.status == "booked") == (slot.slot_id in active), slot.slot_id
    for a in seed.appointments.values():
        slot = seed.slots[a.slot_id]
        assert a.doctor_id == slot.doctor_id and a.start == slot.start
        assert a.uhid in seed.patients
        assert a.booked_on < NOW_ISO  # assert_no_booking_made relies on this
        assert (a.status == "booked") == (a.start > NOW_ISO)


def test_department_eligibility(seed):
    def age(uhid):
        dob = date.fromisoformat(seed.patients[uhid].date_of_birth)
        today = NOW.date()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    for a in seed.appointments.values():
        dep = seed.doctors[a.doctor_id].department_id
        if dep == "DEP-PAED":
            assert age(a.uhid) < 18
        elif dep == "DEP-OBGY":
            assert seed.patients[a.uhid].gender == "female"


def test_no_duplicate_upcoming_booking_per_doctor(seed):
    pairs = Counter(
        (a.uhid, a.doctor_id) for a in seed.appointments.values() if a.status == "booked"
    )
    assert max(pairs.values()) == 1


def test_references(seed):
    for d in seed.doctors.values():
        assert d.department_id in seed.departments
    for p in seed.patients.values():
        assert p.insurance.plan_id in seed.insurance_plans
        assert p.registered_on >= p.date_of_birth
    for r in seed.lab_reports.values():
        assert r.uhid in seed.patients and r.service_code in seed.services
        assert (r.findings is not None) == (r.status in ("ready", "collected"))
    assert len({p.phone_number for p in seed.patients.values()}) == len(seed.patients)
    assert seed.escalations == {}
