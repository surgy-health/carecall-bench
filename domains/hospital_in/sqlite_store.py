"""Convert HospitalDB <-> SQLite using data/schema.sql.

tau2 runs on the in-memory Pydantic model; SQLite is the storage/interchange
format (and what the FastAPI mock hospital will serve). The schema's foreign keys
and CHECK constraints validate the seed; from_sqlite(to_sqlite(db)) == db.
"""

import sqlite3
from pathlib import Path

from domains.hospital_in.data_model import HospitalDB
from domains.hospital_in.utils import SCHEMA_PATH


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def to_sqlite(db: HospitalDB, path: str | Path) -> None:
    """Write db to a new SQLite file (overwrites). Use ':memory:' only via write_into."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = _connect(path)
    try:
        write_into(conn, db)
    finally:
        conn.close()


def write_into(conn: sqlite3.Connection, db: HospitalDB) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys = ON")
    h = db.hospital
    with conn:
        conn.execute(
            "INSERT INTO hospital_info VALUES (1, ?, ?, ?, ?, ?, ?)",
            (h.name, h.city, h.address, h.main_phone, h.emergency_phone, h.ambulance_number),
        )
        conn.executemany(
            "INSERT INTO departments VALUES (?, ?, ?, ?)",
            [(d.department_id, d.name, d.location, d.treats) for d in db.departments.values()],
        )
        for d in db.doctors.values():
            conn.execute(
                "INSERT INTO doctors VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    d.doctor_id,
                    d.full_name,
                    d.department_id,
                    d.designation,
                    d.qualifications,
                    d.consultation_fee_inr,
                    d.slot_minutes,
                ),
            )
            conn.executemany(
                "INSERT INTO doctor_languages VALUES (?, ?, ?)",
                [(d.doctor_id, i, lang) for i, lang in enumerate(d.languages)],
            )
            conn.executemany(
                "INSERT INTO opd_sessions VALUES (?, ?, ?, ?, ?)",
                [(d.doctor_id, i, s.weekday, s.start, s.end) for i, s in enumerate(d.opd_sessions)],
            )
        conn.executemany(
            "INSERT INTO slots VALUES (?, ?, ?, ?)",
            [(s.slot_id, s.doctor_id, s.start, s.status) for s in db.slots.values()],
        )
        conn.executemany(
            "INSERT INTO services VALUES (?, ?, ?, ?, ?)",
            [
                (s.service_code, s.name, s.category, s.price_inr, s.notes)
                for s in db.services.values()
            ],
        )
        for p in db.insurance_plans.values():
            conn.execute(
                "INSERT INTO insurance_plans VALUES (?, ?, ?, ?)",
                (p.plan_id, p.name, p.kind, p.copay_percent),
            )
            conn.executemany(
                "INSERT INTO plan_categories VALUES (?, ?, ?, ?)",
                [(p.plan_id, "covered", i, c) for i, c in enumerate(p.covered_categories)]
                + [(p.plan_id, "preauth", i, c) for i, c in enumerate(p.preauth_categories)],
            )
        conn.executemany(
            "INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    p.uhid,
                    p.full_name,
                    p.date_of_birth,
                    p.gender,
                    p.phone_number,
                    p.preferred_language,
                    p.area,
                    p.insurance.plan_id,
                    p.insurance.policy_number,
                    p.registered_on,
                )
                for p in db.patients.values()
            ],
        )
        conn.executemany(
            "INSERT INTO appointments VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (a.appointment_id, a.uhid, a.doctor_id, a.slot_id, a.start, a.status, a.booked_on)
                for a in db.appointments.values()
            ],
        )
        conn.executemany(
            "INSERT INTO lab_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.report_id,
                    r.uhid,
                    r.service_code,
                    r.test_name,
                    r.ordered_on,
                    r.status,
                    r.expected_ready_on,
                    r.findings,
                )
                for r in db.lab_reports.values()
            ],
        )
        conn.executemany(
            "INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?)",
            [
                (e.escalation_id, e.kind, e.created_at, e.uhid, e.callback_number, e.summary)
                for e in db.escalations.values()
            ],
        )


def from_sqlite(path: str | Path) -> HospitalDB:
    conn = _connect(path)
    try:
        return read_from(conn)
    finally:
        conn.close()


def read_from(conn: sqlite3.Connection) -> HospitalDB:
    conn.row_factory = sqlite3.Row

    def rows(sql: str) -> list[dict]:
        return [dict(r) for r in conn.execute(sql)]

    hospital = rows("SELECT * FROM hospital_info")[0]
    hospital.pop("id")

    langs: dict[str, list[str]] = {}
    for r in rows("SELECT * FROM doctor_languages ORDER BY doctor_id, position"):
        langs.setdefault(r["doctor_id"], []).append(r["language"])
    sessions: dict[str, list[dict]] = {}
    for r in rows("SELECT * FROM opd_sessions ORDER BY doctor_id, position"):
        sessions.setdefault(r["doctor_id"], []).append(
            {"weekday": r["weekday"], "start": r["start"], "end": r["end"]}
        )
    doctors = {}
    for r in rows("SELECT * FROM doctors ORDER BY rowid"):
        r["languages"] = langs.get(r["doctor_id"], [])
        r["opd_sessions"] = sessions.get(r["doctor_id"], [])
        doctors[r["doctor_id"]] = r

    plan_cats: dict[tuple[str, str], list[str]] = {}
    for r in rows("SELECT * FROM plan_categories ORDER BY plan_id, rule, position"):
        plan_cats.setdefault((r["plan_id"], r["rule"]), []).append(r["category"])
    plans = {}
    for r in rows("SELECT * FROM insurance_plans ORDER BY rowid"):
        r["covered_categories"] = plan_cats.get((r["plan_id"], "covered"), [])
        r["preauth_categories"] = plan_cats.get((r["plan_id"], "preauth"), [])
        plans[r["plan_id"]] = r

    patients = {}
    for r in rows("SELECT * FROM patients ORDER BY rowid"):
        r["insurance"] = {"plan_id": r.pop("plan_id"), "policy_number": r.pop("policy_number")}
        patients[r["uhid"]] = r

    def keyed(table: str, key: str) -> dict:
        return {r[key]: r for r in rows(f"SELECT * FROM {table} ORDER BY rowid")}

    return HospitalDB.model_validate(
        {
            "hospital": hospital,
            "departments": keyed("departments", "department_id"),
            "doctors": doctors,
            "slots": keyed("slots", "slot_id"),
            "patients": patients,
            "appointments": keyed("appointments", "appointment_id"),
            "lab_reports": keyed("lab_reports", "report_id"),
            "services": keyed("services", "service_code"),
            "insurance_plans": plans,
            "escalations": keyed("escalations", "escalation_id"),
        }
    )
