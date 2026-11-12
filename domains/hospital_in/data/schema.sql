-- SQLite schema for CareCall General Hospital (hospital_in domain).
-- Mirrors domains/hospital_in/data_model.py; sqlite_store.py converts both ways and
-- tests/test_hospital_in checks the seed round-trips through this schema.
-- Times are IST local, ISO 8601 text ('YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
-- All data is synthetic.

PRAGMA foreign_keys = ON;

CREATE TABLE hospital_info (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    name             TEXT NOT NULL,
    city             TEXT NOT NULL,
    address          TEXT NOT NULL,
    main_phone       TEXT NOT NULL,
    emergency_phone  TEXT NOT NULL,
    ambulance_number TEXT NOT NULL
);

CREATE TABLE departments (
    department_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    location      TEXT NOT NULL,
    treats        TEXT NOT NULL
);

CREATE TABLE doctors (
    doctor_id           TEXT PRIMARY KEY,
    full_name           TEXT NOT NULL,
    department_id       TEXT NOT NULL REFERENCES departments(department_id),
    designation         TEXT NOT NULL,
    qualifications      TEXT NOT NULL,
    consultation_fee_inr INTEGER NOT NULL CHECK (consultation_fee_inr > 0),
    slot_minutes        INTEGER NOT NULL CHECK (slot_minutes > 0)
);

CREATE TABLE doctor_languages (
    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
    position  INTEGER NOT NULL,
    language  TEXT NOT NULL,
    PRIMARY KEY (doctor_id, position)
);

CREATE TABLE opd_sessions (
    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
    position  INTEGER NOT NULL,
    weekday   TEXT NOT NULL CHECK (weekday IN ('Mon','Tue','Wed','Thu','Fri','Sat','Sun')),
    start     TEXT NOT NULL,
    "end"     TEXT NOT NULL CHECK ("end" > start),
    PRIMARY KEY (doctor_id, position)
);

CREATE TABLE slots (
    slot_id   TEXT PRIMARY KEY,
    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
    start     TEXT NOT NULL,
    status    TEXT NOT NULL CHECK (status IN ('available','booked','blocked')),
    UNIQUE (doctor_id, start)
);

CREATE TABLE services (
    service_code TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL CHECK (category IN ('consultation','lab','imaging','procedure','package')),
    price_inr    INTEGER NOT NULL CHECK (price_inr >= 0),
    notes        TEXT
);

CREATE TABLE insurance_plans (
    plan_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('self_pay','private_insurance','corporate','government_scheme')),
    copay_percent INTEGER NOT NULL CHECK (copay_percent BETWEEN 0 AND 100)
);

-- Which service categories a plan covers outright ('covered') or after pre-authorisation ('preauth').
CREATE TABLE plan_categories (
    plan_id  TEXT NOT NULL REFERENCES insurance_plans(plan_id),
    rule     TEXT NOT NULL CHECK (rule IN ('covered','preauth')),
    position INTEGER NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('consultation','lab','imaging','procedure','package')),
    PRIMARY KEY (plan_id, rule, position),
    UNIQUE (plan_id, category)
);

CREATE TABLE patients (
    uhid               TEXT PRIMARY KEY,
    full_name          TEXT NOT NULL,
    date_of_birth      TEXT NOT NULL,
    gender             TEXT NOT NULL CHECK (gender IN ('female','male','other')),
    phone_number       TEXT NOT NULL CHECK (phone_number GLOB '+915[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'),
    preferred_language TEXT NOT NULL,
    area               TEXT NOT NULL,
    plan_id            TEXT NOT NULL REFERENCES insurance_plans(plan_id),
    policy_number      TEXT,
    registered_on      TEXT NOT NULL CHECK (registered_on >= date_of_birth)
);
CREATE INDEX ix_patients_phone ON patients(phone_number);

CREATE TABLE appointments (
    appointment_id TEXT PRIMARY KEY,
    uhid           TEXT NOT NULL REFERENCES patients(uhid),
    doctor_id      TEXT NOT NULL REFERENCES doctors(doctor_id),
    slot_id        TEXT NOT NULL REFERENCES slots(slot_id),
    start          TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('booked','completed','cancelled','no_show')),
    booked_on      TEXT NOT NULL
);
-- A slot holds at most one non-cancelled appointment.
CREATE UNIQUE INDEX ux_appointments_active_slot ON appointments(slot_id) WHERE status != 'cancelled';
CREATE INDEX ix_appointments_uhid ON appointments(uhid);

CREATE TABLE lab_reports (
    report_id         TEXT PRIMARY KEY,
    uhid              TEXT NOT NULL REFERENCES patients(uhid),
    service_code      TEXT NOT NULL REFERENCES services(service_code),
    test_name         TEXT NOT NULL,
    ordered_on        TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('sample_pending','processing','ready','collected')),
    expected_ready_on TEXT,
    findings          TEXT  -- confidential; never read out on a call
);
CREATE INDEX ix_lab_reports_uhid ON lab_reports(uhid);

-- uhid is free text from the call (may be wrong or absent), so it has no foreign key.
CREATE TABLE escalations (
    escalation_id   TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('emergency','human_agent')),
    created_at      TEXT NOT NULL,
    uhid            TEXT,
    callback_number TEXT,
    summary         TEXT NOT NULL
);
