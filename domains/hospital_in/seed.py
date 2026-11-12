"""Deterministic synthetic seed for CareCall General Hospital, Bengaluru.

    uv run python -m domains.hospital_in.seed            # rewrite data/db.json
    uv run python -m domains.hospital_in.seed --sqlite runs/hospital_in.sqlite

Everything here is synthetic: names are random combinations of common Indian
first names and surnames; mobile numbers are random and all start with 5, which
no real Indian mobile does. Changing this file changes
db.json, which can break task gold actions; tests/test_hospital_in checks both.
"""

import argparse
import json
import random
from datetime import date, datetime, timedelta

from domains.hospital_in.data_model import (
    Appointment,
    Department,
    Doctor,
    HospitalDB,
    HospitalInfo,
    InsurancePlan,
    LabReport,
    OpdSession,
    Patient,
    PatientInsurance,
    Service,
    Slot,
)
from domains.hospital_in.utils import CALENDAR_END, CALENDAR_START, DB_PATH, NOW

SEED = 20261116
N_PATIENTS = 2000
FUTURE_FILL = 0.45  # share of future slots already booked
PAST_FILL = 0.70

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MORNING = ("09:00", "13:00")
EVENING = ("16:00", "19:00")

HOSPITAL = HospitalInfo(
    name="CareCall General Hospital",
    city="Bengaluru",
    address="14 Kaveri Main Road, Bengaluru 560038",
    main_phone="080-4100-2000",
    emergency_phone="080-4100-2108",
    ambulance_number="108",
)

# (id, name, location, treats, number of doctors)
DEPARTMENTS = [
    (
        "DEP-GMED",
        "General Medicine",
        "Ground floor, Block A",
        "Fever, infections, diabetes and BP follow-up, general check-ups",
        5,
    ),
    (
        "DEP-CARD",
        "Cardiology",
        "First floor, Block B",
        "Heart conditions, BP, palpitations, ECG and echo follow-up",
        4,
    ),
    (
        "DEP-ORTH",
        "Orthopaedics",
        "Ground floor, Block B",
        "Bone, joint and back pain, fractures, sports injuries",
        4,
    ),
    (
        "DEP-PAED",
        "Paediatrics",
        "First floor, Block A",
        "Children under 18: fevers, vaccinations, growth",
        4,
    ),
    (
        "DEP-OBGY",
        "Obstetrics and Gynaecology",
        "Second floor, Block A",
        "Pregnancy care, periods, fertility, women's health",
        4,
    ),
    (
        "DEP-DERM",
        "Dermatology",
        "Second floor, Block B",
        "Skin, hair and nail problems, allergies and rashes",
        3,
    ),
    (
        "DEP-ENTT",
        "ENT",
        "Second floor, Block B",
        "Ear, nose and throat, sinus, hearing problems",
        3,
    ),
    (
        "DEP-OPHT",
        "Ophthalmology",
        "Ground floor, Block C",
        "Eye check-ups, cataract, vision problems",
        3,
    ),
    (
        "DEP-NEUR",
        "Neurology",
        "Third floor, Block B",
        "Headaches, seizures, numbness, nerve problems",
        3,
    ),
    (
        "DEP-GAST",
        "Gastroenterology",
        "Third floor, Block A",
        "Stomach, liver, acidity, digestion problems",
        3,
    ),
    ("DEP-ENDO", "Endocrinology", "First floor, Block C", "Diabetes, thyroid, hormone problems", 2),
    (
        "DEP-PULM",
        "Pulmonology",
        "First floor, Block C",
        "Asthma, chronic cough, breathing and lung problems",
        2,
    ),
]

QUALIFICATIONS = {
    "DEP-GMED": "MBBS, MD (General Medicine)",
    "DEP-CARD": "MBBS, MD, DM (Cardiology)",
    "DEP-ORTH": "MBBS, MS (Orthopaedics)",
    "DEP-PAED": "MBBS, MD (Paediatrics)",
    "DEP-OBGY": "MBBS, MS (OBG)",
    "DEP-DERM": "MBBS, MD (Dermatology)",
    "DEP-ENTT": "MBBS, MS (ENT)",
    "DEP-OPHT": "MBBS, MS (Ophthalmology)",
    "DEP-NEUR": "MBBS, MD, DM (Neurology)",
    "DEP-GAST": "MBBS, MD, DM (Gastroenterology)",
    "DEP-ENDO": "MBBS, MD, DM (Endocrinology)",
    "DEP-PULM": "MBBS, MD (Pulmonary Medicine)",
}

FEMALE_FIRST = [
    "Aishwarya",
    "Ananya",
    "Anjali",
    "Bhavana",
    "Deepa",
    "Divya",
    "Farah",
    "Gayathri",
    "Harini",
    "Indira",
    "Jyothi",
    "Kavya",
    "Lakshmi",
    "Meera",
    "Nandini",
    "Nikhita",
    "Padma",
    "Pooja",
    "Priya",
    "Rashmi",
    "Rekha",
    "Roopa",
    "Sana",
    "Shalini",
    "Shruti",
    "Sneha",
    "Sowmya",
    "Sunita",
    "Swathi",
    "Tanvi",
    "Uma",
    "Vidya",
    "Yamini",
    "Zoya",
    "Mary",
    "Asha",
    "Reshma",
    "Neha",
    "Pallavi",
    "Radha",
]
MALE_FIRST = [
    "Abhishek",
    "Aditya",
    "Ajay",
    "Anand",
    "Arjun",
    "Bharath",
    "Chetan",
    "Darshan",
    "Farhan",
    "Ganesh",
    "Harish",
    "Imran",
    "Karthik",
    "Kiran",
    "Mahesh",
    "Manjunath",
    "Mohan",
    "Naveen",
    "Nikhil",
    "Prakash",
    "Pradeep",
    "Rahul",
    "Raghavendra",
    "Rajesh",
    "Ramesh",
    "Ravi",
    "Rohit",
    "Sachin",
    "Santosh",
    "Siddharth",
    "Srinivas",
    "Suresh",
    "Vikram",
    "Vinay",
    "Vivek",
    "Yusuf",
    "Joseph",
    "Arun",
    "Deepak",
    "Gopal",
]
SURNAMES = [
    "Rao",
    "Reddy",
    "Shetty",
    "Gowda",
    "Hegde",
    "Kulkarni",
    "Nair",
    "Menon",
    "Pillai",
    "Iyer",
    "Iyengar",
    "Sharma",
    "Verma",
    "Gupta",
    "Agarwal",
    "Singh",
    "Patil",
    "Joshi",
    "Bhat",
    "Kamath",
    "Naidu",
    "Murthy",
    "Prasad",
    "Krishnan",
    "Subramanian",
    "Das",
    "Banerjee",
    "Mukherjee",
    "Khan",
    "Shaikh",
    "Ahmed",
    "D'Souza",
    "Fernandes",
    "Thomas",
    "Mathew",
    "George",
    "Jain",
    "Mehta",
    "Chauhan",
    "Yadav",
]
AREAS = [
    "Indiranagar",
    "Jayanagar",
    "Koramangala",
    "Whitefield",
    "Malleshwaram",
    "HSR Layout",
    "Rajajinagar",
    "Yelahanka",
    "Banashankari",
    "Marathahalli",
    "Hebbal",
    "BTM Layout",
    "Basavanagudi",
    "JP Nagar",
    "Electronic City",
    "RT Nagar",
    "Vijayanagar",
    "Frazer Town",
]
LANGUAGE_WEIGHTS = {
    "English": 30,
    "Hindi": 25,
    "Kannada": 25,
    "Tamil": 8,
    "Telugu": 7,
    "Malayalam": 5,
}

SERVICES = [
    ("LAB-CBC", "Complete blood count (CBC)", "lab", 450, None),
    ("LAB-HBA1C", "HbA1c (glycated haemoglobin)", "lab", 650, None),
    ("LAB-FBS", "Fasting blood sugar", "lab", 150, "10-12 hours fasting"),
    ("LAB-LIPID", "Lipid profile", "lab", 800, "10-12 hours fasting"),
    ("LAB-TFT", "Thyroid profile (T3, T4, TSH)", "lab", 700, None),
    ("LAB-LFT", "Liver function test", "lab", 900, None),
    ("LAB-KFT", "Kidney function test", "lab", 950, None),
    ("LAB-URINE", "Urine routine", "lab", 250, None),
    ("LAB-VITD", "Vitamin D (25-OH)", "lab", 1400, None),
    ("IMG-XRAY-CHEST", "X-ray chest", "imaging", 500, None),
    ("IMG-XRAY-KNEE", "X-ray knee (both views)", "imaging", 700, None),
    ("IMG-ECG", "ECG", "imaging", 350, None),
    ("IMG-ECHO", "2D echo", "imaging", 2500, None),
    ("IMG-TMT", "Treadmill test (TMT)", "imaging", 2200, "Light meal 2 hours before"),
    ("IMG-USG-ABD", "Ultrasound abdomen", "imaging", 1500, "6 hours fasting, full bladder"),
    ("IMG-CT-HEAD", "CT scan head", "imaging", 3500, None),
    ("IMG-MRI-BRAIN", "MRI brain", "imaging", 8500, None),
    ("IMG-MRI-KNEE", "MRI knee", "imaging", 7500, None),
    ("PRC-ENDOSCOPY", "Upper GI endoscopy", "procedure", 6000, "8 hours fasting"),
    ("PRC-COLONOSCOPY", "Colonoscopy", "procedure", 9500, "Bowel preparation the day before"),
    (
        "PKG-HEALTH-BASIC",
        "Basic health check package",
        "package",
        2499,
        "CBC, FBS, lipid profile, urine routine, ECG, physician consult",
    ),
    (
        "PKG-HEALTH-EXEC",
        "Executive health check package",
        "package",
        6999,
        "Basic package plus TFT, LFT, KFT, HbA1c, X-ray chest, echo, 3 consults",
    ),
    (
        "PKG-CATARACT",
        "Cataract surgery package (one eye, monofocal lens)",
        "package",
        32000,
        "Day care; includes lens, surgeon and OT charges",
    ),
    (
        "PKG-DELIVERY-NORMAL",
        "Normal delivery package",
        "package",
        55000,
        "2-day stay, twin-sharing room",
    ),
    (
        "PKG-DELIVERY-CSEC",
        "Caesarean delivery package",
        "package",
        95000,
        "4-day stay, twin-sharing room",
    ),
    (
        "PKG-KNEE-REPLACE",
        "Total knee replacement package (one knee)",
        "package",
        285000,
        "5-day stay; standard implant",
    ),
    ("PKG-ANGIOGRAPHY", "Coronary angiography package", "package", 18000, "Day care"),
]

INSURANCE_PLANS = [
    InsurancePlan(
        plan_id="PLAN-SELF",
        name="Self-pay",
        kind="self_pay",
        covered_categories=[],
        preauth_categories=[],
        copay_percent=100,
    ),
    InsurancePlan(
        plan_id="PLAN-PMJAY",
        name="Ayushman Bharat PM-JAY (simplified)",
        kind="government_scheme",
        covered_categories=[],
        preauth_categories=["procedure", "package"],
        copay_percent=0,
    ),
    InsurancePlan(
        plan_id="PLAN-KAVERI-SILVER",
        name="Kaveri Health Insurance Silver (fictional)",
        kind="private_insurance",
        covered_categories=[],
        preauth_categories=["procedure", "package"],
        copay_percent=10,
    ),
    InsurancePlan(
        plan_id="PLAN-KAVERI-GOLD",
        name="Kaveri Health Insurance Gold (fictional)",
        kind="private_insurance",
        covered_categories=["consultation", "lab", "imaging"],
        preauth_categories=["procedure", "package"],
        copay_percent=20,
    ),
    InsurancePlan(
        plan_id="PLAN-NANDI-CORP",
        name="Nandi Tech Employees Group Cover (fictional)",
        kind="corporate",
        covered_categories=["consultation", "lab", "imaging"],
        preauth_categories=["procedure", "package"],
        copay_percent=0,
    ),
]
PLAN_WEIGHTS = {
    "PLAN-SELF": 50,
    "PLAN-PMJAY": 10,
    "PLAN-KAVERI-SILVER": 15,
    "PLAN-KAVERI-GOLD": 10,
    "PLAN-NANDI-CORP": 15,
}

# Tests a department typically orders, with a findings template (all synthetic).
DEPT_TESTS = {
    "DEP-GMED": [("LAB-CBC", "Hb {hb} g/dL, WBC {wbc}/uL"), ("LAB-FBS", "FBS {fbs} mg/dL")],
    "DEP-CARD": [("LAB-LIPID", "LDL {ldl} mg/dL, HDL {hdl} mg/dL"), ("IMG-ECHO", "EF {ef}%")],
    "DEP-ORTH": [
        ("IMG-XRAY-KNEE", "Joint space narrowing grade {grade}"),
        ("LAB-VITD", "Vitamin D {vitd} ng/mL"),
    ],
    "DEP-PAED": [("LAB-CBC", "Hb {hb} g/dL, WBC {wbc}/uL")],
    "DEP-OBGY": [
        ("IMG-USG-ABD", "Findings reported by radiologist, see report"),
        ("LAB-TFT", "TSH {tsh} mIU/L"),
    ],
    "DEP-DERM": [("LAB-CBC", "Hb {hb} g/dL, WBC {wbc}/uL")],
    "DEP-ENTT": [("LAB-CBC", "Hb {hb} g/dL, WBC {wbc}/uL")],
    "DEP-OPHT": [("LAB-FBS", "FBS {fbs} mg/dL")],
    "DEP-NEUR": [
        ("IMG-MRI-BRAIN", "Findings reported by radiologist, see report"),
        ("LAB-VITD", "Vitamin D {vitd} ng/mL"),
    ],
    "DEP-GAST": [
        ("LAB-LFT", "ALT {alt} U/L, AST {ast} U/L"),
        ("IMG-USG-ABD", "Findings reported by radiologist, see report"),
    ],
    "DEP-ENDO": [("LAB-HBA1C", "HbA1c {hba1c}%"), ("LAB-TFT", "TSH {tsh} mIU/L")],
    "DEP-PULM": [("IMG-XRAY-CHEST", "Findings reported by radiologist, see report")],
}


def _weighted(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _phone(rng: random.Random, used: set[str]) -> str:
    """Synthetic mobile '+915XXXXXXXXX'. Real Indian mobiles start with 6-9, so a
    leading 5 can never reach a real person."""
    while True:
        rng.choice("6789")  # discarded: keeps the RNG stream, so all other seed data is unchanged
        num = "+915" + "".join(rng.choice("0123456789") for _ in range(9))
        if num not in used:
            used.add(num)
            return num


def _age(dob: date) -> int:
    today = NOW.date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _fill_findings(rng: random.Random, template: str) -> str:
    vals = {
        "hb": round(rng.uniform(9.5, 15.5), 1),
        "wbc": rng.randrange(4000, 14000, 100),
        "fbs": rng.randrange(80, 210),
        "ldl": rng.randrange(70, 190),
        "hdl": rng.randrange(32, 70),
        "ef": rng.randrange(35, 66),
        "grade": rng.randint(1, 4),
        "vitd": round(rng.uniform(8, 45), 1),
        "tsh": round(rng.uniform(0.4, 9.5), 2),
        "alt": rng.randrange(15, 140),
        "ast": rng.randrange(15, 120),
        "hba1c": round(rng.uniform(5.2, 10.5), 1),
    }
    return template.format(**vals)


def build_db() -> HospitalDB:
    rng = random.Random(SEED)

    # ---- departments and doctors
    departments = {
        d[0]: Department(department_id=d[0], name=d[1], location=d[2], treats=d[3])
        for d in DEPARTMENTS
    }
    doctors: dict[str, Doctor] = {}
    used_names: set[str] = set()
    n = 0
    for dep_id, _, _, _, count in DEPARTMENTS:
        for i in range(count):
            n += 1
            female = rng.random() < (0.85 if dep_id == "DEP-OBGY" else 0.45)
            while True:
                first = rng.choice(FEMALE_FIRST if female else MALE_FIRST)
                name = f"Dr. {first} {rng.choice(SURNAMES)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            senior = i == 0 or rng.random() < 0.3
            langs = ["English"] + rng.sample(
                ["Hindi", "Kannada", "Tamil", "Telugu", "Malayalam"], k=rng.randint(1, 3)
            )
            days = sorted(rng.sample(range(6), k=rng.randint(3, 5)))  # Mon-Sat
            session = MORNING if rng.random() < 0.6 else EVENING
            doc_id = f"DOC-{n:03d}"
            doctors[doc_id] = Doctor(
                doctor_id=doc_id,
                full_name=name,
                department_id=dep_id,
                designation="Senior Consultant" if senior else "Consultant",
                qualifications=QUALIFICATIONS[dep_id],
                languages=langs,
                consultation_fee_inr=(1200 if senior else 800) + rng.choice([0, 100, 200]),
                slot_minutes=15,
                opd_sessions=[
                    OpdSession(weekday=WEEKDAYS[d], start=session[0], end=session[1]) for d in days
                ],
            )

    # ---- slots
    slots: dict[str, Slot] = {}
    day = CALENDAR_START
    while day <= CALENDAR_END:
        wd = WEEKDAYS[day.weekday()]
        for doc in doctors.values():
            for sess in doc.opd_sessions:
                if sess.weekday != wd:
                    continue
                t = datetime.combine(day.date(), datetime.strptime(sess.start, "%H:%M").time())
                end = datetime.combine(day.date(), datetime.strptime(sess.end, "%H:%M").time())
                while t < end:
                    sid = f"{doc.doctor_id}-{t:%Y%m%d-%H%M}"
                    slots[sid] = Slot(
                        slot_id=sid,
                        doctor_id=doc.doctor_id,
                        start=t.strftime("%Y-%m-%dT%H:%M"),
                        status="available",
                    )
                    t += timedelta(minutes=doc.slot_minutes)
        day += timedelta(days=1)

    # ---- patients
    patients: dict[str, Patient] = {}
    used_phones: set[str] = set()
    for i in range(N_PATIENTS):
        uhid = f"CGH{1000001 + i:07d}"
        gender = "female" if rng.random() < 0.5 else "male"
        first = rng.choice(FEMALE_FIRST if gender == "female" else MALE_FIRST)
        age = rng.choice(
            [
                rng.randint(0, 17),
                rng.randint(18, 45),
                rng.randint(18, 45),
                rng.randint(46, 64),
                rng.randint(65, 88),
            ]
        )
        dob = date(NOW.year - age - 1, 1, 1) + timedelta(days=rng.randint(0, 364))
        plan_id = _weighted(rng, PLAN_WEIGHTS)
        reg_days = rng.randint(0, (NOW.date() - date(2018, 1, 1)).days - 30)
        reg = max(date(2018, 1, 1) + timedelta(days=reg_days), dob)
        patients[uhid] = Patient(
            uhid=uhid,
            full_name=f"{first} {rng.choice(SURNAMES)}",
            date_of_birth=dob.isoformat(),
            gender=gender,
            phone_number=_phone(rng, used_phones),
            preferred_language=_weighted(rng, LANGUAGE_WEIGHTS),
            area=rng.choice(AREAS),
            insurance=PatientInsurance(
                plan_id=plan_id,
                policy_number=None
                if plan_id == "PLAN-SELF"
                else f"{plan_id.split('-')[1][:3]}{rng.randint(10**7, 10**8 - 1)}",
            ),
            registered_on=reg.isoformat(),
        )

    # Eligible patients per department (age/sex constraints).
    def eligible(dep_id: str) -> list[str]:
        out = []
        for p in patients.values():
            age = _age(date.fromisoformat(p.date_of_birth))
            if dep_id == "DEP-PAED":
                ok = age < 18
            elif dep_id == "DEP-OBGY":
                ok = p.gender == "female" and 18 <= age <= 50
            elif dep_id in ("DEP-ENTT", "DEP-DERM", "DEP-OPHT"):
                ok = True
            else:
                ok = age >= 18
            if ok:
                out.append(p.uhid)
        return out

    pools = {dep: eligible(dep) for dep in departments}

    # ---- appointments and lab reports
    appointments: dict[str, Appointment] = {}
    lab_reports: dict[str, LabReport] = {}
    active_pairs: set[tuple[str, str]] = set()  # (uhid, doctor) with an upcoming booking
    apt_n = lab_n = 0
    for slot in sorted(slots.values(), key=lambda s: s.slot_id):
        start = datetime.strptime(slot.start, "%Y-%m-%dT%H:%M")
        future = start > NOW
        if rng.random() >= (FUTURE_FILL if future else PAST_FILL):
            continue
        doc = doctors[slot.doctor_id]
        uhid = rng.choice(pools[doc.department_id])
        if future and (uhid, doc.doctor_id) in active_pairs:
            continue
        booked_on = start - timedelta(days=rng.randint(1, 10), hours=rng.randint(0, 8))
        booked_on = min(booked_on, NOW - timedelta(hours=1))
        if future:
            status = "booked"
            active_pairs.add((uhid, doc.doctor_id))
        else:
            status = rng.choices(["completed", "no_show", "cancelled"], weights=[85, 8, 7])[0]
        apt_n += 1
        apt_id = f"APT-{apt_n:06d}"
        appointments[apt_id] = Appointment(
            appointment_id=apt_id,
            uhid=uhid,
            doctor_id=doc.doctor_id,
            slot_id=slot.slot_id,
            start=slot.start,
            status=status,
            booked_on=booked_on.strftime("%Y-%m-%dT%H:%M"),
        )
        if status != "cancelled":
            slot.status = "booked"
        if status == "completed" and rng.random() < 0.35:
            code, template = rng.choice(DEPT_TESTS[doc.department_id])
            ordered = start.date()
            age_days = (NOW.date() - ordered).days
            ready_on = ordered + timedelta(days=1 if code.startswith("LAB") else 2)
            if ready_on > NOW.date():
                rstatus = "processing"
            elif age_days >= 4 and rng.random() < 0.5:
                rstatus = "collected"
            else:
                rstatus = "ready"
            if code.startswith("LAB") and rng.random() < 0.08:
                rstatus = "sample_pending"
            lab_n += 1
            rid = f"LAB-{lab_n:06d}"
            lab_reports[rid] = LabReport(
                report_id=rid,
                uhid=uhid,
                service_code=code,
                test_name=next(s[1] for s in SERVICES if s[0] == code),
                ordered_on=ordered.isoformat(),
                status=rstatus,
                expected_ready_on=ready_on.isoformat() if rstatus != "collected" else None,
                findings=_fill_findings(rng, template)
                if rstatus in ("ready", "collected")
                else None,
            )

    # A few blocked slots (doctor leave), only on free future slots.
    future_free = [
        s
        for s in slots.values()
        if s.status == "available" and s.start > NOW.strftime("%Y-%m-%dT%H:%M")
    ]
    for s in rng.sample(sorted(future_free, key=lambda s: s.slot_id), k=40):
        s.status = "blocked"

    return HospitalDB(
        hospital=HOSPITAL,
        departments=departments,
        doctors=doctors,
        slots=slots,
        patients=patients,
        appointments=appointments,
        lab_reports=lab_reports,
        services={
            s[0]: Service(service_code=s[0], name=s[1], category=s[2], price_inr=s[3], notes=s[4])
            for s in SERVICES
        },
        insurance_plans={p.plan_id: p for p in INSURANCE_PLANS},
        escalations={},
    )


def dump_json(db: HospitalDB) -> str:
    return json.dumps(db.model_dump(), indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", help="Also write a SQLite copy to this path")
    args = parser.parse_args()
    db = build_db()
    DB_PATH.write_text(dump_json(db), encoding="utf-8")
    print(f"wrote {DB_PATH}: {db.get_statistics()}")
    if args.sqlite:
        from domains.hospital_in.sqlite_store import to_sqlite

        to_sqlite(db, args.sqlite)
        print(f"wrote {args.sqlite}")


if __name__ == "__main__":
    main()
