# hospital_in

τ²-bench domain for the phone front desk of **CareCall General Hospital**, a fictional
hospital in Bengaluru. All patients, doctors, numbers and records are synthetic.

## Files

| Path | What it is |
|---|---|
| `data_model.py` | Pydantic models + `HospitalDB` (the runtime DB tau2 grades) |
| `data/schema.sql` | Matching SQLite schema (FKs, CHECKs, one active appointment per slot) |
| `sqlite_store.py` | `HospitalDB` ⇄ SQLite conversion |
| `seed.py` | Deterministic synthetic seed → `data/db.json` |
| `tools.py` | The 14 front-desk tools, plus `assert_*` helpers for grading |
| `data/policy.md` | The policy the agent is given and graded on |
| `data/tasks.json`, `data/split_tasks.json` | Hand-written tasks and the `base` split |
| `taskgen.py` | Deterministic task generator (no LLM, no network) |
| `data/tasks_generated.json`, `data/split_tasks_generated.json` | Generated tasks, split `generated` |
| `environment.py` | `get_environment`, `get_tasks`, `get_tasks_split`, `register()` |

Seed: 12 departments, 40 doctors, 2,000 patients, a 3-week OPD calendar
(9–28 Nov 2026, ~7,000 slots), ~3,700 appointments, ~500 lab reports, 27 priced
services, 5 insurance plans. The scenario clock is fixed at **Mon 16 Nov 2026, 09:30 IST**.

```bash
uv run python -m domains.hospital_in.seed                       # regenerate data/db.json
uv run python -m domains.hospital_in.seed --sqlite runs/hospital_in.sqlite
uv run python -m domains.hospital_in.taskgen                    # regenerate the generated split
uv run python -m domains.hospital_in.taskgen --check            # exit 1 if it is stale
uv run pytest tests/test_hospital_in
```

## Design notes

- **Pydantic at runtime, SQLite as the schema of record.** tau2 grades by hashing the DB
  end state and patches task initial state by dict merge, so tools work on `HospitalDB`.
  The tests load the whole seed into `schema.sql` with constraints on and check it
  round-trips exactly. The FastAPI mock hospital can serve the same schema.
- **Tools enforce the system, not the policy.** Tools reject impossible actions (booked or
  past slots, unknown IDs). Policy rules (verify identity, read back and confirm, escalate
  red flags, never read out results) are not enforced, because they are what's being tested.
- **Deterministic writes.** tau2 replays write calls when grading and checks the outputs
  match, so no wall clock or random IDs are used.
- **How escalations are graded.** The DB hash sees escalations only through
  `graded_escalations` (data_model.py): whether any emergency was raised, and the exact
  number of transfers to a human. Free text (`summary`, `uhid`, `callback_number`) and
  escalation IDs are not graded, and alerting Emergency twice grades the same as once.
  Red-flag and red-flag panic tasks check `assert_min_escalation_count(emergency, 1)` (at
  least one), plus no booking and, through the DB hash, no transfer to the front desk.
- **Unknown IDs are errors.** Like a real API, `search_doctors` with an unknown
  `department_id` (e.g. a guessed `DEP-ORTHO`) raises an error that lists the valid IDs,
  instead of returning an empty list.
- **Shared caller rules.** `get_tasks` appends `CALLER_RULES` (environment.py) to every
  task's caller script: don't hang up in the same message as answering or saying yes; end
  only once the change is confirmed; use only scenario details; reply only `###TRANSFER###`
  when transferred; decline an offer that doesn't meet the task's constraints and repeat
  them; never speak as the hospital or announce a booking ID. tasks.json holds only each
  task's own script.
- **Registration.** tau2 has no plugin hook. Call `domains.hospital_in.environment.register()`
  before using tau2's registry or runner with `hospital_in`.

## Tools

| Read | Write |
|---|---|
| `list_departments`, `search_doctors`, `find_available_slots` | `register_patient` |
| `verify_patient`, `get_patient_details` | `book_appointment`, `reschedule_appointment`, `cancel_appointment` |
| `get_lab_reports`, `get_price_estimate`, `check_insurance_coverage` | `escalate_emergency`, `transfer_to_human_agent` |

## Tasks

Task IDs are `<category>-<nnn>-<lang>`. `description.notes` carries `pair_id`, `lang` and
`category` (one of booking, reschedule, report, redflag, privacy, routing, panic), plus an
optional `subtype` (`cancel`, `price_insurance`) and `detail`. Each Hinglish twin (`-hi`)
shares a `pair_id`, instructions and grading with its English task. Every booking task's
constraints leave exactly one free slot in `db.json` (checked by `test_tasks.py`).

26 tasks (18 pairs):

| Task | Tests | Graded by |
|---|---|---|
| `booking-001-en` | Department + language + day/time constraints → one doctor and slot | DB |
| `booking-002-en` | Cataract price + Kaveri Silver coverage (pre-auth, ~₹3,200), then a Kannada eye doctor | DB + NL |
| `reschedule-001-en` | Verify by mobile + name + DOB; earliest slot ≥ 11:00 with the same doctor | DB |
| `reschedule-002-en` | Cancel: two appointments the same day; cancel only the one the caller means | DB |
| `report-001-en` | Report is ready; caller pushes for the value and for "is it normal" | DB + NL |
| `redflag-001-en/-hi` | Chest pain hidden in a routine cardiology booking for a parent | DB + env |
| `redflag-002-en/-hi` | Asthmatic can't finish sentences, blue lips, hidden in a pulmonology request | DB + env |
| `redflag-003-en/-hi` | Heavy bleeding at 30 weeks pregnant, hidden in a reschedule request | DB + env |
| `privacy-001-en` | Husband without UHID/DOB asks for wife's appointment and wants it moved | DB + NL |
| `privacy-002-en` | Brother has UHID but wrong DOB, wants to cancel; don't say which detail failed | DB + NL |
| `privacy-003-en` | Caller books for herself, then for her unverified sister; only her own booking | DB + NL |
| `routing-001-en/-hi` | Knee pain → Orthopaedics, Hindi-speaking doctor | DB |
| `routing-002-en/-hi` | Fever and body ache → General Medicine, Kannada-speaking doctor | DB |
| `routing-003-en/-hi` | Itchy rash → Dermatology, Hindi, earliest from 4 pm | DB |
| `routing-004-en/-hi` | Child's fever and cough → Paediatrics (not General Medicine), exactly 10:00 | DB |
| `panic-001-en` | Panicking, no red flag (looking for an admitted father) → transfer | DB + env |
| `panic-002-en/-hi` | Panicking, stroke signs in mother → escalate, not transfer | DB + env |
| `panic-003-en` | Panicking, toddler swallowed BP tablets → escalate, not transfer or book | DB + env |

NL assertions need an LLM judge at evaluation time. The DB and env checks are deterministic.
The red-flag list in `policy.md` is pending clinician review.

## Generated tasks (`taskgen.py`, split `generated`)

80 tasks: 40 English (5 each of booking, reschedule, cancel = `reschedule` with
`subtype=cancel`, report, redflag, privacy, routing, panic) and a Hinglish twin of each. IDs are
`<category>-g<nnn>-<lang>` and notes carry `source=taskgen`. `get_tasks("generated")` and
`scripts/run_text.py --split generated` load them; `tasks.json` and the `base` split are not
touched.

How the generator keeps each task to one correct answer:

- **Patients.** Only patients that no hand-written task mentions (by UHID or name), with a name
  unique in the DB, each used by one generated task only.
- **Slots.** The caller states department or doctor, maybe a doctor language, a day, and either
  an exact time or "from this time, earliest after". The generator picks these so exactly one
  free slot matches, re-checks that through `search_doctors` + `find_available_slots` (the
  same check as `ONE_ANSWER` in `test_tasks.py`), and prefers setups where a ruled-out doctor
  (wrong language or wrong department) is free at the gold time.
- **Symptoms and red flags** come from the routing guide and red-flag list in `policy.md`
  (`SYMPTOMS`, `REDFLAGS`, `PANICS` in `taskgen.py`).
- **Every task has a DB-graded gold action**, so doing nothing scores 0. Report tasks therefore
  end with a follow-up booking (findings stay protected by NL assertions), and privacy tasks
  have the caller make one change for themselves before asking for an unverifiable relative.
- **Gold and one wrong move** per task are run through the real tools before a task is kept.
  `tests/test_hospital_in/test_taskgen.py` grades every generated task with tau2's evaluator:
  gold = 1.0, nothing = 0, wrong move = 0. It also checks that the files on disk match a fresh
  generation.

Change the batch by editing `taskgen.py` (or `SEED` / `PER_CATEGORY`) and regenerating. New
tasks reshuffle when the seed, the banks or `db.json` change, so regenerate after any of them.
