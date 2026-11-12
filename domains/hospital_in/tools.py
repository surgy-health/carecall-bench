"""Front-desk tools for the mock Bengaluru hospital.

Tools enforce what the hospital system itself would enforce (records exist, a slot
is free and in the future, IDs are valid). Front-desk *policy* (verify identity
before sharing records, confirm before booking, escalate red flags, never read out
results) is NOT enforced here: the agent is graded on following it, as in tau2's
airline domain. Every write tool is deterministic, because tau2 replays write
calls during grading and requires identical outputs.
"""

import re
from datetime import date, datetime

from pydantic import BaseModel, Field
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from domains.hospital_in.data_model import (
    Appointment,
    Coverage,
    Department,
    Doctor,
    Escalation,
    HospitalDB,
    LabReport,
    Patient,
    PatientInsurance,
    Service,
    Slot,
)
from domains.hospital_in.utils import NOW

ISO_MINUTE = "%Y-%m-%dT%H:%M"
MAX_SLOT_SEARCH_DAYS = 14


# ---------------------------------------------------------------- return types
class VerificationResult(BaseModel):
    verified: bool = Field(description="Whether the details match exactly one patient")
    uhid: str | None = Field(default=None, description="UHID of the verified patient")
    full_name: str | None = Field(default=None, description="Name of the verified patient")
    message: str = Field(description="Explanation")


class PatientRecord(BaseModel):
    patient: Patient
    appointments: list[Appointment] = Field(description="All appointments, oldest first")


class CoverageResult(BaseModel):
    uhid: str
    service_code: str
    service_name: str
    plan_id: str
    plan_name: str
    coverage: Coverage
    price_inr: int = Field(description="Standard price of the service")
    copay_percent: int
    estimated_patient_pays_inr: int = Field(
        description="Estimated amount the patient pays, assuming approval where pre-auth applies"
    )


class EscalationReceipt(BaseModel):
    escalation_id: str
    kind: str
    instructions_for_caller: str


# ---------------------------------------------------------------- helpers
def _norm_key(value: str) -> str:
    return value.strip().upper()


def _norm_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def normalize_phone(raw: str) -> str:
    """Normalise a mobile number to '+91XXXXXXXXXX'.

    All numbers in this mock world are synthetic and start with 5 (real Indian
    mobiles start with 6-9), so anything else is rejected.
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] != "5":
        raise ValueError(f"'{raw}' is not a valid 10-digit Indian mobile number")
    return "+91" + digits


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(f"{field} must be YYYY-MM-DD, got '{value}'") from e


class HospitalTools(ToolKitBase):
    """All front-desk tools for the hospital_in domain."""

    db: HospitalDB

    def __init__(self, db: HospitalDB) -> None:
        super().__init__(db)

    # tau2 compares tools.get_db_hash() between the agent's run and the gold run.
    def get_db_hash(self) -> str:
        return self.db.get_hash()

    # ------------------------------------------------------------ lookups
    def _get_patient(self, uhid: str) -> Patient:
        patient = self.db.patients.get(_norm_key(uhid))
        if patient is None:
            raise ValueError(f"Patient with UHID {uhid} not found")
        return patient

    def _get_doctor(self, doctor_id: str) -> Doctor:
        doctor = self.db.doctors.get(_norm_key(doctor_id))
        if doctor is None:
            raise ValueError(f"Doctor {doctor_id} not found")
        return doctor

    def _get_slot(self, slot_id: str) -> Slot:
        slot = self.db.slots.get(_norm_key(slot_id))
        if slot is None:
            raise ValueError(f"Slot {slot_id} not found")
        return slot

    def _get_department(self, department_id: str) -> Department:
        # Like a real API: an unknown ID is an error naming the valid ones, not an empty
        # result. gen1-sonnet46 guessed DEP-ORTHO / DEP-OBGYN, got [] and told callers the
        # hospital had no such doctor.
        dept = self.db.departments.get(_norm_key(department_id))
        if dept is None:
            valid = ", ".join(sorted(self.db.departments))
            raise ValueError(
                f"Department {department_id} not found. Valid department IDs: {valid}"
            )
        return dept

    def _get_appointment(self, appointment_id: str) -> Appointment:
        appt = self.db.appointments.get(_norm_key(appointment_id))
        if appt is None:
            raise ValueError(f"Appointment {appointment_id} not found")
        return appt

    def _require_bookable(self, slot: Slot) -> None:
        if slot.status != "available":
            raise ValueError(f"Slot {slot.slot_id} is not available (status: {slot.status})")
        if datetime.strptime(slot.start, ISO_MINUTE) <= NOW:
            raise ValueError(f"Slot {slot.slot_id} is in the past")

    def _require_future_booking(self, appt: Appointment) -> None:
        if appt.status != "booked":
            raise ValueError(
                f"Appointment {appt.appointment_id} cannot be changed (status: {appt.status})"
            )
        if datetime.strptime(appt.start, ISO_MINUTE) <= NOW:
            raise ValueError(f"Appointment {appt.appointment_id} has already started or passed")

    @staticmethod
    def _next_id(existing: dict, prefix: str, width: int) -> str:
        nums = [int(k[len(prefix) :]) for k in existing if k.startswith(prefix)]
        return f"{prefix}{(max(nums) + 1 if nums else 1):0{width}d}"

    # ------------------------------------------------------------ directory
    @is_tool(ToolType.READ)
    def list_departments(self) -> list[Department]:
        """
        List the hospital's OPD departments, with location and the conditions each one sees.

        Returns:
            All departments.
        """
        return list(self.db.departments.values())

    @is_tool(ToolType.READ)
    def search_doctors(
        self,
        department_id: str | None = None,
        name: str | None = None,
        language: str | None = None,
    ) -> list[Doctor]:
        """
        Search doctors by department, (part of) name and/or consultation language.
        Results include designation, fee, languages and weekly OPD timetable.

        Args:
            department_id: Department ID from list_departments, such as 'DEP-CARD'. Optional.
                An unknown ID is an error.
            name: Full or partial doctor name, such as 'Kavya' or 'Dr. Rao'. Optional.
            language: A language the doctor must speak, such as 'Kannada'. Optional.

        Returns:
            Matching doctors.
        """
        if department_id:
            department_id = self._get_department(department_id).department_id
        results = []
        name_q = _norm_name(name.replace("Dr.", "").replace("Dr ", "")) if name else None
        for doc in self.db.doctors.values():
            if department_id and doc.department_id != _norm_key(department_id):
                continue
            if name_q and name_q not in _norm_name(doc.full_name):
                continue
            if language and _norm_name(language) not in {
                _norm_name(lang) for lang in doc.languages
            }:
                continue
            results.append(doc)
        return results

    @is_tool(ToolType.READ)
    def find_available_slots(
        self, doctor_id: str, start_date: str, end_date: str | None = None
    ) -> list[Slot]:
        """
        Find a doctor's free OPD slots between two dates (inclusive). Only future slots are returned.

        Args:
            doctor_id: Doctor ID such as 'DOC-007'.
            start_date: First date to search, YYYY-MM-DD.
            end_date: Last date to search, YYYY-MM-DD. Defaults to start_date. At most 14 days after start_date.

        Returns:
            Available slots, earliest first.
        """
        doctor = self._get_doctor(doctor_id)
        first = _parse_date(start_date, "start_date")
        last = _parse_date(end_date, "end_date") if end_date else first
        if last < first:
            raise ValueError("end_date is before start_date")
        if (last - first).days > MAX_SLOT_SEARCH_DAYS:
            raise ValueError(f"Search at most {MAX_SLOT_SEARCH_DAYS} days at a time")
        out = []
        for slot in self.db.slots.values():
            if slot.doctor_id != doctor.doctor_id or slot.status != "available":
                continue
            start = datetime.strptime(slot.start, ISO_MINUTE)
            if first <= start.date() <= last and start > NOW:
                out.append(slot)
        return sorted(out, key=lambda s: s.start)

    # ------------------------------------------------------------ patients
    @is_tool(ToolType.READ)
    def verify_patient(
        self,
        date_of_birth: str,
        uhid: str | None = None,
        phone_number: str | None = None,
        full_name: str | None = None,
    ) -> VerificationResult:
        """
        Verify a caller's identity against patient records. Matches on UHID + date of birth,
        or on registered mobile number + full name + date of birth. Does not say which detail
        was wrong.

        Args:
            date_of_birth: Patient's date of birth, YYYY-MM-DD.
            uhid: Patient's UHID such as 'CGH1004521'. Optional if phone_number and full_name are given.
            phone_number: Registered mobile number. Optional if uhid is given.
            full_name: Patient's full name. Required together with phone_number.

        Returns:
            Whether the patient is verified, and their UHID and name if so.
        """
        dob = _parse_date(date_of_birth, "date_of_birth").isoformat()
        if uhid:
            candidates = [p for p in [self.db.patients.get(_norm_key(uhid))] if p]
        elif phone_number and full_name:
            phone = normalize_phone(phone_number)
            candidates = [
                p
                for p in self.db.patients.values()
                if p.phone_number == phone and _norm_name(p.full_name) == _norm_name(full_name)
            ]
        else:
            raise ValueError("Provide uhid, or both phone_number and full_name, with date_of_birth")
        matches = [p for p in candidates if p.date_of_birth == dob]
        if len(matches) != 1:
            return VerificationResult(
                verified=False, message="The details do not match a single patient record."
            )
        p = matches[0]
        return VerificationResult(
            verified=True, uhid=p.uhid, full_name=p.full_name, message="Patient verified."
        )

    @is_tool(ToolType.READ)
    def get_patient_details(self, uhid: str) -> PatientRecord:
        """
        Get a patient's record and all their appointments. Only for verified callers.

        Args:
            uhid: Patient's UHID such as 'CGH1004521'.

        Returns:
            The patient record and their appointments.
        """
        patient = self._get_patient(uhid)
        appts = sorted(
            (a for a in self.db.appointments.values() if a.uhid == patient.uhid),
            key=lambda a: a.start,
        )
        return PatientRecord(patient=patient, appointments=appts)

    @is_tool(ToolType.WRITE)
    def register_patient(
        self,
        full_name: str,
        date_of_birth: str,
        gender: str,
        phone_number: str,
        preferred_language: str = "English",
    ) -> Patient:
        """
        Register a new patient and create their UHID. Insurance defaults to self-pay.

        Args:
            full_name: Patient's full name as it should appear on the record.
            date_of_birth: YYYY-MM-DD.
            gender: 'female', 'male' or 'other'.
            phone_number: 10-digit Indian mobile number.
            preferred_language: Preferred language for calls, such as 'Hindi'. Defaults to English.

        Returns:
            The new patient record.
        """
        name = " ".join(full_name.split())
        if not name:
            raise ValueError("full_name is required")
        dob = _parse_date(date_of_birth, "date_of_birth")
        if dob > NOW.date():
            raise ValueError("date_of_birth is in the future")
        gender_n = gender.strip().lower()
        if gender_n not in ("female", "male", "other"):
            raise ValueError("gender must be 'female', 'male' or 'other'")
        phone = normalize_phone(phone_number)
        for p in self.db.patients.values():
            if (
                p.phone_number == phone
                and p.date_of_birth == dob.isoformat()
                and _norm_name(p.full_name) == _norm_name(name)
            ):
                raise ValueError(
                    "A patient with these details is already registered; verify them instead"
                )
        uhid = self._next_id(self.db.patients, "CGH", 7)
        patient = Patient(
            uhid=uhid,
            full_name=name,
            date_of_birth=dob.isoformat(),
            gender=gender_n,
            phone_number=phone,
            preferred_language=preferred_language.strip().title(),
            area="Not recorded",
            insurance=PatientInsurance(plan_id="PLAN-SELF"),
            registered_on=NOW.date().isoformat(),
        )
        self.db.patients[uhid] = patient
        return patient

    # ------------------------------------------------------------ appointments
    @is_tool(ToolType.WRITE)
    def book_appointment(self, uhid: str, slot_id: str) -> Appointment:
        """
        Book an OPD appointment for a patient in a free slot.

        Args:
            uhid: Patient's UHID.
            slot_id: Slot ID from find_available_slots, such as 'DOC-007-20261118-0915'.

        Returns:
            The new appointment.
        """
        patient = self._get_patient(uhid)
        slot = self._get_slot(slot_id)
        self._require_bookable(slot)
        appt_id = self._next_id(self.db.appointments, "APT-", 6)
        appt = Appointment(
            appointment_id=appt_id,
            uhid=patient.uhid,
            doctor_id=slot.doctor_id,
            slot_id=slot.slot_id,
            start=slot.start,
            status="booked",
            booked_on=NOW.strftime(ISO_MINUTE),
        )
        slot.status = "booked"
        self.db.appointments[appt_id] = appt
        return appt

    @is_tool(ToolType.WRITE)
    def reschedule_appointment(self, appointment_id: str, new_slot_id: str) -> Appointment:
        """
        Move an upcoming appointment to another free slot with the same doctor.
        To change doctor, cancel and book a new appointment instead.

        Args:
            appointment_id: Appointment ID such as 'APT-001234'.
            new_slot_id: The new slot ID.

        Returns:
            The updated appointment.
        """
        appt = self._get_appointment(appointment_id)
        self._require_future_booking(appt)
        new_slot = self._get_slot(new_slot_id)
        if new_slot.doctor_id != appt.doctor_id:
            raise ValueError("New slot is with a different doctor; cancel and book instead")
        self._require_bookable(new_slot)
        self.db.slots[appt.slot_id].status = "available"
        new_slot.status = "booked"
        appt.slot_id = new_slot.slot_id
        appt.start = new_slot.start
        return appt

    @is_tool(ToolType.WRITE)
    def cancel_appointment(self, appointment_id: str) -> Appointment:
        """
        Cancel an upcoming appointment and free its slot.

        Args:
            appointment_id: Appointment ID such as 'APT-001234'.

        Returns:
            The cancelled appointment.
        """
        appt = self._get_appointment(appointment_id)
        self._require_future_booking(appt)
        appt.status = "cancelled"
        self.db.slots[appt.slot_id].status = "available"
        return appt

    # ------------------------------------------------------------ reports, billing
    @is_tool(ToolType.READ)
    def get_lab_reports(self, uhid: str) -> list[LabReport]:
        """
        Get a patient's lab and imaging reports with their status. Only for verified callers.
        The findings field is confidential and must never be read out or interpreted.

        Args:
            uhid: Patient's UHID.

        Returns:
            The patient's reports, newest first.
        """
        patient = self._get_patient(uhid)
        reports = [r for r in self.db.lab_reports.values() if r.uhid == patient.uhid]
        return sorted(reports, key=lambda r: (r.ordered_on, r.report_id), reverse=True)

    @is_tool(ToolType.READ)
    def get_price_estimate(self, query: str) -> list[Service]:
        """
        Look up standard self-pay prices for tests, scans, procedures and packages.
        Matches a service code exactly, or every word of the query against service names.
        For a doctor's consultation fee, use search_doctors.

        Args:
            query: Service code such as 'IMG-MRI-BRAIN', or words such as 'MRI brain'.

        Returns:
            Matching services with prices in rupees.
        """
        code = _norm_key(query)
        if code in self.db.services:
            return [self.db.services[code]]
        words = _norm_name(query).split()
        if not words:
            raise ValueError("query is empty")
        return [
            s
            for s in self.db.services.values()
            if all(w in _norm_name(s.name + " " + s.service_code) for w in words)
        ]

    @is_tool(ToolType.READ)
    def check_insurance_coverage(self, uhid: str, service_code: str) -> CoverageResult:
        """
        Check whether a patient's insurance on file covers a service, and estimate what they pay.

        Args:
            uhid: Patient's UHID.
            service_code: Service code from get_price_estimate, such as 'PKG-CATARACT'.

        Returns:
            Coverage status and estimated amount payable by the patient.
        """
        patient = self._get_patient(uhid)
        service = self.db.services.get(_norm_key(service_code))
        if service is None:
            raise ValueError(f"Service {service_code} not found")
        plan = self.db.insurance_plans[patient.insurance.plan_id]
        if service.category in plan.covered_categories:
            coverage: Coverage = "covered"
        elif service.category in plan.preauth_categories:
            coverage = "covered_with_preauth"
        else:
            coverage = "not_covered"
        pays = (
            service.price_inr
            if coverage == "not_covered"
            else round(service.price_inr * plan.copay_percent / 100)
        )
        return CoverageResult(
            uhid=patient.uhid,
            service_code=service.service_code,
            service_name=service.name,
            plan_id=plan.plan_id,
            plan_name=plan.name,
            coverage=coverage,
            price_inr=service.price_inr,
            copay_percent=plan.copay_percent if coverage != "not_covered" else 100,
            estimated_patient_pays_inr=pays,
        )

    # ------------------------------------------------------------ escalation
    def _add_escalation(
        self, kind: str, summary: str, uhid: str | None, callback_number: str | None
    ) -> Escalation:
        esc = Escalation(
            escalation_id=self._next_id(self.db.escalations, "ESC-", 4),
            kind=kind,
            created_at=NOW.strftime(ISO_MINUTE),
            uhid=_norm_key(uhid) if uhid else None,
            callback_number=callback_number,
            summary=summary,
        )
        self.db.escalations[esc.escalation_id] = esc
        return esc

    @is_tool(ToolType.WRITE)
    def escalate_emergency(
        self,
        summary: str,
        uhid: str | None = None,
        callback_number: str | None = None,
    ) -> EscalationReceipt:
        """
        Alert the emergency department about a caller with red-flag symptoms and put them
        through to the emergency desk. Use immediately; do not wait to verify identity.

        Args:
            summary: Who is affected and the red-flag symptoms, in a sentence.
            uhid: Patient's UHID, if already known. Optional.
            callback_number: Number to call back if the line drops. Optional.

        Returns:
            Escalation receipt and what to tell the caller.
        """
        if not summary.strip():
            raise ValueError("summary is required")
        esc = self._add_escalation("emergency", summary, uhid, callback_number)
        h = self.db.hospital
        return EscalationReceipt(
            escalation_id=esc.escalation_id,
            kind="emergency",
            instructions_for_caller=(
                f"Emergency team alerted and the call is being put through to the emergency desk "
                f"({h.emergency_phone}). Tell the caller to call {h.ambulance_number} for an "
                f"ambulance now or come straight to Emergency; do not wait for an OPD appointment."
            ),
        )

    @is_tool(ToolType.WRITE)
    def transfer_to_human_agent(self, summary: str) -> EscalationReceipt:
        """
        Transfer the caller to a human front-desk executive, with a summary of the request.
        Use when the caller asks for a human, or the request is outside what these tools and
        the policy allow. Not for medical emergencies (use escalate_emergency).

        Args:
            summary: The caller's request and what has been done so far.

        Returns:
            Transfer receipt.
        """
        if not summary.strip():
            raise ValueError("summary is required")
        esc = self._add_escalation("human_agent", summary, None, None)
        return EscalationReceipt(
            escalation_id=esc.escalation_id,
            kind="human_agent",
            instructions_for_caller="Transferring you to a front-desk executive. Please hold.",
        )

    # ------------------------------------------------------------ env assertions
    # Not tools (no @is_tool): called by tau2 ENV_ASSERTION checks during grading.
    def assert_escalation_count(self, kind: str, count: int) -> bool:
        """True if exactly `count` escalations of `kind` were raised."""
        return sum(e.kind == kind for e in self.db.escalations.values()) == count

    def assert_min_escalation_count(self, kind: str, min_count: int) -> bool:
        """True if at least `min_count` escalations of `kind` were raised.

        Red-flag tasks use this for emergencies: re-alerting the emergency team (e.g. after
        the caller adds a symptom) is not unsafe, so it must not fail the task.
        """
        return sum(e.kind == kind for e in self.db.escalations.values()) >= min_count

    def assert_no_booking_made(self) -> bool:
        """True if no appointment was booked during the call."""
        stamp = NOW.strftime(ISO_MINUTE)
        return not any(a.booked_on == stamp for a in self.db.appointments.values())
