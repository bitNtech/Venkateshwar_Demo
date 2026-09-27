"""Mock tool layer backing the 20 tools golden/main_prompt.txt Sec10 assumes exist.

MockHospitalDb is an in-memory stand-in for a real HMS/EHR - deliberately
simple. Its interface (one method per tool, keyword args matching the tool
schema) is what a real integration would implement later without touching
llm.py or conversation.py, per BACKEND_COMPLETION.md Sec3.1.

Fixture depth is intentionally uneven: lookup/booking/billing (the tools the
early golden flows exercise most) carry realistic seed data; every other tool
is shallow but functional - it returns a well-shaped result rather than
raising, so the LLM tool-calling loop can be exercised for all 20 flows
without a real backend. Deep, flow-accurate fixtures for every flow are the
golden-flow eval harness's job (BACKEND_COMPLETION.md Sec3.6 / roadmap item 4).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aica.tools")

_STR = {"type": "string"}
_STR_ARRAY = {"type": "array", "items": {"type": "string"}}


def _tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool_schema(
        "lookupPatient",
        "Find a patient record by whichever single identifier the caller gave.",
        {"mrn": _STR, "mobile": _STR, "appointment_id": _STR, "bill_number": _STR, "order_id": _STR},
        required=[],
    ),
    _tool_schema(
        "verifyIdentity",
        "Verify a caller's identity against the record found by lookupPatient.",
        {"mrn": _STR, "dob": _STR, "address_readback": _STR},
        required=["mrn"],
    ),
    _tool_schema(
        "searchSlots",
        "Search available appointment slots for a department or doctor.",
        {"department": _STR, "doctor": _STR, "date_range": _STR},
        required=[],
    ),
    _tool_schema(
        "bookAppointment",
        "Book a new appointment.",
        {"mrn": _STR, "department": _STR, "doctor": _STR, "date_time": _STR, "fee": {"type": "integer"}},
        required=["mrn", "department", "doctor", "date_time"],
    ),
    _tool_schema(
        "rescheduleAppointment",
        "Move an existing appointment to a new date/time.",
        {"appointment_id": _STR, "new_date_time": _STR},
        required=["appointment_id", "new_date_time"],
    ),
    _tool_schema(
        "cancelAppointment",
        "Cancel an existing appointment.",
        {"appointment_id": _STR, "reason": _STR},
        required=["appointment_id"],
    ),
    _tool_schema(
        "confirmAppointment",
        "Record the patient's confirmation status for an upcoming appointment.",
        {"appointment_id": _STR, "status": _STR},
        required=["appointment_id", "status"],
    ),
    _tool_schema(
        "raiseRefill",
        "Raise a prescription refill request.",
        {"mrn": _STR, "medications": _STR_ARRAY, "duration": _STR, "delivery": _STR},
        required=["mrn", "medications"],
    ),
    _tool_schema(
        "bookLabOrder",
        "Book a lab or imaging order.",
        {"tests": _STR_ARRAY, "collection_slot": _STR, "imaging_slot": _STR, "address": _STR},
        required=["tests"],
    ),
    _tool_schema(
        "getReportStatus",
        "Check whether a lab/imaging report is ready. Returns readiness only, never values.",
        {"order_id": _STR},
        required=["order_id"],
    ),
    _tool_schema(
        "resendReport",
        "Resend a ready report over SMS, email, or WhatsApp.",
        {"order_id": _STR, "channel": _STR, "destination": _STR},
        required=["order_id", "channel", "destination"],
    ),
    _tool_schema(
        "getReferralStatus",
        "Check the status of a referral to another centre.",
        {"referral_id": _STR, "mrn": _STR},
        required=[],
    ),
    _tool_schema(
        "getPolicyDetails",
        "Fetch insurance policy coverage details.",
        {"policy_number": _STR},
        required=["policy_number"],
    ),
    _tool_schema(
        "createPreAuth",
        "Submit a cashless pre-authorisation request to the insurer.",
        {"policy_number": _STR, "procedure": _STR, "mrn": _STR},
        required=["policy_number", "procedure"],
    ),
    _tool_schema(
        "getBill",
        "Fetch a billing record and its line items.",
        {"bill_number": _STR},
        required=["bill_number"],
    ),
    _tool_schema(
        "createTicket",
        "Open a support ticket (billing dispute, medication query, complaint, ...).",
        {"type": _STR, "details": _STR},
        required=["type"],
    ),
    _tool_schema(
        "logRecordsRequest",
        "Log a medical records request.",
        {"mrn": _STR, "record_type": _STR, "purpose": _STR},
        required=["mrn", "record_type"],
    ),
    _tool_schema(
        "registerPatient",
        "Register a new patient and issue an MRN.",
        {"full_name": _STR, "dob": _STR, "mobile": _STR, "address": _STR, "gender": _STR},
        required=["full_name", "dob", "mobile"],
    ),
    _tool_schema(
        "dispatchAmbulance",
        "Dispatch an ambulance to an address. Returns an ETA.",
        {"address": _STR},
        required=["address"],
    ),
    _tool_schema(
        "escalate",
        "Escalate to a named owner at a given severity level.",
        {"department": _STR, "level": _STR},
        required=["department", "level"],
    ),
    _tool_schema(
        "transferCall",
        "Transfer the call to another desk.",
        {"desk": _STR},
        required=["desk"],
    ),
    _tool_schema(
        "hangUp",
        "End the call.",
        {"reason": _STR},
        required=["reason"],
    ),
]


class MockHospitalDb:
    """In-memory fixture hospital DB shared across calls, like a real EHR connection."""

    def __init__(self) -> None:
        self.patients: dict[str, dict] = {
            "ARV-118342": {
                "mrn": "ARV-118342",
                "name": "Murugesan",
                "mobile": "9840721534",
                "dob": "1966-05-14",
                "address": "12, Bazaar Street, T. Nagar, Chennai",
                "gender": "male",
                "allergies": [],
            },
            "ARV-220981": {
                "mrn": "ARV-220981",
                "name": "Lakshmi Priya",
                "mobile": "9884213307",
                "dob": "1990-11-02",
                "address": "45, Kamaraj Nagar, Perungudi, Chennai",
                "gender": "female",
                "allergies": ["penicillin"],
            },
        }
        self.appointments: dict[str, dict] = {
            "APT-77219": {
                "appointment_id": "APT-77219",
                "mrn": "ARV-118342",
                "doctor": "Dr. Ramanathan",
                "department": "Cardiology",
                "date_time": "2026-09-05 17:00",
                "block": "B",
                "floor": "second",
                "room": "214",
                "fee": 800,
                "status": "confirmed",
            }
        }
        self.bills: dict[str, dict] = {
            "BILL-55210": {
                "bill_number": "BILL-55210",
                "mrn": "ARV-118342",
                "admission_dates": "2026-08-09 to 2026-08-11",
                "total": 42500,
                "items": [
                    {"date": "2026-08-09", "description": "Room rent", "amount": 9000},
                    {"date": "2026-08-09", "description": "Room rent", "amount": 9000},
                    {"date": "2026-08-10", "description": "Cardiology consult", "amount": 800},
                    {"date": "2026-08-11", "description": "Pharmacy", "amount": 4200},
                ],
                "status": "unpaid",
            }
        }
        self.lab_orders: dict[str, dict] = {
            "LAB-33012": {
                "order_id": "LAB-33012",
                "mrn": "ARV-118342",
                "tests": ["CBC", "Lipid Profile"],
                "status": "ready",
                "sent_channel": None,
            }
        }
        self.referrals: dict[str, dict] = {
            "REF-90210": {
                "referral_id": "REF-90210",
                "mrn": "ARV-118342",
                "specialty": "Nephrology",
                "target_centre": "Apollo Hospital",
                "status": "pending_signature",
                "blocker": "doctor signature pending",
            }
        }
        self.policies: dict[str, dict] = {
            "POL-4521": {
                "policy_number": "POL-4521",
                "insurer": "Star Health",
                "tpa": "MedAssist TPA",
                "room_rent_limit": 5000,
                "copay": "10%",
                "exclusions": ["cosmetic procedures"],
            }
        }
        self.slots: dict[str, list[dict]] = {
            "Cardiology": [
                {"doctor": "Dr. Ramanathan", "date_time": "2026-09-05 10:30"},
                {"doctor": "Dr. Ramanathan", "date_time": "2026-09-06 17:00"},
            ],
            "Orthopaedics": [
                {"doctor": "Dr. Meera Krishnan", "date_time": "2026-09-05 11:00"},
            ],
            "General Medicine": [
                {"doctor": "Dr. Anand Kumar", "date_time": "2026-09-05 09:30"},
                {"doctor": "Dr. Anand Kumar", "date_time": "2026-09-05 18:00"},
            ],
        }
        self.tickets: dict[str, dict] = {}
        self.records_requests: dict[str, dict] = {}
        self.preauths: dict[str, dict] = {}
        self._counters: dict[str, int] = {}

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{100000 + self._counters[prefix]}"

    # --- identity & scheduling (deep fixtures) ---

    def lookup_patient(
        self,
        *,
        mrn: str | None = None,
        mobile: str | None = None,
        appointment_id: str | None = None,
        bill_number: str | None = None,
        order_id: str | None = None,
    ) -> dict:
        patient = None
        if mrn:
            patient = self.patients.get(mrn)
        elif mobile:
            patient = next((p for p in self.patients.values() if p["mobile"] == mobile), None)
        elif appointment_id:
            appt = self.appointments.get(appointment_id)
            patient = self.patients.get(appt["mrn"]) if appt else None
        elif bill_number:
            bill = self.bills.get(bill_number)
            patient = self.patients.get(bill["mrn"]) if bill else None
        elif order_id:
            order = self.lab_orders.get(order_id)
            patient = self.patients.get(order["mrn"]) if order else None

        if patient is None:
            return {"found": False}
        result = {
            "found": True,
            "mrn": patient["mrn"],
            "patient_name": patient["name"],
            "caller_mobile": patient["mobile"],
            "address": patient["address"],
        }
        # Surface the patient's appointment_id when there is exactly one -
        # otherwise a caller who never states an ID by name (nobody says
        # "cancel APT-77219", they say "cancel my dermatology appointment")
        # leaves cancelAppointment/rescheduleAppointment with nothing to call.
        # Omitted, not guessed, when a patient has zero or several - picking
        # among several would be inventing which one the caller means.
        matches = [appt_id for appt_id, appt in self.appointments.items() if appt["mrn"] == patient["mrn"]]
        if len(matches) == 1:
            result["appointment_id"] = matches[0]
        return result

    def verify_identity(
        self, *, mrn: str, dob: str | None = None, address_readback: str | None = None
    ) -> dict:
        patient = self.patients.get(mrn)
        if patient is None:
            return {"verified": False, "reason": "no_record"}
        if dob is not None:
            return {"verified": dob == patient["dob"]}
        if address_readback is not None:
            return {"verified": address_readback.strip().lower() in patient["address"].lower()}
        return {"verified": False, "reason": "no_verifier_supplied"}

    def search_slots(
        self, *, department: str | None = None, doctor: str | None = None, date_range: str | None = None
    ) -> dict:
        candidates = self.slots.get(department, []) if department else []
        if doctor:
            matching = [slot for slot in candidates if slot["doctor"] == doctor]
            candidates = matching or candidates
        return {"slots": candidates}

    def book_appointment(
        self, *, mrn: str, department: str, doctor: str, date_time: str, fee: int = 800
    ) -> dict:
        appointment_id = self._next_id("APT")
        self.appointments[appointment_id] = {
            "appointment_id": appointment_id,
            "mrn": mrn,
            "doctor": doctor,
            "department": department,
            "date_time": date_time,
            "status": "confirmed",
            "fee": fee,
        }
        return {"appointment_id": appointment_id, "status": "confirmed"}

    def reschedule_appointment(self, *, appointment_id: str, new_date_time: str) -> dict:
        old = self.appointments.get(appointment_id)
        if old is None:
            return {"found": False}
        old["status"] = "rescheduled"
        new_id = self._next_id("APT")
        self.appointments[new_id] = {**old, "appointment_id": new_id, "date_time": new_date_time, "status": "confirmed"}
        return {"appointment_id": new_id, "status": "confirmed"}

    def cancel_appointment(self, *, appointment_id: str, reason: str | None = None) -> dict:
        appt = self.appointments.get(appointment_id)
        if appt is None:
            return {"found": False}
        appt["status"] = "cancelled"
        return {"cancellation_reference": self._next_id("CANC"), "status": "cancelled"}

    def confirm_appointment(self, *, appointment_id: str, status: str) -> dict:
        appt = self.appointments.get(appointment_id)
        if appt is None:
            return {"found": False}
        appt["confirmation_status"] = status
        return {"appointment_id": appointment_id, "confirmation_status": status}

    def get_bill(self, *, bill_number: str) -> dict:
        bill = self.bills.get(bill_number)
        if bill is None:
            return {"found": False}
        return dict(bill)

    # --- shallow but functional (well-shaped results, no realistic depth) ---

    def raise_refill(
        self, *, mrn: str, medications: list[str], duration: str | None = None, delivery: str | None = None
    ) -> dict:
        needs_approval = any("insulin" in med.lower() or "metformin" in med.lower() for med in medications)
        return {
            "refill_reference": self._next_id("REFILL"),
            "approval_needed": needs_approval,
            "status": "pending_approval" if needs_approval else "approved",
        }

    def book_lab_order(
        self,
        *,
        tests: list[str],
        collection_slot: str | None = None,
        imaging_slot: str | None = None,
        address: str | None = None,
    ) -> dict:
        order_id = self._next_id("LAB")
        self.lab_orders[order_id] = {"order_id": order_id, "tests": tests, "status": "pending", "sent_channel": None}
        return {"order_id": order_id, "status": "pending"}

    def get_report_status(self, *, order_id: str) -> dict:
        order = self.lab_orders.get(order_id)
        if order is None:
            return {"found": False}
        # Readiness only - never the values themselves (golden/main_prompt.txt Sec8, flow 08).
        return {"order_id": order_id, "status": order["status"]}

    def resend_report(self, *, order_id: str, channel: str, destination: str) -> dict:
        order = self.lab_orders.get(order_id)
        if order is None:
            return {"found": False}
        order["sent_channel"] = channel
        return {"order_id": order_id, "sent_channel": channel, "destination": destination}

    def get_referral_status(self, *, referral_id: str | None = None, mrn: str | None = None) -> dict:
        referral = (
            self.referrals.get(referral_id)
            if referral_id
            else next((r for r in self.referrals.values() if r["mrn"] == mrn), None)
        )
        if referral is None:
            return {"found": False}
        return dict(referral)

    def get_policy_details(self, *, policy_number: str) -> dict:
        policy = self.policies.get(policy_number)
        if policy is None:
            return {"found": False}
        return dict(policy)

    def create_pre_auth(self, *, policy_number: str, procedure: str, mrn: str | None = None) -> dict:
        preauth_id = self._next_id("PREAUTH")
        self.preauths[preauth_id] = {"preauth_reference": preauth_id, "policy_number": policy_number, "procedure": procedure}
        return {"preauth_reference": preauth_id, "status": "submitted"}

    def create_ticket(self, *, type: str, details: str | None = None) -> dict:
        ticket_id = self._next_id("TCK")
        self.tickets[ticket_id] = {"ticket_id": ticket_id, "type": type, "details": details, "status": "open"}
        return {"ticket_id": ticket_id, "status": "open"}

    def log_records_request(self, *, mrn: str, record_type: str, purpose: str | None = None) -> dict:
        request_id = self._next_id("REC")
        self.records_requests[request_id] = {"request_id": request_id, "mrn": mrn, "record_type": record_type}
        return {"request_id": request_id, "status": "logged"}

    def register_patient(
        self, *, full_name: str, dob: str, mobile: str, address: str | None = None, gender: str | None = None
    ) -> dict:
        mrn = self._next_id("ARV")
        self.patients[mrn] = {
            "mrn": mrn,
            "name": full_name,
            "mobile": mobile,
            "dob": dob,
            "address": address or "",
            "gender": gender,
            "allergies": [],
        }
        return {"mrn": mrn, "status": "registered"}

    def dispatch_ambulance(self, *, address: str) -> dict:
        return {"dispatch_id": self._next_id("AMB"), "eta_minutes": 12}

    def escalate(self, *, department: str, level: str) -> dict:
        return {"escalation_id": self._next_id("ESC"), "owner": "Duty Manager", "department": department, "level": level}

    def transfer_call(self, *, desk: str) -> dict:
        return {"transferred_to": desk, "status": "transferred"}

    def hang_up(self, *, reason: str) -> dict:
        return {"status": "call_ended", "reason": reason}


_TOOL_METHODS: dict[str, str] = {
    "lookupPatient": "lookup_patient",
    "verifyIdentity": "verify_identity",
    "searchSlots": "search_slots",
    "bookAppointment": "book_appointment",
    "rescheduleAppointment": "reschedule_appointment",
    "cancelAppointment": "cancel_appointment",
    "confirmAppointment": "confirm_appointment",
    "raiseRefill": "raise_refill",
    "bookLabOrder": "book_lab_order",
    "getReportStatus": "get_report_status",
    "resendReport": "resend_report",
    "getReferralStatus": "get_referral_status",
    "getPolicyDetails": "get_policy_details",
    "createPreAuth": "create_pre_auth",
    "getBill": "get_bill",
    "createTicket": "create_ticket",
    "logRecordsRequest": "log_records_request",
    "registerPatient": "register_patient",
    "dispatchAmbulance": "dispatch_ambulance",
    "escalate": "escalate",
    "transferCall": "transfer_call",
    "hangUp": "hang_up",
}

assert {schema["function"]["name"] for schema in TOOL_SCHEMAS} == set(_TOOL_METHODS)


def execute_tool(db: MockHospitalDb, name: str, arguments: dict) -> dict:
    """Run one LLM-issued tool call against the mock DB.

    Never raises: a bad/unknown call must reach the LLM as a tool result it
    can react to (per golden/main_prompt.txt Sec10's grounding rules - "if a
    tool fails... say so honestly"), not crash the call.
    """
    method_name = _TOOL_METHODS.get(name)
    if method_name is None:
        logger.error("unknown tool call: %s", name)
        return {"error": f"unknown_tool:{name}"}

    method = getattr(db, method_name)
    try:
        return method(**arguments)
    except Exception as error:
        logger.exception("tool %s failed with arguments %r", name, arguments)
        return {"error": str(error)}
