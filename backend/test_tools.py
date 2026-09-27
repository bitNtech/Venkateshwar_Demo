"""Self-check for the mock tool layer: schema/method parity, fixture lookups,
ID generation, and execute_tool's never-raise contract.
"""

from __future__ import annotations

from .tools import TOOL_SCHEMAS, MockHospitalDb, execute_tool

TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]


def test_every_schema_dispatches_to_a_real_method() -> None:
    # tools.py asserts schema/method name parity at import time; this checks
    # dispatch actually works, not just that the names line up. A schema with
    # no backing method would otherwise fail silently at call time as
    # "unknown_tool", not surface at import.
    assert len(TOOL_NAMES) == 22
    db = MockHospitalDb()
    for name in TOOL_NAMES:
        result = execute_tool(db, name, {})
        assert result.get("error", "").startswith("unknown_tool:") is False


def test_lookup_patient_by_every_identifier() -> None:
    db = MockHospitalDb()
    by_mrn = execute_tool(db, "lookupPatient", {"mrn": "ARV-118342"})
    by_mobile = execute_tool(db, "lookupPatient", {"mobile": "9840721534"})
    by_appointment = execute_tool(db, "lookupPatient", {"appointment_id": "APT-77219"})
    by_bill = execute_tool(db, "lookupPatient", {"bill_number": "BILL-55210"})
    by_order = execute_tool(db, "lookupPatient", {"order_id": "LAB-33012"})

    for result in (by_mrn, by_mobile, by_appointment, by_bill, by_order):
        assert result["found"] is True
        assert result["mrn"] == "ARV-118342"
        assert result["patient_name"] == "Murugesan"


def test_lookup_patient_not_found_does_not_raise() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "lookupPatient", {"mrn": "ARV-000000"})
    assert result == {"found": False}


def test_verify_identity_dob_and_address() -> None:
    db = MockHospitalDb()
    assert execute_tool(db, "verifyIdentity", {"mrn": "ARV-118342", "dob": "1966-05-14"})["verified"] is True
    assert execute_tool(db, "verifyIdentity", {"mrn": "ARV-118342", "dob": "2000-01-01"})["verified"] is False
    assert execute_tool(db, "verifyIdentity", {"mrn": "ARV-118342", "address_readback": "T. Nagar"})["verified"] is True


def test_search_slots_known_and_unknown_department() -> None:
    db = MockHospitalDb()
    assert execute_tool(db, "searchSlots", {"department": "Cardiology"})["slots"]
    assert execute_tool(db, "searchSlots", {"department": "Neurosurgery"})["slots"] == []


def test_book_then_lookup_round_trips() -> None:
    db = MockHospitalDb()
    booked = execute_tool(
        db,
        "bookAppointment",
        {"mrn": "ARV-220981", "department": "Orthopaedics", "doctor": "Dr. Meera Krishnan", "date_time": "2026-09-10 11:00"},
    )
    assert booked["status"] == "confirmed"
    found = execute_tool(db, "lookupPatient", {"appointment_id": booked["appointment_id"]})
    assert found["mrn"] == "ARV-220981"


def test_reschedule_issues_new_id_and_retires_old() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "rescheduleAppointment", {"appointment_id": "APT-77219", "new_date_time": "2026-09-12 09:00"})
    assert result["appointment_id"] != "APT-77219"
    assert db.appointments["APT-77219"]["status"] == "rescheduled"
    assert db.appointments[result["appointment_id"]]["status"] == "confirmed"


def test_cancel_unknown_appointment_reports_not_found() -> None:
    db = MockHospitalDb()
    assert execute_tool(db, "cancelAppointment", {"appointment_id": "APT-999999"}) == {"found": False}


def test_get_report_status_never_leaks_values() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "getReportStatus", {"order_id": "LAB-33012"})
    assert result == {"order_id": "LAB-33012", "status": "ready"}
    assert "results" not in result and "value" not in result


def test_create_ticket_and_records_request_generate_unique_ids() -> None:
    db = MockHospitalDb()
    first = execute_tool(db, "createTicket", {"type": "billing_dispute"})
    second = execute_tool(db, "createTicket", {"type": "medication_query"})
    assert first["ticket_id"] != second["ticket_id"]

    rec = execute_tool(db, "logRecordsRequest", {"mrn": "ARV-118342", "record_type": "discharge_summary"})
    assert rec["status"] == "logged"


def test_register_patient_issues_new_mrn() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "registerPatient", {"full_name": "Kavitha Raman", "dob": "1985-03-20", "mobile": "9000011122"})
    assert result["status"] == "registered"
    assert result["mrn"] not in ("ARV-118342", "ARV-220981")
    assert db.patients[result["mrn"]]["name"] == "Kavitha Raman"


def test_dispatch_ambulance_escalate_transfer_hangup_are_well_shaped() -> None:
    db = MockHospitalDb()
    assert "eta_minutes" in execute_tool(db, "dispatchAmbulance", {"address": "12 MG Road"})
    assert "escalation_id" in execute_tool(db, "escalate", {"department": "billing", "level": "senior"})
    assert execute_tool(db, "transferCall", {"desk": "pharmacy"})["status"] == "transferred"
    assert execute_tool(db, "hangUp", {"reason": "completed"})["status"] == "call_ended"


def test_execute_tool_unknown_name_does_not_raise() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "deleteEverything", {})
    assert "error" in result


def test_execute_tool_bad_arguments_do_not_raise() -> None:
    db = MockHospitalDb()
    result = execute_tool(db, "bookAppointment", {"unexpected_field": "x"})
    assert "error" in result


if __name__ == "__main__":
    test_every_schema_dispatches_to_a_real_method()
    test_lookup_patient_by_every_identifier()
    test_lookup_patient_not_found_does_not_raise()
    test_verify_identity_dob_and_address()
    test_search_slots_known_and_unknown_department()
    test_book_then_lookup_round_trips()
    test_reschedule_issues_new_id_and_retires_old()
    test_cancel_unknown_appointment_reports_not_found()
    test_get_report_status_never_leaks_values()
    test_create_ticket_and_records_request_generate_unique_ids()
    test_register_patient_issues_new_mrn()
    test_dispatch_ambulance_escalate_transfer_hangup_are_well_shaped()
    test_execute_tool_unknown_name_does_not_raise()
    test_execute_tool_bad_arguments_do_not_raise()
    print("ok")
