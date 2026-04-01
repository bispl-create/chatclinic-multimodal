"""FHIR Browser tool — parse FHIR JSON, XML, or NDJSON bundles into
patient, observation, medication, allergy, vital, timeline, lab, and
care-team artifacts for the FHIR Browser studio card.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app.models import FhirSourceResponse
from app.services.tool_runner import discover_tools

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _find_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _attr_value(element: ET.Element | None) -> str:
    if element is None:
        return "n/a"
    return str(element.attrib.get("value", "n/a"))


# ---------------------------------------------------------------------------
# JSON helpers — resolve resources from Bundle
# ---------------------------------------------------------------------------

def _resolve_fhir_resources_json(payload: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    if str(payload.get("resourceType")) == resource_type:
        return [payload]
    if str(payload.get("resourceType")) != "Bundle":
        return []
    resources: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict) and str(resource.get("resourceType")) == resource_type:
            resources.append(resource)
    return resources


def _first_fhir_patient_json(payload: dict[str, Any]) -> dict[str, Any]:
    patient_resources = _resolve_fhir_resources_json(payload, "Patient")
    if patient_resources:
        return patient_resources[0]
    return payload


def _fhir_code_display_json(resource: dict[str, Any], field_name: str = "code") -> str:
    node = resource.get(field_name)
    if not isinstance(node, dict):
        return "n/a"
    coding = node.get("coding") or []
    if coding and isinstance(coding[0], dict):
        return str(coding[0].get("display") or coding[0].get("code") or "n/a")
    return str(node.get("text") or "n/a")


def _observation_numeric_json(obs: dict[str, Any]) -> tuple[float | None, str, float | None, float | None]:
    quantity = obs.get("valueQuantity")
    if not isinstance(quantity, dict):
        return None, "n/a", None, None
    try:
        numeric = float(quantity.get("value"))
    except Exception:
        numeric = None
    unit = str(quantity.get("unit") or quantity.get("code") or "n/a")
    low = None
    high = None
    ranges = obs.get("referenceRange") or []
    if ranges and isinstance(ranges[0], dict):
        low_node = ranges[0].get("low") or {}
        high_node = ranges[0].get("high") or {}
        try:
            low = float(low_node.get("value")) if isinstance(low_node, dict) and low_node.get("value") is not None else None
        except Exception:
            low = None
        try:
            high = float(high_node.get("value")) if isinstance(high_node, dict) and high_node.get("value") is not None else None
        except Exception:
            high = None
    return numeric, unit, low, high


def _observation_category_json(obs: dict[str, Any]) -> str:
    categories = obs.get("category") or []
    if not isinstance(categories, list):
        return "n/a"
    for category in categories:
        if not isinstance(category, dict):
            continue
        coding = category.get("coding") or []
        if coding and isinstance(coding[0], dict):
            return str(coding[0].get("code") or coding[0].get("display") or "n/a")
    return "n/a"


def _blood_pressure_value_json(obs: dict[str, Any]) -> str:
    components = obs.get("component") or []
    systolic = None
    diastolic = None
    unit = "mmHg"
    for component in components:
        if not isinstance(component, dict):
            continue
        label = _fhir_code_display_json(component)
        quantity = component.get("valueQuantity") if isinstance(component.get("valueQuantity"), dict) else {}
        value = quantity.get("value")
        unit = str(quantity.get("unit") or unit)
        if "systolic" in label.lower():
            systolic = value
        if "diastolic" in label.lower():
            diastolic = value
    if systolic is not None or diastolic is not None:
        return f"{systolic or '?'} / {diastolic or '?'} {unit}".strip()
    return "n/a"


# ---------------------------------------------------------------------------
# Patient browser (JSON)
# ---------------------------------------------------------------------------

def _patient_browser_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    names = payload.get("name") or []
    full_name = "n/a"
    if names and isinstance(names, list) and isinstance(names[0], dict):
        given = names[0].get("given") or []
        family = names[0].get("family") or ""
        parts = []
        if isinstance(given, list):
            parts.extend(str(item) for item in given)
        if family:
            parts.append(str(family))
        full_name = " ".join(part for part in parts if part).strip() or "n/a"
    identifiers = []
    for item in payload.get("identifier") or []:
        if not isinstance(item, dict):
            continue
        identifiers.append({"system": str(item.get("system", "n/a")), "value": str(item.get("value", "n/a")), "use": str(item.get("use", "n/a"))})
    telecom = []
    for item in payload.get("telecom") or []:
        if not isinstance(item, dict):
            continue
        telecom.append({"system": str(item.get("system", "n/a")), "value": str(item.get("value", "n/a")), "use": str(item.get("use", "n/a"))})
    addresses = []
    for item in payload.get("address") or []:
        if not isinstance(item, dict):
            continue
        line = item.get("line") or []
        line_text = ", ".join(str(part) for part in line) if isinstance(line, list) else str(line)
        addresses.append({"line": line_text or "n/a", "city": str(item.get("city", "n/a")), "state": str(item.get("state", "n/a")), "postalCode": str(item.get("postalCode", "n/a")), "country": str(item.get("country", "n/a"))})
    return {
        "resource_type": str(payload.get("resourceType", "Unknown")),
        "id": str(payload.get("id", "n/a")),
        "full_name": full_name,
        "gender": str(payload.get("gender", "n/a")),
        "birth_date": str(payload.get("birthDate", "n/a")),
        "active": str(payload.get("active", "n/a")),
        "identifiers": identifiers,
        "telecom": telecom,
        "addresses": addresses,
        "managing_organization": str(((payload.get("managingOrganization") or {}).get("reference")) or "n/a"),
    }


# ---------------------------------------------------------------------------
# Patient browser (XML)
# ---------------------------------------------------------------------------

def _patient_browser_from_xml(element: ET.Element) -> dict[str, Any]:
    name_el = _find_child(element, "name")
    full_name = "n/a"
    if name_el is not None:
        given_els = _find_children(name_el, "given")
        family_el = _find_child(name_el, "family")
        parts = [_attr_value(g) for g in given_els if _attr_value(g) != "n/a"]
        family = _attr_value(family_el)
        if family != "n/a":
            parts.append(family)
        full_name = " ".join(parts).strip() or "n/a"
    identifiers: list[dict[str, Any]] = []
    for ident in _find_children(element, "identifier"):
        system_el = _find_child(ident, "system")
        value_el = _find_child(ident, "value")
        use_el = _find_child(ident, "use")
        identifiers.append({
            "system": _attr_value(system_el),
            "value": _attr_value(value_el),
            "use": _attr_value(use_el),
        })
    telecom: list[dict[str, Any]] = []
    for tel in _find_children(element, "telecom"):
        system_el = _find_child(tel, "system")
        value_el = _find_child(tel, "value")
        use_el = _find_child(tel, "use")
        telecom.append({
            "system": _attr_value(system_el),
            "value": _attr_value(value_el),
            "use": _attr_value(use_el),
        })
    addresses: list[dict[str, Any]] = []
    for addr in _find_children(element, "address"):
        line_els = _find_children(addr, "line")
        line_text = ", ".join(_attr_value(l) for l in line_els if _attr_value(l) != "n/a") or "n/a"
        addresses.append({
            "line": line_text,
            "city": _attr_value(_find_child(addr, "city")),
            "state": _attr_value(_find_child(addr, "state")),
            "postalCode": _attr_value(_find_child(addr, "postalCode")),
            "country": _attr_value(_find_child(addr, "country")),
        })
    id_el = _find_child(element, "id")
    gender_el = _find_child(element, "gender")
    birth_el = _find_child(element, "birthDate")
    active_el = _find_child(element, "active")
    managing_org_el = _find_child(element, "managingOrganization")
    org_ref_el = _find_child(managing_org_el, "reference") if managing_org_el is not None else None
    return {
        "resource_type": _local_name(element.tag),
        "id": _attr_value(id_el),
        "full_name": full_name,
        "gender": _attr_value(gender_el),
        "birth_date": _attr_value(birth_el),
        "active": _attr_value(active_el),
        "identifiers": identifiers,
        "telecom": telecom,
        "addresses": addresses,
        "managing_organization": _attr_value(org_ref_el),
    }


# ---------------------------------------------------------------------------
# Observation viewer (JSON)
# ---------------------------------------------------------------------------

def _observation_viewer_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    observations = _resolve_fhir_resources_json(payload, "Observation")
    items: list[dict[str, Any]] = []
    for obs in observations[:24]:
        code = _fhir_code_display_json(obs)
        value = "n/a"
        numeric_value, unit, ref_low, ref_high = _observation_numeric_json(obs)
        if numeric_value is not None:
            value = f"{numeric_value} {unit}".strip()
        elif "valueString" in obs:
            value = str(obs.get("valueString"))
        elif "valueCodeableConcept" in obs and isinstance(obs.get("valueCodeableConcept"), dict):
            concepts = (obs.get("valueCodeableConcept") or {}).get("coding") or []
            if concepts and isinstance(concepts[0], dict):
                value = str(concepts[0].get("display") or concepts[0].get("code") or "n/a")
        elif (obs.get("component") or []) and "blood pressure" in code.lower():
            value = _blood_pressure_value_json(obs)
        items.append({
            "code": code, "value": value, "status": str(obs.get("status", "n/a")),
            "effective": str(obs.get("effectiveDateTime", obs.get("issued", "n/a"))),
            "category": _observation_category_json(obs),
            "numeric_value": numeric_value, "unit": unit,
            "reference_low": ref_low, "reference_high": ref_high,
        })
    return {"count": len(observations), "items": items}


# ---------------------------------------------------------------------------
# Observation viewer (XML)
# ---------------------------------------------------------------------------

def _fhir_code_display_xml(element: ET.Element, field_name: str = "code") -> str:
    code_el = _find_child(element, field_name)
    if code_el is None:
        return "n/a"
    coding_el = _find_child(code_el, "coding")
    if coding_el is not None:
        display_el = _find_child(coding_el, "display")
        code_val_el = _find_child(coding_el, "code")
        display = _attr_value(display_el)
        if display != "n/a":
            return display
        return _attr_value(code_val_el)
    text_el = _find_child(code_el, "text")
    return _attr_value(text_el)


def _observation_numeric_xml(obs: ET.Element) -> tuple[float | None, str, float | None, float | None]:
    vq = _find_child(obs, "valueQuantity")
    if vq is None:
        return None, "n/a", None, None
    value_el = _find_child(vq, "value")
    unit_el = _find_child(vq, "unit")
    code_el = _find_child(vq, "code")
    try:
        numeric = float(_attr_value(value_el))
    except Exception:
        numeric = None
    unit_str = _attr_value(unit_el)
    if unit_str == "n/a":
        unit_str = _attr_value(code_el)
    low: float | None = None
    high: float | None = None
    rr = _find_child(obs, "referenceRange")
    if rr is not None:
        low_el = _find_child(rr, "low")
        high_el = _find_child(rr, "high")
        if low_el is not None:
            try:
                low = float(_attr_value(_find_child(low_el, "value")))
            except Exception:
                pass
        if high_el is not None:
            try:
                high = float(_attr_value(_find_child(high_el, "value")))
            except Exception:
                pass
    return numeric, unit_str, low, high


def _observation_category_xml(obs: ET.Element) -> str:
    for cat in _find_children(obs, "category"):
        coding_el = _find_child(cat, "coding")
        if coding_el is not None:
            code_el = _find_child(coding_el, "code")
            display_el = _find_child(coding_el, "display")
            val = _attr_value(code_el)
            if val != "n/a":
                return val
            return _attr_value(display_el)
    return "n/a"


def _observation_viewer_from_xml(root: ET.Element) -> dict[str, Any]:
    observations = _resolve_fhir_resources_xml(root, "Observation")
    items: list[dict[str, Any]] = []
    for obs in observations[:24]:
        code = _fhir_code_display_xml(obs)
        value = "n/a"
        numeric_value, unit, ref_low, ref_high = _observation_numeric_xml(obs)
        if numeric_value is not None:
            value = f"{numeric_value} {unit}".strip()
        else:
            vs = _find_child(obs, "valueString")
            if vs is not None:
                value = _attr_value(vs)
            else:
                vcc = _find_child(obs, "valueCodeableConcept")
                if vcc is not None:
                    coding = _find_child(vcc, "coding")
                    if coding is not None:
                        display = _find_child(coding, "display")
                        code_el = _find_child(coding, "code")
                        val = _attr_value(display)
                        if val == "n/a":
                            val = _attr_value(code_el)
                        value = val
        status_el = _find_child(obs, "status")
        eff_el = _find_child(obs, "effectiveDateTime")
        issued_el = _find_child(obs, "issued")
        items.append({
            "code": code,
            "value": value,
            "status": _attr_value(status_el),
            "effective": _attr_value(eff_el) if _attr_value(eff_el) != "n/a" else _attr_value(issued_el),
            "category": _observation_category_xml(obs),
            "numeric_value": numeric_value,
            "unit": unit,
            "reference_low": ref_low,
            "reference_high": ref_high,
        })
    return {"count": len(observations), "items": items}


# ---------------------------------------------------------------------------
# Medication timeline (JSON)
# ---------------------------------------------------------------------------

def _medication_timeline_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    meds = _resolve_fhir_resources_json(payload, "MedicationRequest") + _resolve_fhir_resources_json(payload, "MedicationStatement")
    items: list[dict[str, Any]] = []
    for med in meds[:24]:
        med_name = "n/a"
        concept = med.get("medicationCodeableConcept")
        if isinstance(concept, dict):
            coding = concept.get("coding") or []
            if coding and isinstance(coding[0], dict):
                med_name = str(coding[0].get("display") or coding[0].get("code") or "n/a")
            elif concept.get("text"):
                med_name = str(concept.get("text"))
        status = str(med.get("status", "n/a"))
        intent = str(med.get("intent", "n/a"))
        authored = str(med.get("authoredOn", med.get("effectiveDateTime", "n/a")))
        dosage = "n/a"
        dosage_list = med.get("dosageInstruction") or []
        if dosage_list and isinstance(dosage_list[0], dict):
            dosage = str(dosage_list[0].get("text") or "n/a")
        start = str(med.get("authoredOn") or ((med.get("effectivePeriod") or {}).get("start") if isinstance(med.get("effectivePeriod"), dict) else None) or med.get("effectiveDateTime") or "n/a")
        end = str(((med.get("dispenseRequest") or {}).get("validityPeriod") or {}).get("end") if isinstance((med.get("dispenseRequest") or {}).get("validityPeriod"), dict) else ((med.get("effectivePeriod") or {}).get("end") if isinstance(med.get("effectivePeriod"), dict) else "n/a"))
        duration_days = None
        duration = (med.get("dispenseRequest") or {}).get("expectedSupplyDuration") if isinstance(med.get("dispenseRequest"), dict) else None
        if isinstance(duration, dict):
            try:
                duration_days = float(duration.get("value"))
            except Exception:
                duration_days = None
        current = status.lower() in {"active", "in-progress", "on-hold"}
        items.append({"medication": med_name, "status": status, "intent": intent, "date": authored, "dosage": dosage, "start": start, "end": end, "duration_days": duration_days, "current": current})
    return {"count": len(meds), "items": items}


# ---------------------------------------------------------------------------
# Medication timeline (XML)
# ---------------------------------------------------------------------------

def _medication_timeline_from_xml(root: ET.Element) -> dict[str, Any]:
    meds = _resolve_fhir_resources_xml(root, "MedicationRequest") + _resolve_fhir_resources_xml(root, "MedicationStatement")
    items: list[dict[str, Any]] = []
    for med in meds[:24]:
        med_name = "n/a"
        concept = _find_child(med, "medicationCodeableConcept")
        if concept is not None:
            coding = _find_child(concept, "coding")
            if coding is not None:
                display = _find_child(coding, "display")
                code_el = _find_child(coding, "code")
                val = _attr_value(display)
                if val == "n/a":
                    val = _attr_value(code_el)
                med_name = val
            else:
                text_el = _find_child(concept, "text")
                val = _attr_value(text_el)
                if val != "n/a":
                    med_name = val
        status_el = _find_child(med, "status")
        intent_el = _find_child(med, "intent")
        authored_el = _find_child(med, "authoredOn")
        eff_el = _find_child(med, "effectiveDateTime")
        status = _attr_value(status_el)
        intent = _attr_value(intent_el)
        authored = _attr_value(authored_el) if _attr_value(authored_el) != "n/a" else _attr_value(eff_el)
        dosage = "n/a"
        dosage_inst = _find_child(med, "dosageInstruction")
        if dosage_inst is not None:
            text_el = _find_child(dosage_inst, "text")
            val = _attr_value(text_el)
            if val != "n/a":
                dosage = val
        start = _attr_value(authored_el)
        if start == "n/a":
            ep = _find_child(med, "effectivePeriod")
            if ep is not None:
                start = _attr_value(_find_child(ep, "start"))
            if start == "n/a":
                start = _attr_value(eff_el)
        end = "n/a"
        dr = _find_child(med, "dispenseRequest")
        if dr is not None:
            vp = _find_child(dr, "validityPeriod")
            if vp is not None:
                end = _attr_value(_find_child(vp, "end"))
        if end == "n/a":
            ep = _find_child(med, "effectivePeriod")
            if ep is not None:
                end = _attr_value(_find_child(ep, "end"))
        duration_days: float | None = None
        if dr is not None:
            esd = _find_child(dr, "expectedSupplyDuration")
            if esd is not None:
                try:
                    duration_days = float(_attr_value(_find_child(esd, "value")))
                except Exception:
                    pass
        current = status.lower() in {"active", "in-progress", "on-hold"}
        items.append({"medication": med_name, "status": status, "intent": intent, "date": authored, "dosage": dosage, "start": start, "end": end, "duration_days": duration_days, "current": current})
    return {"count": len(meds), "items": items}


# ---------------------------------------------------------------------------
# Allergy (JSON)
# ---------------------------------------------------------------------------

def _allergy_summary_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    allergies = _resolve_fhir_resources_json(payload, "AllergyIntolerance")
    items: list[dict[str, Any]] = []
    for allergy in allergies[:12]:
        items.append({"substance": _fhir_code_display_json(allergy), "criticality": str(allergy.get("criticality", "n/a")), "clinical_status": _fhir_code_display_json(allergy, "clinicalStatus"), "verification_status": _fhir_code_display_json(allergy, "verificationStatus")})
    return {"count": len(allergies), "items": items}


# ---------------------------------------------------------------------------
# Allergy (XML)
# ---------------------------------------------------------------------------

def _allergy_summary_from_xml(root: ET.Element) -> dict[str, Any]:
    allergies = _resolve_fhir_resources_xml(root, "AllergyIntolerance")
    items: list[dict[str, Any]] = []
    for allergy in allergies[:12]:
        substance = _fhir_code_display_xml(allergy)
        criticality_el = _find_child(allergy, "criticality")
        items.append({
            "substance": substance,
            "criticality": _attr_value(criticality_el),
            "clinical_status": _fhir_code_display_xml(allergy, "clinicalStatus"),
            "verification_status": _fhir_code_display_xml(allergy, "verificationStatus"),
        })
    return {"count": len(allergies), "items": items}


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------

def _vital_summary_from_observations(observation_artifact: dict[str, Any]) -> dict[str, Any]:
    wanted = [("blood pressure", "Blood pressure"), ("body weight", "Weight"), ("glucose", "Glucose"), ("heart rate", "Heart rate"), ("temperature", "Temperature"), ("oxygen saturation", "O2 saturation")]
    latest: list[dict[str, Any]] = []
    items = observation_artifact.get("items") or []
    for needle, label in wanted:
        matches = [item for item in items if needle in str(item.get("code", "")).lower()]
        if matches:
            latest.append({"label": label, "value": matches[0].get("value", "n/a"), "effective": matches[0].get("effective", "n/a"), "status": matches[0].get("status", "n/a")})
    return {"items": latest}


# ---------------------------------------------------------------------------
# Timeline events (JSON)
# ---------------------------------------------------------------------------

def _timeline_events_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    encounters = _resolve_fhir_resources_json(payload, "Encounter")
    procedures = _resolve_fhir_resources_json(payload, "Procedure")
    events: list[dict[str, Any]] = []
    for encounter in encounters[:12]:
        period = encounter.get("period") if isinstance(encounter.get("period"), dict) else {}
        events.append({"type": "Encounter", "label": _fhir_code_display_json(encounter, "type"), "start": str(period.get("start") or encounter.get("actualPeriod", {}).get("start") if isinstance(encounter.get("actualPeriod"), dict) else period.get("start") or "n/a"), "end": str(period.get("end") or encounter.get("actualPeriod", {}).get("end") if isinstance(encounter.get("actualPeriod"), dict) else period.get("end") or "n/a"), "status": str(encounter.get("status", "n/a"))})
    for procedure in procedures[:12]:
        performed = procedure.get("performedPeriod") if isinstance(procedure.get("performedPeriod"), dict) else {}
        events.append({"type": "Procedure", "label": _fhir_code_display_json(procedure), "start": str(performed.get("start") or procedure.get("performedDateTime") or "n/a"), "end": str(performed.get("end") or "n/a"), "status": str(procedure.get("status", "n/a"))})
    return {"events": events}


# ---------------------------------------------------------------------------
# Timeline events (XML)
# ---------------------------------------------------------------------------

def _timeline_events_from_xml(root: ET.Element) -> dict[str, Any]:
    encounters = _resolve_fhir_resources_xml(root, "Encounter")
    procedures = _resolve_fhir_resources_xml(root, "Procedure")
    events: list[dict[str, Any]] = []
    for enc in encounters[:12]:
        period = _find_child(enc, "period")
        start = "n/a"
        end = "n/a"
        if period is not None:
            start = _attr_value(_find_child(period, "start"))
            end = _attr_value(_find_child(period, "end"))
        if start == "n/a":
            ap = _find_child(enc, "actualPeriod")
            if ap is not None:
                start = _attr_value(_find_child(ap, "start"))
                if end == "n/a":
                    end = _attr_value(_find_child(ap, "end"))
        status_el = _find_child(enc, "status")
        events.append({
            "type": "Encounter",
            "label": _fhir_code_display_xml(enc, "type"),
            "start": start,
            "end": end,
            "status": _attr_value(status_el),
        })
    for proc in procedures[:12]:
        pp = _find_child(proc, "performedPeriod")
        start = "n/a"
        end = "n/a"
        if pp is not None:
            start = _attr_value(_find_child(pp, "start"))
            end = _attr_value(_find_child(pp, "end"))
        if start == "n/a":
            pdt = _find_child(proc, "performedDateTime")
            start = _attr_value(pdt)
        status_el = _find_child(proc, "status")
        events.append({
            "type": "Procedure",
            "label": _fhir_code_display_xml(proc),
            "start": start,
            "end": end,
            "status": _attr_value(status_el),
        })
    return {"events": events}


# ---------------------------------------------------------------------------
# Lab trends
# ---------------------------------------------------------------------------

def _lab_trends_from_observations(observation_artifact: dict[str, Any]) -> dict[str, Any]:
    series_map: dict[str, list[dict[str, Any]]] = {}
    for item in observation_artifact.get("items") or []:
        numeric_value = item.get("numeric_value")
        if numeric_value is None:
            continue
        key = str(item.get("code") or "Unknown")
        series_map.setdefault(key, []).append({"date": str(item.get("effective", "n/a")), "value": numeric_value, "unit": str(item.get("unit", "n/a")), "low": item.get("reference_low"), "high": item.get("reference_high")})
    series = [{"label": label, "points": points[:16]} for label, points in list(series_map.items())[:6]]
    latest = []
    for item in series[:4]:
        point = item["points"][0] if item["points"] else {}
        latest.append({"label": item["label"], "value": point.get("value", "n/a"), "unit": point.get("unit", "n/a"), "low": point.get("low"), "high": point.get("high")})
    return {"series": series, "latest": latest}


# ---------------------------------------------------------------------------
# Care team (JSON)
# ---------------------------------------------------------------------------

def _care_team_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    practitioners = _resolve_fhir_resources_json(payload, "Practitioner")
    organizations = _resolve_fhir_resources_json(payload, "Organization")
    practitioner_cards: list[dict[str, Any]] = []
    for practitioner in practitioners[:12]:
        name = _patient_browser_from_json(practitioner).get("full_name", "n/a")
        telecom = practitioner.get("telecom") or []
        practitioner_cards.append({"name": name, "role": "Practitioner", "contact": str((telecom[0] or {}).get("value")) if telecom and isinstance(telecom[0], dict) else "n/a", "organization": "n/a"})
    organization_cards: list[dict[str, Any]] = []
    for org in organizations[:12]:
        telecom = org.get("telecom") or []
        organization_cards.append({"name": str(org.get("name", "n/a")), "contact": str((telecom[0] or {}).get("value")) if telecom and isinstance(telecom[0], dict) else "n/a"})
    return {"practitioners": practitioner_cards, "organizations": organization_cards}


# ---------------------------------------------------------------------------
# Care team (XML)
# ---------------------------------------------------------------------------

def _care_team_from_xml(root: ET.Element) -> dict[str, Any]:
    practitioners = _resolve_fhir_resources_xml(root, "Practitioner")
    organizations = _resolve_fhir_resources_xml(root, "Organization")
    practitioner_cards: list[dict[str, Any]] = []
    for prac in practitioners[:12]:
        name = _patient_browser_from_xml(prac).get("full_name", "n/a")
        telecom_els = _find_children(prac, "telecom")
        contact = "n/a"
        if telecom_els:
            val_el = _find_child(telecom_els[0], "value")
            contact = _attr_value(val_el)
        practitioner_cards.append({"name": name, "role": "Practitioner", "contact": contact, "organization": "n/a"})
    organization_cards: list[dict[str, Any]] = []
    for org in organizations[:12]:
        name_el = _find_child(org, "name")
        telecom_els = _find_children(org, "telecom")
        contact = "n/a"
        if telecom_els:
            val_el = _find_child(telecom_els[0], "value")
            contact = _attr_value(val_el)
        organization_cards.append({"name": _attr_value(name_el), "contact": contact})
    return {"practitioners": practitioner_cards, "organizations": organization_cards}


# ---------------------------------------------------------------------------
# XML resource resolver
# ---------------------------------------------------------------------------

def _resolve_fhir_resources_xml(root: ET.Element, resource_type: str) -> list[ET.Element]:
    if _local_name(root.tag) == resource_type:
        return [root]
    if _local_name(root.tag) != "Bundle":
        return []
    resources: list[ET.Element] = []
    for entry in _find_children(root, "entry"):
        res = _find_child(entry, "resource")
        if res is not None:
            inner = list(res)
            if inner and _local_name(inner[0].tag) == resource_type:
                resources.append(inner[0])
    return resources


def _first_fhir_patient_xml(root: ET.Element) -> ET.Element | None:
    patients = _resolve_fhir_resources_xml(root, "Patient")
    return patients[0] if patients else None


# ---------------------------------------------------------------------------
# NDJSON bundle builder
# ---------------------------------------------------------------------------

def _fhir_bundle_from_ndjson_files(files: list[tuple[str, bytes, str]]) -> tuple[dict[str, Any], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    resource_counts: dict[str, int] = {}
    for file_name, raw, _suffix in files:
        decoded = raw.decode("utf-8", errors="replace")
        for line in decoded.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                resource = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(resource, dict):
                continue
            resource_type = str(resource.get("resourceType", "Unknown"))
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
            entries.append({"fullUrl": f"urn:chatclinic:{file_name}:{resource_type}:{resource_counts[resource_type]}", "resource": resource})
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
    return bundle, resource_counts


# ---------------------------------------------------------------------------
# High-level analysis functions
# ---------------------------------------------------------------------------

def _build_artifacts_json(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build all 8 artifact dicts from a parsed JSON payload."""
    patient_raw = _first_fhir_patient_json(payload)
    patient = _patient_browser_from_json(patient_raw)
    observations = _observation_viewer_from_json(payload)
    medications = _medication_timeline_from_json(payload)
    allergies = _allergy_summary_from_json(payload)
    vitals = _vital_summary_from_observations(observations)
    timeline = _timeline_events_from_json(payload)
    labs = _lab_trends_from_observations(observations)
    care_team = _care_team_from_json(payload)
    return {
        "patient": patient,
        "observations": observations,
        "medications": medications,
        "allergies": allergies,
        "vitals": vitals,
        "timeline": timeline,
        "labs": labs,
        "care_team": care_team,
    }


def _build_artifacts_xml(root: ET.Element) -> dict[str, dict[str, Any]]:
    """Build all 8 artifact dicts from a parsed XML root."""
    patient_el = _first_fhir_patient_xml(root)
    patient = _patient_browser_from_xml(patient_el) if patient_el is not None else {}
    observations = _observation_viewer_from_xml(root)
    medications = _medication_timeline_from_xml(root)
    allergies = _allergy_summary_from_xml(root)
    vitals = _vital_summary_from_observations(observations)
    timeline = _timeline_events_from_xml(root)
    labs = _lab_trends_from_observations(observations)
    care_team = _care_team_from_xml(root)
    return {
        "patient": patient,
        "observations": observations,
        "medications": medications,
        "allergies": allergies,
        "vitals": vitals,
        "timeline": timeline,
        "labs": labs,
        "care_team": care_team,
    }


def _resource_count_from_bundle(payload: dict[str, Any]) -> int:
    entries = payload.get("entry") or []
    return len(entries) if isinstance(entries, list) else 0


def _resource_type_label(payload: dict[str, Any]) -> str:
    return str(payload.get("resourceType", "Unknown"))


def _build_draft_answer(file_name: str, artifacts: dict[str, dict[str, Any]], resource_count: int) -> str:
    parts = [f"Parsed FHIR source **{file_name}** with {resource_count} resource(s)."]
    patient = artifacts.get("patient", {})
    if patient.get("full_name") and patient["full_name"] != "n/a":
        parts.append(f"Patient: {patient['full_name']}, {patient.get('gender', 'n/a')}, DOB {patient.get('birth_date', 'n/a')}.")
    obs = artifacts.get("observations", {})
    if obs.get("count"):
        parts.append(f"{obs['count']} observation(s) found.")
    meds = artifacts.get("medications", {})
    if meds.get("count"):
        parts.append(f"{meds['count']} medication(s) found.")
    allergies = artifacts.get("allergies", {})
    if allergies.get("count"):
        parts.append(f"{allergies['count']} allergy record(s).")
    return " ".join(parts)


def _build_response(
    file_name: str,
    resource_type: str,
    resource_count: int,
    patient_summary: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    warnings: list[str],
) -> FhirSourceResponse:
    studio_cards = [{"id": "fhir_browser", "title": "FHIR Browser", "subtitle": "Patient, observations, and medications"}]
    draft_answer = _build_draft_answer(file_name, artifacts, resource_count)
    return FhirSourceResponse(
        analysis_id="",
        source_fhir_path=None,
        file_name=file_name,
        file_kind="FHIR",
        resource_type=resource_type,
        resource_count=resource_count,
        patient_summary=patient_summary,
        metadata_items=[],
        studio_cards=studio_cards,
        artifacts=artifacts,
        warnings=warnings,
        draft_answer=draft_answer,
        used_tools=["fhir_browser_tool"],
    )


# ---------------------------------------------------------------------------
# Format-specific entry points
# ---------------------------------------------------------------------------

def analyze_fhir_json(file_name: str, raw_bytes: bytes) -> FhirSourceResponse:
    decoded = raw_bytes.decode("utf-8", errors="replace")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("FHIR JSON did not parse to a dict.")
    artifacts = _build_artifacts_json(payload)
    resource_type = _resource_type_label(payload)
    resource_count = _resource_count_from_bundle(payload)
    if resource_count == 0 and resource_type != "Bundle":
        resource_count = 1
    patient_summary = artifacts.get("patient", {})
    return _build_response(file_name, resource_type, resource_count, patient_summary, artifacts, [])


def analyze_fhir_xml(file_name: str, raw_bytes: bytes) -> FhirSourceResponse:
    decoded = raw_bytes.decode("utf-8", errors="replace")
    root = ET.fromstring(decoded)
    artifacts = _build_artifacts_xml(root)
    resource_type = _local_name(root.tag)
    entries = _find_children(root, "entry")
    resource_count = len(entries) if entries else 1
    patient_summary = artifacts.get("patient", {})
    return _build_response(file_name, resource_type, resource_count, patient_summary, artifacts, [])


def analyze_fhir_ndjson(file_name: str, raw_bytes: bytes) -> FhirSourceResponse:
    bundle, resource_counts = _fhir_bundle_from_ndjson_files([(file_name, raw_bytes, ".ndjson")])
    artifacts = _build_artifacts_json(bundle)
    total = sum(resource_counts.values())
    patient_summary = artifacts.get("patient", {})
    return _build_response(file_name, "Bundle", total, patient_summary, artifacts, [])


# ---------------------------------------------------------------------------
# Auto-detect entry point
# ---------------------------------------------------------------------------

def analyze_fhir_source(fhir_path: str, file_name: str) -> FhirSourceResponse:
    """Read a FHIR file, auto-detect its format, and return a FhirSourceResponse."""
    path = Path(fhir_path)
    raw_bytes = path.read_bytes()
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith(".ndjson"):
        result = analyze_fhir_ndjson(file_name, raw_bytes)
    elif suffix.endswith(".xml") or suffix.endswith(".fhir.xml"):
        result = analyze_fhir_xml(file_name, raw_bytes)
    else:
        # Default: try JSON
        text = raw_bytes.decode("utf-8", errors="replace").lstrip()
        if text.startswith("{") and '"resourceType"' in text[:512]:
            result = analyze_fhir_json(file_name, raw_bytes)
        else:
            # Fall back to JSON attempt anyway
            result = analyze_fhir_json(file_name, raw_bytes)

    result.source_fhir_path = fhir_path
    return result


# ---------------------------------------------------------------------------
# Plugin entrypoint
# ---------------------------------------------------------------------------

def execute(payload: dict[str, Any]) -> dict[str, Any]:
    """Plugin entrypoint called by tool_runner."""
    fhir_path = str(payload.get("fhir_path") or payload.get("source_fhir_path") or "")
    file_name = str(payload.get("file_name") or payload.get("original_name") or Path(fhir_path).name or "bundle.fhir.json")
    if not fhir_path:
        raise FileNotFoundError("No fhir_path provided in payload.")
    result = analyze_fhir_source(fhir_path, file_name)
    return result.model_dump()
