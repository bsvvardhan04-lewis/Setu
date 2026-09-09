"""Form validation - the rule layer the LLM cannot overrule."""

from __future__ import annotations

import pytest

from setu.pipeline.form_agent import normalise_digits, validate, verhoeff_valid


@pytest.mark.parametrize(
    "number,expected",
    [
        ("234567890124", True),   # valid Verhoeff check digit
        ("234567890123", False),  # wrong check digit
        ("123456789012", False),  # may not start with 0 or 1
        ("23456789012", False),   # too short
    ],
)
def test_verhoeff(number, expected):
    assert verhoeff_valid(number) is expected


def test_aadhaar_is_formatted_when_valid():
    value, error = validate("aadhaar", "2345 6789 0124")
    assert error is None
    assert value == "2345 6789 0124"


def test_aadhaar_with_bad_checksum_is_refused():
    value, error = validate("aadhaar", "2345 6789 0123")
    assert value is None
    assert "checksum" in error.lower()


def test_devanagari_numerals_are_normalised():
    assert normalise_digits("९८७६५४३२१०") == "9876543210"


def test_mobile_accepts_spoken_country_code():
    value, error = validate("mobile", "+91 98765 43210")
    assert error is None
    assert value == "9876543210"


def test_mobile_rejects_impossible_prefix():
    value, error = validate("mobile", "1234567890")
    assert value is None
    assert error


def test_pincode_rules():
    assert validate("pincode", "560001") == ("560001", None)
    assert validate("pincode", "056001")[0] is None


def test_date_normalises_to_dd_mm_yyyy():
    assert validate("date", "5-3-89")[0] == "05/03/1989"
    assert validate("date", "32/01/2020")[0] is None


def test_none_sentinel_from_the_model_clears_the_slot():
    assert validate("text", "NONE") == (None, None)
