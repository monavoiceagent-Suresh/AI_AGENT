import re

_DIGITS = re.compile(r"^[1-9][0-9]{9,14}$")
# +91 then ten digits, none of them a trunk 0: the shape of every Indian number.
_INDIAN_E164 = re.compile(r"^\+91[1-9][0-9]{9}$")


def normalize_phone(raw: str | None) -> str | None:
    """Digits-only form with the country code first, or None if it cannot be a phone number.

    Exotel's `to` may arrive as +918047491899, 918047491899, 08047491899 or 8047491899. A leading 0 or a
    bare ten-digit number is read as an Indian number (country code 91); a trunk 0 typed after +91
    (+91 0 8047491899) is dropped; anything else is taken as given.
    """
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("0"):
        digits = "91" + digits[1:]
    elif len(digits) == 10:
        digits = "91" + digits
    elif len(digits) == 13 and digits.startswith("910"):
        digits = "91" + digits[3:]
    return digits if _DIGITS.match(digits) else None


def indian_number_problem(e164: str) -> str | None:
    """Why a +91 number cannot be right (None for other countries or a well-formed Indian number)."""
    if not e164.startswith("+91") or _INDIAN_E164.match(e164):
        return None
    local = e164[3:]
    if local.startswith("0"):
        return f"Indian numbers are +91 followed by ten digits without a leading 0 (did you mean +91{local[1:]}?)"
    return "Indian numbers are +91 followed by exactly ten digits"
