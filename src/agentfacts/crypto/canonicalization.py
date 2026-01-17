"""
Canonicalization engine for AgentFacts SDK.

Ensures deterministic JSON serialization before signing to prevent
"signature mismatch" errors caused by key ordering or whitespace differences.

Implements RFC 8785 (JSON Canonicalization Scheme - JCS) for consistent
serialization across different platforms and implementations.
"""

import json
from decimal import Decimal
from typing import Any, cast


def _serialize_number(num: float | int) -> str:
    """
    Serialize a number according to RFC 8785.

    - Integers are serialized without decimal point
    - Floats use exponential notation when appropriate
    - Special handling for very large/small numbers
    """
    if isinstance(num, bool):
        return "true" if num else "false"

    if isinstance(num, int):
        return str(num)

    # Handle special float values
    if num != num:  # NaN
        raise ValueError("NaN is not allowed in canonical JSON")
    if num == float("inf") or num == float("-inf"):
        raise ValueError("Infinity is not allowed in canonical JSON")

    # Normalize negative zero
    if num == 0:
        return "0"

    # Check if it's effectively an integer within safe range
    if num == int(num) and abs(num) < 2**53:
        return str(int(num))

    # Use Decimal based on shortest float repr
    dec = Decimal(repr(num))
    abs_num = abs(num)

    # ECMAScript threshold: use exponential for abs < 1e-6 or abs >= 1e21
    if abs_num < 1e-6 or abs_num >= 1e21:
        return _format_exponent(dec)

    return _format_decimal(dec)


def _serialize_string(s: str) -> str:
    """
    Serialize a string according to RFC 8785.

    Escapes control characters and required JSON metacharacters.
    """
    result = ['"']
    for char in s:
        code = ord(char)
        if char == '"':
            result.append('\\"')
        elif char == "\\":
            result.append("\\\\")
        elif char == "\b":
            result.append("\\b")
        elif char == "\f":
            result.append("\\f")
        elif char == "\n":
            result.append("\\n")
        elif char == "\r":
            result.append("\\r")
        elif char == "\t":
            result.append("\\t")
        elif code < 0x20:
            result.append(f"\\u{code:04x}")
        else:
            result.append(char)
    result.append('"')
    return "".join(result)


def _format_decimal(dec: Decimal) -> str:
    """Format a Decimal using plain notation with no trailing zeros."""
    s = format(dec, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _format_exponent(dec: Decimal) -> str:
    """Format a Decimal using exponent notation with minimal exponent."""
    s = format(dec.normalize(), "E")
    mantissa, exponent = s.split("E", 1)
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exp_int = int(exponent)
    exp_sign = "+" if exp_int >= 0 else ""
    return f"{mantissa}e{exp_sign}{exp_int}"


def _utf16_sort_key(value: str) -> bytes:
    """Sort key based on UTF-16 code units (RFC 8785)."""
    return value.encode("utf-16-be", "surrogatepass")


def _canonicalize(value: Any) -> str:
    """
    Recursively canonicalize a JSON value.

    Args:
        value: Any JSON-serializable value

    Returns:
        Canonical JSON string
    """
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return _serialize_number(value)

    if isinstance(value, str):
        return _serialize_string(value)

    if isinstance(value, (list, tuple)):
        elements = [_canonicalize(item) for item in value]
        return "[" + ",".join(elements) + "]"

    if isinstance(value, dict):
        # Sort keys lexicographically by UTF-16 code units
        sorted_keys = sorted(value.keys(), key=_utf16_sort_key)
        pairs = [
            _serialize_string(key) + ":" + _canonicalize(value[key])
            for key in sorted_keys
        ]
        return "{" + ",".join(pairs) + "}"

    # For other types, try to convert to a serializable form
    raise TypeError(f"Cannot canonicalize type: {type(value).__name__}")


def canonicalize_json(data: dict[str, Any] | list[Any]) -> bytes:
    """
    Canonicalize a JSON object/array to bytes.

    This is the main entry point for canonicalization before signing.

    Args:
        data: A JSON-serializable dict or list

    Returns:
        UTF-8 encoded canonical JSON bytes
    """
    canonical_str = _canonicalize(data)
    return canonical_str.encode("utf-8")


def canonicalize_to_string(data: dict[str, Any] | list[Any]) -> str:
    """
    Canonicalize a JSON object/array to a string.

    Args:
        data: A JSON-serializable dict or list

    Returns:
        Canonical JSON string
    """
    return _canonicalize(data)


def normalize_for_signing(
    data: dict[str, Any], *, exclude_signature: bool = True
) -> dict[str, Any]:
    """
    Prepare data for signing by normalizing and optionally removing signature fields.

    Args:
        data: The data to normalize
        exclude_signature: Whether to remove 'signature' and 'proof' fields

    Returns:
        Normalized copy of the data
    """
    # Deep copy via JSON round-trip (also ensures serializability)
    normalized = cast(dict[str, Any], json.loads(json.dumps(data, default=str)))

    if exclude_signature:
        normalized.pop("signature", None)
        normalized.pop("log_proof", None)
        normalized.pop("proof", None)

    return normalized


def compute_hash(data: dict[str, Any] | bytes) -> str:
    """
    Compute a SHA-256 hash of canonical JSON data.

    Args:
        data: Either a dict to canonicalize or pre-canonicalized bytes

    Returns:
        Hex-encoded SHA-256 hash
    """
    import hashlib

    if isinstance(data, dict):
        data = canonicalize_json(data)

    return hashlib.sha256(data).hexdigest()
