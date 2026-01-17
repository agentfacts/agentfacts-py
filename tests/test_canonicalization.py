"""Tests for JSON canonicalization."""

from agentfacts.crypto.canonicalization import (
    canonicalize_json,
    canonicalize_to_string,
    compute_hash,
    normalize_for_signing,
)


class TestCanonicalization:
    """Tests for JSON canonicalization (RFC 8785)."""

    def test_empty_object(self):
        """Test canonicalizing empty object."""
        result = canonicalize_to_string({})
        assert result == "{}"

    def test_empty_array(self):
        """Test canonicalizing empty array."""
        result = canonicalize_to_string([])
        assert result == "[]"

    def test_key_ordering(self):
        """Test that keys are sorted lexicographically."""
        result = canonicalize_to_string({"z": 1, "a": 2, "m": 3})
        assert result == '{"a":2,"m":3,"z":1}'

    def test_nested_key_ordering(self):
        """Test key ordering in nested objects."""
        result = canonicalize_to_string(
            {
                "b": {"z": 1, "a": 2},
                "a": 1,
            }
        )
        assert result == '{"a":1,"b":{"a":2,"z":1}}'

    def test_no_whitespace(self):
        """Test that output has no extra whitespace."""
        result = canonicalize_to_string({"key": "value", "num": 123})
        assert " " not in result
        assert "\n" not in result
        assert "\t" not in result

    def test_string_escaping(self):
        """Test proper string escaping."""
        result = canonicalize_to_string({"text": 'Hello "World"'})
        assert result == '{"text":"Hello \\"World\\""}'

    def test_control_character_escaping(self):
        """Test control character escaping."""
        result = canonicalize_to_string({"text": "line1\nline2\ttab"})
        assert "\\n" in result
        assert "\\t" in result

    def test_integer_serialization(self):
        """Test integer serialization."""
        result = canonicalize_to_string({"num": 42})
        assert result == '{"num":42}'

    def test_float_serialization(self):
        """Test float serialization."""
        result = canonicalize_to_string({"num": 3.14})
        assert "3.14" in result

    def test_float_exponent_thresholds(self):
        """Test ECMAScript exponent thresholds."""
        assert canonicalize_to_string({"num": 1e-6}) == '{"num":0.000001}'
        assert canonicalize_to_string({"num": 1e-7}) == '{"num":1e-7}'
        assert canonicalize_to_string({"num": 1e21}) == '{"num":1e+21}'

    def test_boolean_serialization(self):
        """Test boolean serialization."""
        result = canonicalize_to_string({"t": True, "f": False})
        assert result == '{"f":false,"t":true}'

    def test_null_serialization(self):
        """Test null serialization."""
        result = canonicalize_to_string({"value": None})
        assert result == '{"value":null}'

    def test_array_serialization(self):
        """Test array serialization."""
        result = canonicalize_to_string({"arr": [1, 2, 3]})
        assert result == '{"arr":[1,2,3]}'

    def test_deterministic_output(self):
        """Test that same input always produces same output."""
        data = {"b": 2, "a": 1, "nested": {"z": 3, "y": 4}}

        result1 = canonicalize_to_string(data)
        result2 = canonicalize_to_string(data)

        assert result1 == result2

    def test_canonicalize_json_returns_bytes(self):
        """Test that canonicalize_json returns UTF-8 bytes."""
        result = canonicalize_json({"key": "value"})

        assert isinstance(result, bytes)
        assert result == b'{"key":"value"}'

    def test_unicode_handling(self):
        """Test Unicode string handling."""
        result = canonicalize_to_string({"emoji": "👍", "chinese": "中文"})
        assert "👍" in result
        assert "中文" in result


class TestNormalization:
    """Tests for signing normalization."""

    def test_removes_signature(self):
        """Test that signature field is removed."""
        data = {"name": "test", "signature": "abc123"}
        result = normalize_for_signing(data)

        assert "signature" not in result
        assert result["name"] == "test"

    def test_removes_proof(self):
        """Test that proof field is removed."""
        data = {"name": "test", "proof": {"type": "Ed25519"}}
        result = normalize_for_signing(data)

        assert "proof" not in result

    def test_preserves_merkle_root(self):
        """Test that merkle_root is preserved unless explicitly removed."""
        data = {"name": "test", "merkle_root": "abc"}
        result = normalize_for_signing(data)

        assert result["merkle_root"] == "abc"

    def test_removes_log_proof(self):
        """Test that log_proof field is removed."""
        data = {"name": "test", "log_proof": {"root_hash": "abc"}}
        result = normalize_for_signing(data)

        assert "log_proof" not in result

    def test_preserves_other_fields(self):
        """Test that other fields are preserved."""
        data = {
            "name": "test",
            "did": "did:key:z123",
            "capabilities": [{"name": "tool"}],
            "signature": "remove_me",
        }
        result = normalize_for_signing(data)

        assert result["name"] == "test"
        assert result["did"] == "did:key:z123"
        assert len(result["capabilities"]) == 1

    def test_deep_copy(self):
        """Test that normalization creates a deep copy."""
        data = {"nested": {"value": 1}, "signature": "abc"}
        result = normalize_for_signing(data)

        # Modify result
        result["nested"]["value"] = 999

        # Original should be unchanged
        assert data["nested"]["value"] == 1


class TestHashComputation:
    """Tests for hash computation."""

    def test_compute_hash_dict(self):
        """Test computing hash of a dict."""
        data = {"key": "value"}
        hash1 = compute_hash(data)

        assert len(hash1) == 64  # SHA-256 hex = 64 chars

    def test_compute_hash_bytes(self):
        """Test computing hash of bytes."""
        data = b"hello world"
        hash1 = compute_hash(data)

        assert len(hash1) == 64

    def test_hash_determinism(self):
        """Test that same data produces same hash."""
        data = {"b": 2, "a": 1}

        hash1 = compute_hash(data)
        hash2 = compute_hash(data)

        assert hash1 == hash2

    def test_hash_sensitivity(self):
        """Test that different data produces different hash."""
        hash1 = compute_hash({"key": "value1"})
        hash2 = compute_hash({"key": "value2"})

        assert hash1 != hash2
