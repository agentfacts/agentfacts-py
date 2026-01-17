"""
Message signing and verification utilities.

Provides high-level functions for signing and verifying messages,
including support for canonicalized JSON-LD payloads.
"""

from typing import Any

from agentfacts.crypto.keys import KeyPair


def sign_message(message: bytes, key_pair: KeyPair) -> str:
    """
    Sign a message and return the base64-encoded signature.

    Args:
        message: The message bytes to sign
        key_pair: The key pair containing the private key

    Returns:
        Base64url-encoded signature
    """
    return key_pair.sign_base64(message)


def verify_signature(message: bytes, signature: str, key_pair: KeyPair) -> bool:
    """
    Verify a signature against a message.

    Args:
        message: The original message bytes
        signature: Base64url-encoded signature
        key_pair: The key pair containing the public key

    Returns:
        True if signature is valid, False otherwise
    """
    return key_pair.verify_base64(message, signature)


def sign_json(
    data: dict[str, Any], key_pair: KeyPair, *, exclude_fields: list[str] | None = None
) -> str:
    """
    Sign a JSON object after canonicalization.

    Args:
        data: The JSON-serializable data to sign
        key_pair: The key pair containing the private key
        exclude_fields: Fields to exclude before signing (e.g., ['signature'])

    Returns:
        Base64url-encoded signature
    """
    from agentfacts.crypto.canonicalization import canonicalize_json

    # Remove excluded fields
    if exclude_fields:
        data = {k: v for k, v in data.items() if k not in exclude_fields}

    canonical = canonicalize_json(data)
    return sign_message(canonical, key_pair)


def verify_json_signature(
    data: dict[str, Any],
    signature: str,
    key_pair: KeyPair,
    *,
    exclude_fields: list[str] | None = None,
) -> bool:
    """
    Verify a signature against a JSON object.

    Args:
        data: The JSON-serializable data that was signed
        signature: Base64url-encoded signature to verify
        key_pair: The key pair containing the public key
        exclude_fields: Fields to exclude (should match what was used during signing)

    Returns:
        True if signature is valid, False otherwise
    """
    from agentfacts.crypto.canonicalization import canonicalize_json

    # Remove excluded fields
    if exclude_fields:
        data = {k: v for k, v in data.items() if k not in exclude_fields}

    canonical = canonicalize_json(data)
    return verify_signature(canonical, signature, key_pair)


def create_detached_signature(
    data: dict[str, Any], key_pair: KeyPair, *, algorithm: str = "Ed25519"
) -> dict[str, Any]:
    """
    Create a detached JWS-like signature object.

    Returns a signature object that can be stored separately from the data.
    """
    from agentfacts.crypto.canonicalization import canonicalize_json

    canonical = canonicalize_json(data)
    signature = sign_message(canonical, key_pair)

    return {
        "algorithm": algorithm,
        "public_key": key_pair.public_key_base64,
        "signature": signature,
    }


def verify_detached_signature(
    data: dict[str, Any], signature_obj: dict[str, Any]
) -> bool:
    """
    Verify a detached signature object.

    Args:
        data: The original data that was signed
        signature_obj: The detached signature object

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        public_key_b64 = signature_obj["public_key"]
        signature = signature_obj["signature"]

        key_pair = KeyPair.from_public_key_base64(public_key_b64)
        return verify_json_signature(data, signature, key_pair)
    except Exception:
        return False
