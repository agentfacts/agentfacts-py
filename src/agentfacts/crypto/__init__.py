"""
Cryptographic primitives for AgentFacts SDK.

Provides Ed25519 key management, DID generation, signing/verification,
and JSON canonicalization.
"""

from agentfacts.crypto.canonicalization import canonicalize_json, compute_hash
from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair
from agentfacts.crypto.signing import sign_message, verify_signature

__all__ = [
    "KeyPair",
    "DID",
    "sign_message",
    "verify_signature",
    "canonicalize_json",
    "compute_hash",
]
