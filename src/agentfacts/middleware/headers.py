"""
HTTP header utilities for agent identity.

Defines standard headers for transmitting agent credentials
and provides utilities for injection/extraction.
"""

from collections.abc import Generator
from typing import Any

# Standard header names for agent identity
HEADER_AGENT_DID = "X-Agent-DID"
HEADER_AGENT_SIGNATURE = "X-Agent-Signature"
HEADER_AGENT_PUBLIC_KEY = "X-Agent-Public-Key"
HEADER_NONCE = "X-Agent-Nonce"
HEADER_NONCE_SIGNATURE = "X-Agent-Nonce-Signature"
HEADER_METADATA_HASH = "X-Agent-Metadata-Hash"


def inject_agent_headers(
    headers: dict[str, str],
    agent_facts: Any,  # AgentFacts instance
    nonce: str | None = None,
) -> dict[str, str]:
    """
    Inject agent identity headers into a request.

    Args:
        headers: Existing headers dict to update
        agent_facts: AgentFacts instance
        nonce: Optional nonce for handshake (will be signed)

    Returns:
        Updated headers dict
    """
    headers[HEADER_AGENT_DID] = agent_facts.did
    headers[HEADER_AGENT_PUBLIC_KEY] = agent_facts.public_key

    if agent_facts.is_signed:
        headers[HEADER_AGENT_SIGNATURE] = agent_facts.signature or ""

    if nonce:
        headers[HEADER_NONCE] = nonce
        signature = agent_facts.key_pair.sign_base64(nonce.encode())
        headers[HEADER_NONCE_SIGNATURE] = signature

    return headers


def extract_agent_headers(headers: dict[str, str]) -> dict[str, str | None]:
    """
    Extract agent identity headers from a request.

    Args:
        headers: Request headers dict

    Returns:
        Dict with extracted values (None if not present)
    """
    # Normalize header names (case-insensitive)
    normalized = {k.lower(): v for k, v in headers.items()}

    return {
        "did": normalized.get(HEADER_AGENT_DID.lower()),
        "signature": normalized.get(HEADER_AGENT_SIGNATURE.lower()),
        "public_key": normalized.get(HEADER_AGENT_PUBLIC_KEY.lower()),
        "nonce": normalized.get(HEADER_NONCE.lower()),
        "nonce_signature": normalized.get(HEADER_NONCE_SIGNATURE.lower()),
        "metadata_hash": normalized.get(HEADER_METADATA_HASH.lower()),
    }


def create_httpx_auth(agent_facts: Any) -> Any:
    """
    Create an httpx Auth class for automatic header injection.

    Usage:
        ```python
        import httpx
        from agentfacts.middleware.headers import create_httpx_auth

        auth = create_httpx_auth(my_agent_facts)
        response = httpx.get("https://api.example.com/data", auth=auth)
        ```
    """
    try:
        import httpx
    except ImportError as err:
        raise ImportError("httpx is required for create_httpx_auth") from err

    class AgentFactsAuth(httpx.Auth):
        def __init__(self, facts: Any) -> None:
            self.facts = facts

        def auth_flow(
            self, request: httpx.Request
        ) -> Generator[httpx.Request, httpx.Response, None]:
            request.headers[HEADER_AGENT_DID] = self.facts.did
            request.headers[HEADER_AGENT_PUBLIC_KEY] = self.facts.public_key
            if self.facts.is_signed:
                request.headers[HEADER_AGENT_SIGNATURE] = self.facts.signature or ""
            yield request

    return AgentFactsAuth(agent_facts)
