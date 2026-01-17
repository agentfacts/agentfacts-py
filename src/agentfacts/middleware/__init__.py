"""
HTTP middleware for AgentFacts SDK.

Provides FastAPI and Flask middleware for lightweight header
verification (DID/public key match and optional nonce signature).
Full metadata signature verification and policy checks are opt-in
via middleware configuration.
"""

from agentfacts.middleware.fastapi import (
    AgentFactsMiddleware as FastAPIMiddleware,
)
from agentfacts.middleware.fastapi import (
    get_verified_agent,
    require_verified_agent,
)
from agentfacts.middleware.flask import (
    AgentFactsMiddleware as FlaskMiddleware,
)
from agentfacts.middleware.flask import (
    flask_require_verified_agent,
    get_flask_verified_agent,
)
from agentfacts.middleware.headers import (
    HEADER_AGENT_DID,
    HEADER_AGENT_PUBLIC_KEY,
    HEADER_AGENT_SIGNATURE,
    HEADER_NONCE,
    HEADER_NONCE_SIGNATURE,
    extract_agent_headers,
    inject_agent_headers,
)

__all__ = [
    "FastAPIMiddleware",
    "FlaskMiddleware",
    "require_verified_agent",
    "get_verified_agent",
    "flask_require_verified_agent",
    "get_flask_verified_agent",
    "HEADER_AGENT_DID",
    "HEADER_AGENT_SIGNATURE",
    "HEADER_AGENT_PUBLIC_KEY",
    "HEADER_NONCE",
    "HEADER_NONCE_SIGNATURE",
    "inject_agent_headers",
    "extract_agent_headers",
]
