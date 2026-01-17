"""
Flask middleware for AgentFacts verification.

Provides middleware and decorators for lightweight
agent identity checks on incoming requests.
"""

import inspect
import json
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

from agentfacts.core import AgentFacts
from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair
from agentfacts.middleware.headers import extract_agent_headers
from agentfacts.models import VerificationResult
from agentfacts.policy.rules import Policy

if TYPE_CHECKING:
    from agentfacts.plugins import VerificationContext

try:
    from flask import g, jsonify
    from flask import request as flask_request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class AgentFactsMiddleware:
    """
    Flask middleware for agent verification.

    Verifies agent identity headers (DID/public key match and optional
    nonce signature). Full verification (metadata signature + policy)
    is available when full_verify=True with a metadata_provider.

    Example:
        ```python
        from flask import Flask
        from agentfacts.middleware import FlaskMiddleware

        app = Flask(__name__)
        middleware = FlaskMiddleware(app, verify_peers=True)
        ```
    """

    def __init__(
        self,
        app: Any | None = None,
        verify_peers: bool = False,
        policy: Policy | None = None,
        exclude_paths: list[str] | None = None,
        full_verify: bool = False,
        metadata_provider: Callable[..., Any] | None = None,
        verification_context: Optional["VerificationContext"] = None,
    ) -> None:
        """
        Initialize the middleware.

        Args:
            app: Flask application (can also use init_app)
            verify_peers: If True, reject requests without valid agent headers
            policy: Optional policy to evaluate against
            exclude_paths: Paths to exclude from verification
            full_verify: If True, verify signed metadata and enforce policy
            metadata_provider: Callable that resolves agent metadata by DID
            verification_context: Optional VerificationContext for enhanced
                verification with external DID resolvers, attestation verifiers,
                and status checkers.
        """
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask is required for AgentFactsMiddleware. "
                "Install with: pip install agentfacts[middleware]"
            )

        self.verify_peers = verify_peers
        self.policy = policy
        self.exclude_paths = set(exclude_paths or [])
        self.full_verify = full_verify
        self.metadata_provider = metadata_provider
        self.verification_context = verification_context

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Any) -> None:
        """Initialize the middleware with a Flask app."""
        app.before_request(self._before_request)

    def _before_request(self) -> Any | None:
        """Process request before it reaches the view."""
        # Skip excluded paths
        if flask_request.path in self.exclude_paths:
            g.verified_agent = None
            return None

        # Extract agent headers
        headers = dict(flask_request.headers)
        agent_info = extract_agent_headers(headers)

        # If lightweight or full verification is enabled, require valid agent identity
        if self.verify_peers or self.full_verify:
            if not agent_info["did"] or not agent_info["public_key"]:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Missing agent identity headers.",
                        }
                    ),
                    403,
                )

            # Verify DID matches public key
            try:
                key_pair = KeyPair.from_public_key_base64(agent_info["public_key"])
                expected_did = DID.from_key_pair(key_pair)

                if expected_did.uri != agent_info["did"]:
                    return (
                        jsonify(
                            {
                                "error": "Agent Identity Unverified",
                                "code": 403,
                                "message": "DID does not match public key.",
                            }
                        ),
                        403,
                    )

                # Verify nonce signature if provided
                if (
                    agent_info["nonce"]
                    and agent_info["nonce_signature"]
                    and not key_pair.verify_base64(
                        agent_info["nonce"].encode(), agent_info["nonce_signature"]
                    )
                ):
                    return (
                        jsonify(
                            {
                                "error": "Agent Identity Unverified",
                                "code": 403,
                                "message": "Nonce signature verification failed.",
                            }
                        ),
                        403,
                    )

            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": f"Identity verification failed: {str(e)}",
                        }
                    ),
                    403,
                )

        if self.full_verify:
            if not self.metadata_provider:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Full verification requires a metadata_provider.",
                        }
                    ),
                    403,
                )

            try:
                did = agent_info["did"]
                if not did:
                    return (
                        jsonify(
                            {
                                "error": "Agent Identity Unverified",
                                "code": 403,
                                "message": "Missing agent DID.",
                            }
                        ),
                        403,
                    )
                metadata = self._resolve_metadata(did)
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": f"Metadata resolution failed: {str(e)}",
                        }
                    ),
                    403,
                )

            if not metadata:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "No metadata available for agent.",
                        }
                    ),
                    403,
                )

            try:
                facts = self._build_agent_facts(metadata, agent_info["public_key"])
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": f"Invalid agent metadata: {str(e)}",
                        }
                    ),
                    403,
                )

            if facts.did != agent_info["did"]:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "DID does not match metadata.",
                        }
                    ),
                    403,
                )

            verify_key = None
            if agent_info["public_key"]:
                verify_key = KeyPair.from_public_key_base64(agent_info["public_key"])

            verification = facts.verify(verify_key, context=self.verification_context)
            if not verification.valid:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Metadata signature verification failed.",
                            "errors": verification.errors,
                        }
                    ),
                    403,
                )

            if (
                facts.metadata.signature.value
                and agent_info["signature"]
                and agent_info["signature"] != facts.metadata.signature.value
            ):
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Signature header does not match metadata.",
                        }
                    ),
                    403,
                )

            if agent_info["public_key"] and facts.metadata.signature.value:
                for key in facts.metadata.publisher.keys:
                    if key.id == facts.metadata.signature.key_id:
                        if key.public_key != agent_info["public_key"]:
                            return (
                                jsonify(
                                    {
                                        "error": "Agent Identity Unverified",
                                        "code": 403,
                                        "message": "Public key does not match publisher key.",
                                    }
                                ),
                                403,
                            )
                        break

            if self.policy:
                policy_result = self.policy.evaluate(facts.metadata)
                if not policy_result.passed:
                    return (
                        jsonify(
                            {
                                "error": "Agent Identity Unverified",
                                "code": 403,
                                "message": "Agent metadata failed policy evaluation.",
                                "violations": [
                                    str(v) for v in policy_result.violations
                                ],
                            }
                        ),
                        403,
                    )

            # Store verified agent info in Flask g object
            g.verified_agent = verification
        else:
            # Store verified agent info in Flask g object
            g.verified_agent = (
                VerificationResult(
                    valid=True,
                    did=agent_info["did"],
                )
                if agent_info["did"]
                else None
            )

        return None

    def _resolve_metadata(self, did: str) -> Any:
        provider = self.metadata_provider
        if provider is None:
            return None

        try:
            sig = inspect.signature(provider)
            params = list(sig.parameters.values())
            accepts_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
            )
            if accepts_varargs or len(params) >= 2:
                return provider(did, flask_request)
            if len(params) == 1:
                return provider(did)
            return provider()
        except (TypeError, ValueError):
            try:
                return provider(did, flask_request)
            except TypeError:
                return provider(did)

    def _build_agent_facts(
        self, metadata: Any, header_public_key: str | None
    ) -> AgentFacts:
        if isinstance(metadata, AgentFacts):
            return metadata

        data: dict[str, Any] | None = None
        if isinstance(metadata, str):
            data = json.loads(metadata)
        elif isinstance(metadata, dict):
            data = metadata

        if data is None:
            raise ValueError("Unsupported metadata format")

        key_pair = None
        if header_public_key:
            key_pair = KeyPair.from_public_key_base64(header_public_key)

        return AgentFacts.from_dict(data, key_pair=key_pair)


def flask_require_verified_agent(
    policy: Policy | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Flask decorator that requires a verified agent.

    Example:
        ```python
        from flask import Flask
        from agentfacts.middleware import flask_require_verified_agent

        app = Flask(__name__)

        @app.route("/secure")
        @flask_require_verified_agent()
        def secure_endpoint():
            agent = g.verified_agent
            return {"message": f"Hello, agent {agent.did}"}
        ```
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required")

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            agent = getattr(g, "verified_agent", None)

            if not agent or not agent.valid:
                return (
                    jsonify(
                        {
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "This endpoint requires verified agent identity.",
                        }
                    ),
                    403,
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def get_flask_verified_agent() -> VerificationResult | None:
    """
    Get the verified agent from Flask's g object.

    Returns:
        VerificationResult or None
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required")

    return getattr(g, "verified_agent", None)
