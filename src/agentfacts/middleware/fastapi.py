"""
FastAPI middleware for AgentFacts verification.

Provides middleware and dependency injection for lightweight
agent identity checks on incoming requests.
"""

import inspect
import json
from collections.abc import Awaitable, Callable
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
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    class BaseHTTPMiddleware:  # type: ignore[no-redef]
        """Fallback base class when FastAPI/Starlette are unavailable."""

        pass


# Request state key for verified agent info
AGENT_STATE_KEY = "verified_agent"


class AgentFactsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for agent verification.

    Verifies agent identity headers (DID/public key match and optional
    nonce signature). Full verification (metadata signature + policy)
    is available when full_verify=True with a metadata_provider.

    Example:
        ```python
        from fastapi import FastAPI
        from agentfacts.middleware import FastAPIMiddleware
        from agentfacts.policy import Policy

        app = FastAPI()
        app.add_middleware(
            FastAPIMiddleware,
            verify_peers=True,
            policy=Policy.basic_trust(),
        )
        ```
    """

    def __init__(
        self,
        app: Any,
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
            app: FastAPI application
            verify_peers: If True, reject requests without valid agent headers
            policy: Optional policy to evaluate against
            exclude_paths: Paths to exclude from verification (e.g., ["/health"])
            full_verify: If True, verify signed metadata and enforce policy
            metadata_provider: Callable that resolves agent metadata by DID
            verification_context: Optional VerificationContext for enhanced
                verification with external DID resolvers, attestation verifiers,
                and status checkers. When provided, enables:
                - External DID method resolution (did:web, did:ion, etc.)
                - Verifiable Credential verification (SD-JWT-VC, etc.)
                - Revocation/status checking
                - Governance policy evaluation
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for AgentFactsMiddleware. "
                "Install with: pip install agentfacts[middleware]"
            )

        super().__init__(app)
        self.verify_peers = verify_peers
        self.policy = policy
        self.exclude_paths = set(exclude_paths or [])
        self.full_verify = full_verify
        self.metadata_provider = metadata_provider
        self.verification_context = verification_context

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        """Process the request and verify agent identity."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Extract agent headers
        headers = dict(request.headers)
        agent_info = extract_agent_headers(headers)

        # If lightweight or full verification is enabled, require valid agent identity
        if self.verify_peers or self.full_verify:
            if not agent_info["did"] or not agent_info["public_key"]:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "Missing agent identity headers. "
                        "Please include X-Agent-DID and X-Agent-Public-Key headers.",
                    },
                )

            # Verify DID matches public key
            try:
                key_pair = KeyPair.from_public_key_base64(agent_info["public_key"])
                expected_did = DID.from_key_pair(key_pair)

                if expected_did.uri != agent_info["did"]:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "DID does not match public key.",
                        },
                    )

                # Verify nonce signature if provided
                if (
                    agent_info["nonce"]
                    and agent_info["nonce_signature"]
                    and not key_pair.verify_base64(
                        agent_info["nonce"].encode(), agent_info["nonce_signature"]
                    )
                ):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Nonce signature verification failed.",
                        },
                    )

            except Exception as e:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": f"Identity verification failed: {str(e)}",
                    },
                )

        if self.full_verify:
            if not self.metadata_provider:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "Full verification requires a metadata_provider.",
                    },
                )

            try:
                did = agent_info["did"]
                if not did:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Missing agent DID.",
                        },
                    )
                metadata = await self._resolve_metadata(did, request)
            except Exception as e:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": f"Metadata resolution failed: {str(e)}",
                    },
                )

            if not metadata:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "No metadata available for agent.",
                    },
                )

            try:
                facts = self._build_agent_facts(metadata, agent_info["public_key"])
            except Exception as e:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": f"Invalid agent metadata: {str(e)}",
                    },
                )

            if facts.did != agent_info["did"]:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "DID does not match metadata.",
                    },
                )

            verify_key = None
            if agent_info["public_key"]:
                verify_key = KeyPair.from_public_key_base64(agent_info["public_key"])

            verification = facts.verify(verify_key, context=self.verification_context)
            if not verification.valid:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "Metadata signature verification failed.",
                        "errors": verification.errors,
                    },
                )

            if (
                facts.metadata.signature.value
                and agent_info["signature"]
                and agent_info["signature"] != facts.metadata.signature.value
            ):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Agent Identity Unverified",
                        "code": 403,
                        "message": "Signature header does not match metadata.",
                    },
                )

            if agent_info["public_key"] and facts.metadata.signature.value:
                for key in facts.metadata.publisher.keys:
                    if key.id == facts.metadata.signature.key_id:
                        if key.public_key != agent_info["public_key"]:
                            return JSONResponse(
                                status_code=403,
                                content={
                                    "error": "Agent Identity Unverified",
                                    "code": 403,
                                    "message": "Public key does not match publisher key.",
                                },
                            )
                        break

            if self.policy:
                policy_result = self.policy.evaluate(facts.metadata)
                if not policy_result.passed:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent Identity Unverified",
                            "code": 403,
                            "message": "Agent metadata failed policy evaluation.",
                            "violations": [str(v) for v in policy_result.violations],
                        },
                    )

            # Store verified agent info in request state
            request.state.verified_agent = verification
        else:
            # Store verified agent info in request state
            request.state.verified_agent = (
                VerificationResult(
                    valid=True,
                    did=agent_info["did"],
                )
                if agent_info["did"]
                else None
            )

        return await call_next(request)

    async def _resolve_metadata(self, did: str, request: Request) -> Any:
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
                result = provider(did, request)
            else:
                result = provider(did)
        except (TypeError, ValueError):
            try:
                result = provider(did, request)
            except TypeError:
                result = provider(did)

        if inspect.isawaitable(result):
            return await result
        return result

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


def get_verified_agent(request: Request) -> VerificationResult | None:
    """
    FastAPI dependency to get verified agent info.

    Example:
        ```python
        from fastapi import FastAPI, Depends
        from agentfacts.middleware import get_verified_agent

        app = FastAPI()

        @app.get("/data")
        def get_data(agent: VerificationResult = Depends(get_verified_agent)):
            if agent and agent.valid:
                return {"message": f"Hello, agent {agent.did}"}
            return {"message": "Hello, anonymous"}
        ```
    """
    return getattr(request.state, AGENT_STATE_KEY, None)


def require_verified_agent(
    policy: Policy | None = None,
) -> Callable[[Request], Awaitable[VerificationResult]]:
    """
    FastAPI dependency that requires a verified agent.

    Raises HTTPException 403 if agent is not verified.

    Example:
        ```python
        from fastapi import FastAPI, Depends
        from agentfacts.middleware import require_verified_agent
        from agentfacts.policy import Policy

        app = FastAPI()

        @app.get("/secure")
        def secure_endpoint(
            agent: VerificationResult = Depends(require_verified_agent())
        ):
            return {"message": f"Verified agent: {agent.did}"}
        ```
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required")

    async def dependency(request: Request) -> VerificationResult:
        agent = getattr(request.state, AGENT_STATE_KEY, None)

        if not isinstance(agent, VerificationResult) or not agent.valid:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Agent Identity Unverified",
                    "code": 403,
                    "message": "This endpoint requires verified agent identity.",
                },
            )

        return agent

    return dependency
