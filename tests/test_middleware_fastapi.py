import importlib

import pytest

from agentfacts.core import AgentFacts
from agentfacts.middleware import FastAPIMiddleware, require_verified_agent
from agentfacts.middleware.headers import (
    HEADER_AGENT_DID,
    HEADER_AGENT_PUBLIC_KEY,
    HEADER_AGENT_SIGNATURE,
)
from agentfacts.models import BaselineModel, ModelProvider
from agentfacts.policy import Policy, RequireProvider

fastapi = pytest.importorskip("fastapi")
Depends = fastapi.Depends
FastAPI = fastapi.FastAPI
TestClient = importlib.import_module("fastapi.testclient").TestClient


def _signed_facts(provider: ModelProvider) -> AgentFacts:
    facts = AgentFacts(
        name="Test Agent",
        baseline_model=BaselineModel(name="gpt-4", provider=provider),
    )
    facts.sign()
    return facts


def _headers(facts: AgentFacts) -> dict[str, str]:
    return {
        HEADER_AGENT_DID: facts.did,
        HEADER_AGENT_PUBLIC_KEY: facts.public_key,
        HEADER_AGENT_SIGNATURE: facts.signature or "",
    }


def test_fastapi_full_verify_success() -> None:
    facts = _signed_facts(ModelProvider.OPENAI)

    def metadata_provider(did: str):
        assert did == facts.did
        return facts.to_dict()

    app = FastAPI()
    app.add_middleware(
        FastAPIMiddleware,
        full_verify=True,
        metadata_provider=metadata_provider,
        policy=Policy.basic_trust(),
    )

    verified_agent_dep = Depends(require_verified_agent())

    @app.get("/secure")
    async def secure_endpoint(agent=verified_agent_dep):
        return {"did": agent.did}

    client = TestClient(app)
    response = client.get("/secure", headers=_headers(facts))

    assert response.status_code == 200
    assert response.json()["did"] == facts.did


def test_fastapi_full_verify_policy_violation() -> None:
    facts = _signed_facts(ModelProvider.UNKNOWN)

    def metadata_provider(did: str):
        assert did == facts.did
        return facts.to_dict()

    policy = Policy(
        name="require_openai",
        rules=[RequireProvider([ModelProvider.OPENAI])],
    )

    app = FastAPI()
    app.add_middleware(
        FastAPIMiddleware,
        full_verify=True,
        metadata_provider=metadata_provider,
        policy=policy,
    )

    verified_agent_dep = Depends(require_verified_agent())

    @app.get("/secure")
    async def secure_endpoint(agent=verified_agent_dep):
        return {"did": agent.did}

    client = TestClient(app)
    response = client.get("/secure", headers=_headers(facts))

    assert response.status_code == 403
    body = response.json()
    assert body["message"] == "Agent metadata failed policy evaluation."
