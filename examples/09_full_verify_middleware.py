"""
Example: Full verification middleware with an in-memory metadata provider.

Requires agentfacts[middleware]. Runs quick checks using FastAPI and Flask
test clients to show signed metadata verification + policy enforcement.
"""

from agentfacts import AgentFacts
from agentfacts.middleware.headers import (
    HEADER_AGENT_DID,
    HEADER_AGENT_PUBLIC_KEY,
    HEADER_AGENT_SIGNATURE,
)
from agentfacts.models import BaselineModel, ModelProvider
from agentfacts.policy import Policy


def _signed_facts() -> AgentFacts:
    facts = AgentFacts(
        name="In-Memory Agent",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )
    facts.sign()
    return facts


def _headers(facts: AgentFacts) -> dict[str, str]:
    return {
        HEADER_AGENT_DID: facts.did,
        HEADER_AGENT_PUBLIC_KEY: facts.public_key,
        HEADER_AGENT_SIGNATURE: facts.signature or "",
    }


def run_fastapi_demo(facts: AgentFacts, metadata_provider) -> None:
    try:
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from agentfacts.middleware import FastAPIMiddleware, require_verified_agent
    except ImportError:
        print("FastAPI not installed; skipping FastAPI demo.")
        return

    app = FastAPI()
    app.add_middleware(
        FastAPIMiddleware,
        full_verify=True,
        metadata_provider=metadata_provider,
        policy=Policy.basic_trust(),
    )

    @app.get("/secure")
    def secure_endpoint(agent=Depends(require_verified_agent())):
        return {"did": agent.did}

    client = TestClient(app)
    response = client.get("/secure", headers=_headers(facts))
    print("FastAPI:", response.status_code, response.json())


def run_flask_demo(facts: AgentFacts, metadata_provider) -> None:
    try:
        from flask import Flask
        from agentfacts.middleware import FlaskMiddleware, get_flask_verified_agent
    except ImportError:
        print("Flask not installed; skipping Flask demo.")
        return

    app = Flask(__name__)
    FlaskMiddleware(
        app,
        full_verify=True,
        metadata_provider=metadata_provider,
        policy=Policy.basic_trust(),
    )

    @app.route("/secure")
    def secure_endpoint():
        agent = get_flask_verified_agent()
        return {"did": agent.did}

    client = app.test_client()
    response = client.get("/secure", headers=_headers(facts))
    print("Flask:", response.status_code, response.json)


def main() -> None:
    facts = _signed_facts()
    metadata_store = {facts.did: facts.to_dict()}

    def metadata_provider(did: str, request=None):
        return metadata_store.get(did)

    run_fastapi_demo(facts, metadata_provider)
    run_flask_demo(facts, metadata_provider)


if __name__ == "__main__":
    main()
