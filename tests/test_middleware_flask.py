import pytest

from agentfacts.core import AgentFacts
from agentfacts.middleware import FlaskMiddleware, get_flask_verified_agent
from agentfacts.middleware.headers import (
    HEADER_AGENT_DID,
    HEADER_AGENT_PUBLIC_KEY,
    HEADER_AGENT_SIGNATURE,
)
from agentfacts.models import BaselineModel, ModelProvider
from agentfacts.policy import Policy, RequireProvider

flask = pytest.importorskip("flask")
Flask = flask.Flask


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


def test_flask_full_verify_success() -> None:
    facts = _signed_facts(ModelProvider.OPENAI)

    def metadata_provider(did: str, request):
        assert did == facts.did
        return facts.to_dict()

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

    assert response.status_code == 200
    assert response.json["did"] == facts.did


def test_flask_full_verify_policy_violation() -> None:
    facts = _signed_facts(ModelProvider.UNKNOWN)

    def metadata_provider(did: str, request):
        assert did == facts.did
        return facts.to_dict()

    policy = Policy(
        name="require_openai",
        rules=[RequireProvider([ModelProvider.OPENAI])],
    )

    app = Flask(__name__)
    FlaskMiddleware(
        app,
        full_verify=True,
        metadata_provider=metadata_provider,
        policy=policy,
    )

    @app.route("/secure")
    def secure_endpoint():
        agent = get_flask_verified_agent()
        return {"did": agent.did}

    client = app.test_client()
    response = client.get("/secure", headers=_headers(facts))

    assert response.status_code == 403
    body = response.json
    assert body["message"] == "Agent metadata failed policy evaluation."
