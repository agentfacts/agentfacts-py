"""Tests for policy-based verification."""

from datetime import datetime, timezone

from agentfacts.models import (
    AgentFactsCard,
    AgentInfo,
    Attestation,
    BaselineModel,
    Capability,
    ComplianceInfo,
    LogProof,
    ModelProvider,
    OperationalConstraints,
    Publisher,
    SignatureBlock,
)
from agentfacts.models import (
    Policy as PolicyModel,
)
from agentfacts.policy.engine import PolicyEngine
from agentfacts.policy.rules import (
    DenyCapability,
    Policy,
    RequireAttestation,
    RequireCapability,
    RequireCompliance,
    RequireModel,
    RequireProvider,
    RequireRiskLevel,
    RequireSignature,
)


def make_metadata(**kwargs) -> AgentFactsCard:
    """Helper to create test AgentFacts cards."""
    did = kwargs.pop("did", "did:key:z123")
    name = kwargs.pop("name", "Test Agent")
    baseline_model = kwargs.pop(
        "baseline_model",
        BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )
    capabilities = kwargs.pop("capabilities", [])
    compliance = kwargs.pop("compliance", ComplianceInfo())
    constraints = kwargs.pop("constraints", OperationalConstraints())
    attestations = kwargs.pop("attestations", [])
    publisher = kwargs.pop("publisher", Publisher(id=did))
    policy = kwargs.pop(
        "policy", PolicyModel(compliance=compliance, constraints=constraints)
    )

    signature_value = kwargs.pop("signature", None)
    log_proof = kwargs.pop("log_proof", None)
    signed = kwargs.pop("signed", False)

    signature_block = None
    if isinstance(signature_value, SignatureBlock):
        signature_block = signature_value
    elif isinstance(signature_value, str):
        signature_block = SignatureBlock(
            alg="ed25519",
            key_id=f"{did}#sig-1",
            value=signature_value,
        )
    elif signed:
        signature_block = SignatureBlock(
            alg="ed25519",
            key_id=f"{did}#sig-1",
            value="signed",
        )
    else:
        signature_block = SignatureBlock(
            alg="ed25519",
            key_id=f"{did}#sig-1",
            value="",
        )

    log_proof_block = None
    if isinstance(log_proof, LogProof):
        log_proof_block = log_proof
    else:
        log_proof_block = LogProof(
            log_id="local",
            leaf_hash="00" if signature_block.value else "",
            root_hash="00" if signature_block.value else "",
            inclusion=[],
        )

    return AgentFactsCard(
        agent=AgentInfo(
            id=did,
            name=name,
            model=baseline_model,
            capabilities=capabilities,
        ),
        publisher=publisher,
        policy=policy,
        attestations=attestations,
        signature=signature_block,
        log_proof=log_proof_block,
    )


class TestPolicyRules:
    """Tests for individual policy rules."""

    def test_require_signature_fails_unsigned(self):
        """Test RequireSignature fails for unsigned metadata."""
        rule = RequireSignature()
        metadata = make_metadata()

        result = rule.evaluate(metadata)

        assert not result.passed
        assert len(result.violations) == 1

    def test_require_signature_passes_signed(self):
        """Test RequireSignature passes for signed metadata."""
        rule = RequireSignature()
        metadata = make_metadata(signature="abc123")

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_provider_passes(self):
        """Test RequireProvider passes for allowed provider."""
        rule = RequireProvider([ModelProvider.OPENAI, ModelProvider.ANTHROPIC])
        metadata = make_metadata()

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_provider_fails(self):
        """Test RequireProvider fails for disallowed provider."""
        rule = RequireProvider([ModelProvider.ANTHROPIC])
        metadata = make_metadata(
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI)
        )

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_model_passes(self):
        """Test RequireModel passes for allowed model."""
        rule = RequireModel(["gpt-4", "gpt-3.5"])
        metadata = make_metadata()

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_model_prefix_match(self):
        """Test RequireModel with prefix matching."""
        rule = RequireModel(["gpt"], allow_prefix_match=True)
        metadata = make_metadata(
            baseline_model=BaselineModel(
                name="gpt-4-turbo", provider=ModelProvider.OPENAI
            )
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_model_fails(self):
        """Test RequireModel fails for disallowed model."""
        rule = RequireModel(["claude"], allow_prefix_match=False)
        metadata = make_metadata()

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_attestation_passes(self):
        """Test RequireAttestation passes when attestation exists."""
        rule = RequireAttestation("security_audit")
        metadata = make_metadata(
            attestations=[
                Attestation(
                    id="att1",
                    type="security_audit",
                    issuer="did:key:auditor",
                    subject="did:key:z123",
                )
            ]
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_attestation_fails_missing(self):
        """Test RequireAttestation fails when attestation is missing."""
        rule = RequireAttestation("security_audit")
        metadata = make_metadata()

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_attestation_with_issuer(self):
        """Test RequireAttestation with specific issuer."""
        rule = RequireAttestation("security_audit", issuer_did="did:key:trusted")
        metadata = make_metadata(
            attestations=[
                Attestation(
                    id="att1",
                    type="security_audit",
                    issuer="did:key:untrusted",
                    subject="did:key:z123",
                )
            ]
        )

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_capability_passes(self):
        """Test RequireCapability passes when capability exists."""
        rule = RequireCapability(["search", "calculator"])
        metadata = make_metadata(
            capabilities=[
                Capability(name="search", description="Search"),
                Capability(name="calculator", description="Math"),
            ]
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_capability_fails(self):
        """Test RequireCapability fails when capability is missing."""
        rule = RequireCapability(["search", "database"])
        metadata = make_metadata(
            capabilities=[
                Capability(name="search", description="Search"),
            ]
        )

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_deny_capability_passes(self):
        """Test DenyCapability passes when capability is absent."""
        rule = DenyCapability(["shell", "sudo"])
        metadata = make_metadata(
            capabilities=[
                Capability(name="search", description="Search"),
            ]
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_deny_capability_fails(self):
        """Test DenyCapability fails when denied capability exists."""
        rule = DenyCapability(["shell"])
        metadata = make_metadata(
            capabilities=[
                Capability(name="shell", description="Execute commands"),
            ]
        )

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_compliance_passes(self):
        """Test RequireCompliance passes with required frameworks."""
        rule = RequireCompliance(["EU_AI_ACT", "SOC2"])
        metadata = make_metadata(
            compliance=ComplianceInfo(frameworks=["EU_AI_ACT", "SOC2", "GDPR"])
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_compliance_fails(self):
        """Test RequireCompliance fails with missing frameworks."""
        rule = RequireCompliance(["EU_AI_ACT", "HIPAA"])
        metadata = make_metadata(compliance=ComplianceInfo(frameworks=["EU_AI_ACT"]))

        result = rule.evaluate(metadata)

        assert not result.passed

    def test_require_risk_level_passes(self):
        """Test RequireRiskLevel passes with acceptable risk."""
        rule = RequireRiskLevel("medium")
        metadata = make_metadata(
            capabilities=[
                Capability(name="search", risk_level="low"),
                Capability(name="api", risk_level="medium"),
            ]
        )

        result = rule.evaluate(metadata)

        assert result.passed

    def test_require_risk_level_fails(self):
        """Test RequireRiskLevel fails with high risk capability."""
        rule = RequireRiskLevel("medium")
        metadata = make_metadata(
            capabilities=[
                Capability(name="shell", risk_level="high"),
            ]
        )

        result = rule.evaluate(metadata)

        assert not result.passed


class TestPolicy:
    """Tests for composite policies."""

    def test_policy_all_rules_pass(self):
        """Test policy with all rules passing."""
        policy = Policy(
            name="test",
            rules=[
                RequireSignature(),
                RequireProvider([ModelProvider.OPENAI]),
            ],
        )
        metadata = make_metadata(signature="abc")

        result = policy.evaluate(metadata)

        assert result.passed

    def test_policy_one_rule_fails(self):
        """Test policy fails when one rule fails."""
        policy = Policy(
            name="test",
            rules=[
                RequireSignature(),
                RequireProvider([ModelProvider.ANTHROPIC]),
            ],
        )
        metadata = make_metadata(signature="abc")

        result = policy.evaluate(metadata)

        assert not result.passed
        assert len(result.violations) == 1

    def test_policy_or_logic(self):
        """Test policy with OR logic (any rule passes)."""
        policy = Policy(
            name="test",
            rules=[
                RequireProvider([ModelProvider.ANTHROPIC]),
                RequireProvider([ModelProvider.OPENAI]),
            ],
            require_all=False,
        )
        metadata = make_metadata()

        result = policy.evaluate(metadata)

        assert result.passed

    def test_basic_trust_policy(self):
        """Test pre-built basic trust policy."""
        policy = Policy.basic_trust()
        signed_metadata = make_metadata(signature="abc")
        unsigned_metadata = make_metadata()

        assert policy.evaluate(signed_metadata).passed
        assert not policy.evaluate(unsigned_metadata).passed

    def test_strict_enterprise_policy(self):
        """Test pre-built strict enterprise policy."""
        policy = Policy.strict_enterprise()
        metadata = make_metadata(
            signature="abc",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            attestations=[
                Attestation(
                    id="att1",
                    type="security_audit",
                    issuer="did:key:auditor",
                    subject="did:key:z123",
                    issued_at=datetime.now(timezone.utc),
                )
            ],
        )

        result = policy.evaluate(metadata)

        assert result.passed


class TestPolicyEngine:
    """Tests for the policy engine."""

    def test_register_and_evaluate(self):
        """Test registering and evaluating policies."""
        engine = PolicyEngine()
        engine.register_policy(Policy.basic_trust())
        engine.set_default_policy(Policy.basic_trust())

        metadata = make_metadata(signature="abc")

        result = engine.evaluate(metadata)

        assert result.passed

    def test_evaluate_by_name(self):
        """Test evaluating specific named policy."""
        engine = PolicyEngine()
        engine.register_policy(Policy.basic_trust())
        engine.register_policy(Policy.strict_enterprise())

        metadata = make_metadata(signature="abc")

        # Basic should pass
        assert engine.evaluate(metadata, "basic_trust").passed

        # Strict should fail (no attestation)
        assert not engine.evaluate(metadata, "strict_enterprise").passed

    def test_evaluate_all(self):
        """Test evaluating all registered policies."""
        engine = PolicyEngine.with_defaults()
        metadata = make_metadata(signature="abc")

        results = engine.evaluate_all(metadata)

        assert "basic_trust" in results
        assert "strict_enterprise" in results

    def test_is_trusted(self):
        """Test quick trust check."""
        engine = PolicyEngine()
        engine.register_policy(Policy.basic_trust())
        engine.set_default_policy(Policy.basic_trust())

        signed = make_metadata(signature="abc")
        unsigned = make_metadata()

        assert engine.is_trusted(signed)
        assert not engine.is_trusted(unsigned)

    def test_strict_mode_requires_all_policies(self):
        """Test strict mode evaluates all registered policies."""
        engine = PolicyEngine(strict_mode=True)
        engine.register_policy(Policy.basic_trust())
        engine.register_policy(Policy.strict_enterprise())

        metadata = make_metadata(signature="abc")

        result = engine.evaluate(metadata)

        assert not result.passed

    def test_create_403_response(self):
        """Test creating rejection response."""
        engine = PolicyEngine()

        from agentfacts.policy.rules import PolicyResult, PolicyViolation

        result = PolicyResult(
            passed=False,
            violations=[PolicyViolation(rule="test", message="Failed")],
        )

        response = engine.create_403_response(result)

        assert response["code"] == 403
        assert "Agent Identity Unverified" in response["error"]
