"""
Tests for the plugin system (adapters, verification context, registry).
"""

from datetime import datetime, timezone

from agentfacts import AgentFacts
from agentfacts.models import (
    Attestation,
    BaselineModel,
    Capability,
    ModelProvider,
)
from agentfacts.plugins import (
    AttestationPayload,
    AttestationVerificationResult,
    DenyCapabilityIR,
    # Registry
    PluginRegistry,
    RequireAttestationIR,
    RequireCapabilityIR,
    RequireComplianceIR,
    # Policy IR
    RequireIssuerIR,
    RequireStatusNotRevokedIR,
    # Interfaces
    ResolvedDID,
    StatusCheckResult,
    # Context
    VerificationContext,
    get_plugin_registry,
    reset_plugin_registry,
)
from agentfacts.policy.engine import PolicyEngine, ir_to_rule

# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockDIDResolver:
    """Mock DID resolver for testing."""

    def __init__(self, public_key: str):
        self.public_key = public_key
        self.resolve_count = 0

    def supports(self, did: str) -> bool:
        return did.startswith("did:mock:")

    def resolve(self, did: str) -> ResolvedDID:
        self.resolve_count += 1
        return ResolvedDID(
            did=did,
            public_key_base64=self.public_key,
            metadata={"resolved": True},
        )


class MockAttestationVerifier:
    """Mock attestation verifier for testing."""

    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
        self.verify_count = 0

    @property
    def formats(self) -> set[str]:
        return {"mock-vc", "test-vc"}

    def verify(
        self, attestation: Attestation, context: VerificationContext
    ) -> AttestationVerificationResult:
        self.verify_count += 1
        if self.should_pass:
            return AttestationVerificationResult(
                valid=True,
                payload=AttestationPayload(
                    format=attestation.format or "mock-vc",
                    issuer=attestation.issuer,
                    subject=attestation.subject,
                    claims=attestation.claims,
                ),
            )
        return AttestationVerificationResult(
            valid=False,
            errors=["Mock verification failed"],
        )


class MockStatusChecker:
    """Mock status checker for testing."""

    def __init__(self, status: str = "active"):
        self.status = status
        self.check_count = 0

    def supports(self, status_ref: str) -> bool:
        return status_ref.startswith("mock://")

    def check(self, status_ref: str, context: VerificationContext) -> StatusCheckResult:
        self.check_count += 1
        is_valid = self.status == "active"
        return StatusCheckResult(
            valid=is_valid,
            status=self.status,
            checked_at=datetime.now(timezone.utc),
        )


class MockGovernanceAdapter:
    """Mock governance adapter for testing."""

    def to_policy_ir(self, doc: dict) -> list:
        policies = []
        if "required_issuer" in doc:
            policies.append(RequireIssuerIR(did=doc["required_issuer"]))
        if "required_compliance" in doc:
            policies.append(RequireComplianceIR(framework=doc["required_compliance"]))
        if "denied_capability" in doc:
            policies.append(DenyCapabilityIR(capability=doc["denied_capability"]))
        return policies


# =============================================================================
# VerificationContext Tests
# =============================================================================


class TestVerificationContext:
    """Tests for VerificationContext."""

    def test_default_context(self):
        """Test creating a default context with no plugins."""
        context = VerificationContext()
        assert context.did_resolver is None
        assert context.attestation_verifiers == []
        assert context.status_checkers == []
        assert context.governance_adapter is None

    def test_context_with_resolver(self):
        """Test context with a DID resolver."""
        resolver = MockDIDResolver("test_key")
        context = VerificationContext(did_resolver=resolver)

        assert context.get_did_resolver() is resolver
        assert resolver.supports("did:mock:test")
        assert not resolver.supports("did:web:example.com")

    def test_context_get_verifier_for_format(self):
        """Test finding verifier by format."""
        verifier = MockAttestationVerifier()
        context = VerificationContext(attestation_verifiers=[verifier])

        assert context.get_verifier_for_format("mock-vc") is verifier
        assert context.get_verifier_for_format("test-vc") is verifier
        assert context.get_verifier_for_format("unknown") is None

    def test_context_get_status_checker(self):
        """Test finding status checker."""
        checker = MockStatusChecker()
        context = VerificationContext(status_checkers=[checker])

        assert context.get_status_checker("mock://status/1") is checker
        assert context.get_status_checker("https://other.com") is None

    def test_context_get_policy_ir(self):
        """Test getting policy IR from governance doc."""
        adapter = MockGovernanceAdapter()
        doc = {
            "required_issuer": "did:key:z6MkTest",
            "required_compliance": "EU_AI_ACT",
        }
        context = VerificationContext(
            governance_adapter=adapter,
            governance_doc=doc,
        )

        policies = context.get_policy_ir()
        assert len(policies) == 2
        assert isinstance(policies[0], RequireIssuerIR)
        assert policies[0].did == "did:key:z6MkTest"
        assert isinstance(policies[1], RequireComplianceIR)
        assert policies[1].framework == "EU_AI_ACT"

    def test_context_get_policy_ir_no_adapter(self):
        """Test policy IR returns empty when no adapter."""
        context = VerificationContext(governance_doc={"some": "doc"})
        assert context.get_policy_ir() == []

    def test_context_get_time(self):
        """Test custom clock function."""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        context = VerificationContext(clock=lambda: fixed_time)
        assert context.get_time() == fixed_time


# =============================================================================
# PolicyIR Tests
# =============================================================================


class TestPolicyIR:
    """Tests for Policy IR conversion."""

    def test_ir_to_rule_require_issuer(self):
        """Test RequireIssuerIR conversion."""
        ir = RequireIssuerIR(did="did:key:z6MkTest")
        rule = ir_to_rule(ir)
        assert rule.name == "require_issuer"

    def test_ir_to_rule_require_attestation(self):
        """Test RequireAttestationIR conversion."""
        ir = RequireAttestationIR(
            attestation_type="security_audit",
            issuer="did:key:z6MkAuditor",
            max_age_days=90,
        )
        rule = ir_to_rule(ir)
        assert rule.name == "require_attestation"

    def test_ir_to_rule_require_capability(self):
        """Test RequireCapabilityIR conversion."""
        ir = RequireCapabilityIR(capability="web_search")
        rule = ir_to_rule(ir)
        assert rule.name == "require_capability"

    def test_ir_to_rule_require_risk_level(self):
        """Test RequireCapabilityIR with max_risk_level."""
        ir = RequireCapabilityIR(max_risk_level="medium")
        rule = ir_to_rule(ir)
        assert rule.name == "require_risk_level"

    def test_ir_to_rule_require_compliance(self):
        """Test RequireComplianceIR conversion."""
        ir = RequireComplianceIR(framework="GDPR")
        rule = ir_to_rule(ir)
        assert rule.name == "require_compliance"

    def test_ir_to_rule_deny_capability(self):
        """Test DenyCapabilityIR conversion."""
        ir = DenyCapabilityIR(capability="shell_exec")
        rule = ir_to_rule(ir)
        assert rule.name == "deny_capability"

    def test_ir_to_rule_status_not_revoked(self):
        """Test RequireStatusNotRevokedIR conversion."""
        ir = RequireStatusNotRevokedIR()
        rule = ir_to_rule(ir)
        # Status checking is done in pipeline, rule always passes
        assert rule.name == "always_pass"


class TestPolicyEngineEvaluateIR:
    """Tests for PolicyEngine.evaluate_ir()."""

    def test_evaluate_ir_empty(self):
        """Test evaluating empty IR list."""
        engine = PolicyEngine()
        facts = AgentFacts(
            name="Test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        result = engine.evaluate_ir(facts.metadata, [])
        assert result.passed

    def test_evaluate_ir_require_compliance_pass(self):
        """Test require compliance that passes."""
        engine = PolicyEngine()
        facts = AgentFacts(
            name="Test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.policy.compliance.frameworks = ["GDPR", "SOC2"]

        result = engine.evaluate_ir(
            facts.metadata,
            [
                RequireComplianceIR(framework="GDPR"),
            ],
        )
        assert result.passed

    def test_evaluate_ir_require_compliance_fail(self):
        """Test require compliance that fails."""
        engine = PolicyEngine()
        facts = AgentFacts(
            name="Test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        result = engine.evaluate_ir(
            facts.metadata,
            [
                RequireComplianceIR(framework="EU_AI_ACT"),
            ],
        )
        assert not result.passed
        assert len(result.violations) == 1

    def test_evaluate_ir_deny_capability(self):
        """Test deny capability."""
        engine = PolicyEngine()
        facts = AgentFacts(
            name="Test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[
                Capability(name="shell_exec", description="Execute shell commands"),
            ],
        )

        result = engine.evaluate_ir(
            facts.metadata,
            [
                DenyCapabilityIR(capability="shell_exec"),
            ],
        )
        assert not result.passed

    def test_evaluate_ir_require_attestation_format_issuer(self):
        """Test require attestation with format/issuer only."""
        engine = PolicyEngine()
        facts = AgentFacts(
            name="Test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.attestations.append(
            Attestation(
                id="att-1",
                type="security_audit",
                issuer="did:key:issuer",
                subject=facts.did,
                format="sd-jwt-vc",
            )
        )

        result = engine.evaluate_ir(
            facts.metadata,
            [
                RequireAttestationIR(format="sd-jwt-vc", issuer="did:key:issuer"),
            ],
        )
        assert result.passed

        result = engine.evaluate_ir(
            facts.metadata,
            [
                RequireAttestationIR(format="anoncreds", issuer="did:key:issuer"),
            ],
        )
        assert not result.passed


# =============================================================================
# Plugin Registry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_plugin_registry()

    def test_registry_singleton(self):
        """Test that get_plugin_registry returns same instance."""
        reg1 = get_plugin_registry()
        reg2 = get_plugin_registry()
        assert reg1 is reg2

    def test_register_did_resolver(self):
        """Test registering a DID resolver."""
        registry = PluginRegistry()
        resolver = MockDIDResolver("test_key")

        registry.register_did_resolver("mock", resolver)
        assert "mock" in registry.registered_did_methods

        found = registry.get_did_resolver("did:mock:test")
        assert found is resolver

    def test_unregister_did_resolver(self):
        """Test unregistering a DID resolver."""
        registry = PluginRegistry()
        resolver = MockDIDResolver("test_key")

        registry.register_did_resolver("mock", resolver)
        assert registry.unregister_did_resolver("mock")
        assert "mock" not in registry.registered_did_methods
        assert not registry.unregister_did_resolver("mock")  # Already removed

    def test_register_attestation_verifier(self):
        """Test registering an attestation verifier."""
        registry = PluginRegistry()
        verifier = MockAttestationVerifier()

        registry.register_attestation_verifier(verifier)
        assert "mock-vc" in registry.registered_formats

        found = registry.get_attestation_verifier("mock-vc")
        assert found is verifier

    def test_register_status_checker(self):
        """Test registering a status checker."""
        registry = PluginRegistry()
        checker = MockStatusChecker()

        registry.register_status_checker(checker)

        found = registry.get_status_checker("mock://status/1")
        assert found is checker

    def test_register_governance_adapter(self):
        """Test registering a governance adapter."""
        registry = PluginRegistry()
        adapter = MockGovernanceAdapter()

        registry.register_governance_adapter("mock", adapter)
        assert "mock" in registry.registered_governance_adapters

        found = registry.get_governance_adapter("mock")
        assert found is adapter

    def test_create_context(self):
        """Test creating context from registry."""
        registry = PluginRegistry()
        resolver = MockDIDResolver("test_key")
        verifier = MockAttestationVerifier()
        checker = MockStatusChecker()
        adapter = MockGovernanceAdapter()

        registry.register_did_resolver("mock", resolver)
        registry.register_attestation_verifier(verifier)
        registry.register_status_checker(checker)
        registry.register_governance_adapter("mock", adapter)

        context = registry.create_context(
            governance_adapter_name="mock",
            governance_doc={"required_issuer": "did:key:z6MkTest"},
        )

        assert context.did_resolver is not None
        assert len(context.attestation_verifiers) == 1
        assert len(context.status_checkers) == 1
        assert context.governance_adapter is adapter
        assert context.governance_doc == {"required_issuer": "did:key:z6MkTest"}

    def test_registry_clear(self):
        """Test clearing all registered plugins."""
        registry = PluginRegistry()
        registry.register_did_resolver("mock", MockDIDResolver("key"))
        registry.register_attestation_verifier(MockAttestationVerifier())

        registry.clear()

        assert registry.registered_did_methods == []
        assert registry.registered_formats == set()


# =============================================================================
# Attestation Model Extension Tests
# =============================================================================


class TestAttestationExtension:
    """Tests for extended Attestation model fields."""

    def test_attestation_with_vc_fields(self):
        """Test creating attestation with VC fields."""
        attestation = Attestation(
            id="att-001",
            type="security_audit",
            issuer="did:key:z6MkAuditor",
            subject="did:key:z6MkAgent",
            format="sd-jwt-vc",
            payload={"type": "VerifiableCredential"},
            status_ref="mock://status/1",
            proof_type="Ed25519Signature2020",
        )

        assert attestation.format == "sd-jwt-vc"
        assert attestation.payload == {"type": "VerifiableCredential"}
        assert attestation.status_ref == "mock://status/1"
        assert attestation.proof_type == "Ed25519Signature2020"

    def test_attestation_backwards_compatible(self):
        """Test attestation without new fields still works."""
        attestation = Attestation(
            id="att-001",
            type="security_audit",
            issuer="did:key:z6MkAuditor",
            subject="did:key:z6MkAgent",
        )

        assert attestation.format is None
        assert attestation.payload is None
        assert attestation.status_ref is None
        assert attestation.proof_type is None


# =============================================================================
# Enhanced Verification Tests
# =============================================================================


class TestEnhancedVerification:
    """Tests for verify() with VerificationContext."""

    def test_verify_default_path_unchanged(self):
        """Test that verification without context works as before."""
        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.sign()

        # Verify without context
        result = facts.verify()
        assert result.valid

    def test_verify_with_empty_context(self):
        """Test verification with empty context."""
        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.sign()

        context = VerificationContext()
        result = facts.verify(context=context)
        assert result.valid

    def test_verify_with_attestation_verifier(self):
        """Test verification with attestation verifier."""
        verifier = MockAttestationVerifier(should_pass=True)

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        # Add attestation with external format
        facts.add_attestation(
            Attestation(
                id="att-001",
                type="security_audit",
                issuer="did:key:z6MkAuditor",
                subject=facts.did,
                format="mock-vc",
                payload={"verified": True},
            )
        )
        facts.sign()

        context = VerificationContext(attestation_verifiers=[verifier])
        result = facts.verify(context=context)

        assert result.valid
        assert verifier.verify_count == 1

    def test_verify_attestation_fails(self):
        """Test verification fails when attestation verification fails."""
        verifier = MockAttestationVerifier(should_pass=False)

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        facts.add_attestation(
            Attestation(
                id="att-001",
                type="security_audit",
                issuer="did:key:z6MkAuditor",
                subject=facts.did,
                format="mock-vc",
            )
        )
        facts.sign()

        context = VerificationContext(attestation_verifiers=[verifier])
        result = facts.verify(context=context)

        assert not result.valid
        assert any("Mock verification failed" in e for e in result.errors)

    def test_verify_with_status_checker(self):
        """Test verification with status checker."""
        checker = MockStatusChecker(status="active")

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        facts.add_attestation(
            Attestation(
                id="att-001",
                type="security_audit",
                issuer="did:key:z6MkAuditor",
                subject=facts.did,
                status_ref="mock://status/1",
            )
        )
        facts.sign()

        context = VerificationContext(status_checkers=[checker])
        result = facts.verify(context=context)

        assert result.valid
        assert checker.check_count == 1

    def test_verify_status_revoked(self):
        """Test verification fails when status is revoked."""
        checker = MockStatusChecker(status="revoked")

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        facts.add_attestation(
            Attestation(
                id="att-001",
                type="security_audit",
                issuer="did:key:z6MkAuditor",
                subject=facts.did,
                status_ref="mock://status/1",
            )
        )
        facts.sign()

        context = VerificationContext(status_checkers=[checker])
        result = facts.verify(context=context)

        assert not result.valid
        assert any("revoked" in e for e in result.errors)

    def test_verify_with_governance_policy(self):
        """Test verification with governance policy."""
        adapter = MockGovernanceAdapter()
        doc = {"required_compliance": "GDPR"}

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.policy.compliance.frameworks = ["GDPR"]
        facts.sign()

        context = VerificationContext(
            governance_adapter=adapter,
            governance_doc=doc,
        )
        result = facts.verify(context=context)

        assert result.valid
        assert result.policy_violations == []

    def test_verify_governance_policy_violation(self):
        """Test policy violations are reported."""
        adapter = MockGovernanceAdapter()
        doc = {"denied_capability": "shell_exec"}

        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[
                Capability(name="shell_exec", description="Execute shell commands"),
            ],
        )
        facts.sign()

        context = VerificationContext(
            governance_adapter=adapter,
            governance_doc=doc,
        )
        result = facts.verify(context=context)

        # Signature is valid but policy is violated
        assert result.valid  # Base verification passes
        assert len(result.policy_violations) == 1
        assert "shell_exec" in result.policy_violations[0]

    def test_verify_no_verifier_for_format_warns(self):
        """Test warning when no verifier for attestation format."""
        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        facts.add_attestation(
            Attestation(
                id="att-001",
                type="security_audit",
                issuer="did:key:z6MkAuditor",
                subject=facts.did,
                format="unknown-format",
            )
        )
        facts.sign()

        context = VerificationContext()  # No verifiers
        result = facts.verify(context=context)

        assert result.valid  # Should still pass
        assert any("No verifier" in w for w in result.warnings)
