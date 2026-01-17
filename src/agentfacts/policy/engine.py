"""
Policy engine for evaluating agents against trust policies.

Provides a high-level interface for policy-based verification
including caching and batch evaluation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentfacts.models import AgentFactsCard, VerificationResult
from agentfacts.policy.rules import Policy, PolicyResult, PolicyRule, PolicyViolation

if TYPE_CHECKING:
    from agentfacts.plugins import PolicyIR


@dataclass
class PolicyEngine:
    """
    Engine for evaluating agents against policies.

    Supports multiple policies with different enforcement levels
    and provides detailed evaluation results.
    """

    default_policy: Policy | None = None
    policies: dict[str, Policy] = field(default_factory=dict)
    strict_mode: bool = False  # Reject on any policy failure

    def register_policy(self, policy: Policy) -> None:
        """Register a named policy."""
        self.policies[policy.name] = policy

    def set_default_policy(self, policy: Policy) -> None:
        """Set the default policy for evaluation."""
        self.default_policy = policy

    def evaluate(
        self,
        metadata: AgentFactsCard,
        policy_name: str | None = None,
    ) -> PolicyResult:
        """
        Evaluate agent metadata against a policy.

        Args:
            metadata: The agent metadata to evaluate
            policy_name: Name of policy to use (uses default if not specified)

        Returns:
            PolicyResult with pass/fail status and any violations
        """
        if self.strict_mode and policy_name is None:
            return self._evaluate_strict(metadata)

        if policy_name:
            policy = self.policies.get(policy_name)
            if not policy:
                raise ValueError(f"Unknown policy: {policy_name}")
        elif self.default_policy:
            policy = self.default_policy
        else:
            raise ValueError("No policy specified and no default policy set")

        return policy.evaluate(metadata)

    def evaluate_all(
        self,
        metadata: AgentFactsCard,
    ) -> dict[str, PolicyResult]:
        """
        Evaluate agent metadata against all registered policies.

        Args:
            metadata: The agent metadata to evaluate

        Returns:
            Dict mapping policy names to results
        """
        results = {}
        for name, policy in self.policies.items():
            results[name] = policy.evaluate(metadata)
        return results

    def is_trusted(
        self,
        metadata: AgentFactsCard,
        policy_name: str | None = None,
    ) -> bool:
        """
        Quick check if an agent is trusted according to a policy.

        Args:
            metadata: The agent metadata to check
            policy_name: Name of policy to use

        Returns:
            True if the agent passes the policy
        """
        result = self.evaluate(metadata, policy_name)
        return result.passed

    def verify_and_evaluate(
        self,
        agent_facts: Any,  # AgentFacts instance
        policy_name: str | None = None,
    ) -> tuple[VerificationResult, PolicyResult]:
        """
        Verify an agent's signature and evaluate against policy.

        Args:
            agent_facts: AgentFacts instance to verify
            policy_name: Policy to evaluate against

        Returns:
            Tuple of (VerificationResult, PolicyResult)
        """
        # First verify the signature
        verification = agent_facts.verify()

        # Then evaluate against policy
        policy_result = self.evaluate(agent_facts.metadata, policy_name)

        return verification, policy_result

    def _evaluate_strict(self, metadata: AgentFactsCard) -> PolicyResult:
        """Evaluate all policies and fail on any violation."""
        results = list(self.evaluate_all(metadata).values())

        if self.default_policy and self.default_policy.name not in self.policies:
            results.append(self.default_policy.evaluate(metadata))

        if not results:
            raise ValueError("No policies registered and no default policy set")

        return self._combine_results(results)

    def _combine_results(self, results: list[PolicyResult]) -> PolicyResult:
        """Combine multiple policy results into a single aggregate."""
        violations = []
        warnings = []
        passed = True

        for result in results:
            if not result.passed:
                passed = False
            violations.extend(result.violations)
            warnings.extend(result.warnings)

        return PolicyResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
        )

    def create_403_response(self, result: PolicyResult) -> dict[str, Any]:
        """
        Create a 403 response payload for rejected agents.

        Useful for middleware to return standardized rejection responses.
        """
        return {
            "error": "Agent Identity Unverified",
            "code": 403,
            "violations": [str(v) for v in result.violations],
            "message": "This endpoint requires verified agent identity. "
            "Please ensure your agent metadata is signed and meets the required policy.",
        }

    @classmethod
    def with_defaults(cls) -> "PolicyEngine":
        """
        Create an engine with default policies pre-registered.

        Includes:
        - basic_trust: Requires signature only
        - strict_enterprise: Requires signature + known provider + recent security audit
        """
        engine = cls()
        engine.register_policy(Policy.basic_trust())
        engine.register_policy(Policy.strict_enterprise())
        engine.set_default_policy(Policy.basic_trust())
        return engine

    def evaluate_ir(
        self,
        metadata: AgentFactsCard,
        policies: list["PolicyIR"],
    ) -> PolicyResult:
        """
        Evaluate agent metadata against a list of PolicyIR rules.

        PolicyIR (Intermediate Representation) rules are simple dataclasses
        that can be generated by governance adapters. This method converts
        them to concrete PolicyRule instances and evaluates them.

        Args:
            metadata: The agent metadata to evaluate
            policies: List of PolicyIR rules to evaluate

        Returns:
            PolicyResult with pass/fail status and any violations

        Example:
            ```python
            from agentfacts.plugins import RequireIssuerIR, RequireComplianceIR

            engine = PolicyEngine()
            result = engine.evaluate_ir(metadata, [
                RequireIssuerIR(did="did:key:z6Mk..."),
                RequireComplianceIR(framework="EU_AI_ACT"),
            ])
            if not result.passed:
                print(result.violations)
            ```
        """
        rules = [ir_to_rule(p) for p in policies]
        combined = Policy(name="ir_policy", rules=rules)
        return combined.evaluate(metadata)


def ir_to_rule(ir: "PolicyIR") -> "PolicyRule":
    """
    Convert a PolicyIR to a concrete PolicyRule.

    This function maps the simple PolicyIR dataclasses to the
    full-featured PolicyRule classes for evaluation.

    Args:
        ir: A PolicyIR instance

    Returns:
        Corresponding PolicyRule instance

    Raises:
        ValueError: If the IR type is not recognized
    """
    from agentfacts.plugins import (
        DenyCapabilityIR,
        RequireAttestationIR,
        RequireCapabilityIR,
        RequireComplianceIR,
        RequireIssuerIR,
        RequireStatusNotRevokedIR,
    )
    from agentfacts.policy.rules import (
        DenyCapability,
        RequireAttestation,
        RequireCapability,
        RequireCompliance,
        RequireRiskLevel,
    )

    if isinstance(ir, RequireIssuerIR):
        # Custom rule for issuer requirement
        return _RequireIssuer(issuer_did=ir.did)

    elif isinstance(ir, RequireAttestationIR):
        if not any([ir.attestation_type, ir.format, ir.issuer, ir.max_age_days]):
            return _AlwaysPass()
        return RequireAttestation(
            attestation_type=ir.attestation_type or "",
            issuer_did=ir.issuer,
            max_age_days=ir.max_age_days,
            format=ir.format,
        )

    elif isinstance(ir, RequireCapabilityIR):
        if ir.capability and ir.max_risk_level:
            # Need both capability and risk level check
            # Return a composite - for now, prioritize risk level
            return RequireRiskLevel(max_risk_level=ir.max_risk_level)
        elif ir.capability:
            return RequireCapability(required_capabilities=[ir.capability])
        elif ir.max_risk_level:
            return RequireRiskLevel(max_risk_level=ir.max_risk_level)
        else:
            # Empty IR - always passes
            return _AlwaysPass()

    elif isinstance(ir, RequireStatusNotRevokedIR):
        # Status is checked in the verification pipeline, not policy
        return _AlwaysPass()

    elif isinstance(ir, RequireComplianceIR):
        return RequireCompliance(required_frameworks=[ir.framework])

    elif isinstance(ir, DenyCapabilityIR):
        return DenyCapability(denied_capabilities=[ir.capability])

    else:
        raise ValueError(f"Unknown PolicyIR type: {type(ir).__name__}")


# Helper rules for IR conversion
@dataclass
class _RequireIssuer(PolicyRule):
    """Require attestation from a specific issuer."""

    issuer_did: str

    @property
    def name(self) -> str:
        return "require_issuer"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        matching = [a for a in metadata.attestations if a.issuer == self.issuer_did]
        if not matching:
            return PolicyResult(
                passed=False,
                violations=[
                    PolicyViolation(
                        rule=self.name,
                        message=f"No attestation from required issuer: {self.issuer_did}",
                    )
                ],
            )
        return PolicyResult(passed=True)


@dataclass
class _AlwaysPass(PolicyRule):
    """A rule that always passes (placeholder for empty IR)."""

    @property
    def name(self) -> str:
        return "always_pass"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        return PolicyResult(passed=True)
