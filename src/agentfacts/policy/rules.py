"""
Policy rules for Zero Trust verification.

Provides building blocks for defining access policies
that agents must satisfy before being trusted.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from agentfacts.models import AgentFactsCard, ModelProvider
from agentfacts.utils import utcnow as _utcnow


@dataclass
class PolicyViolation:
    """A single policy violation."""

    rule: str
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule}: {self.message}"


@dataclass
class PolicyResult:
    """Result of evaluating a policy against agent metadata."""

    passed: bool
    violations: list[PolicyViolation] = field(default_factory=list)
    warnings: list[PolicyViolation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    @property
    def messages(self) -> list[str]:
        """Get all violation messages."""
        return [str(v) for v in self.violations + self.warnings]


class PolicyRule(ABC):
    """
    Abstract base class for policy rules.

    Each rule evaluates a specific aspect of an agent's metadata
    and returns whether the agent passes the check.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name for error messages."""
        pass

    @abstractmethod
    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        """
        Evaluate this rule against agent metadata.

        Args:
            metadata: The agent metadata to evaluate

        Returns:
            PolicyResult indicating pass/fail and any violations
        """
        pass


@dataclass
class RequireSignature(PolicyRule):
    """Require that the agent metadata is signed."""

    @property
    def name(self) -> str:
        return "require_signature"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        if (
            not metadata.signature
            or not metadata.signature.value
            or not metadata.log_proof
            or not metadata.log_proof.root_hash
        ):
            return PolicyResult(
                passed=False,
                violations=[
                    PolicyViolation(
                        rule=self.name,
                        message="Agent card is missing signature or log proof",
                    )
                ],
            )
        return PolicyResult(passed=True)


@dataclass
class RequireProvider(PolicyRule):
    """Require specific model provider(s)."""

    allowed_providers: list[ModelProvider]

    @property
    def name(self) -> str:
        return "require_provider"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        provider = metadata.agent.model.provider
        if provider not in self.allowed_providers:
            allowed = ", ".join(p.value for p in self.allowed_providers)
            return PolicyResult(
                passed=False,
                violations=[
                    PolicyViolation(
                        rule=self.name,
                        message=f"Provider '{provider.value}' not in allowed list: [{allowed}]",
                    )
                ],
            )
        return PolicyResult(passed=True)


@dataclass
class RequireModel(PolicyRule):
    """Require specific model name(s)."""

    allowed_models: list[str]
    allow_prefix_match: bool = True

    @property
    def name(self) -> str:
        return "require_model"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        model_name = metadata.agent.model.name.lower()

        for allowed in self.allowed_models:
            allowed_lower = allowed.lower()
            if self.allow_prefix_match:
                if model_name.startswith(allowed_lower) or allowed_lower.startswith(
                    model_name
                ):
                    return PolicyResult(passed=True)
            else:
                if model_name == allowed_lower:
                    return PolicyResult(passed=True)

        return PolicyResult(
            passed=False,
            violations=[
                PolicyViolation(
                    rule=self.name,
                    message=(
                        f"Model '{metadata.agent.model.name}' not in allowed list: "
                        f"{self.allowed_models}"
                    ),
                )
            ],
        )


@dataclass
class RequireAttestation(PolicyRule):
    """Require attestation of a specific type, optionally from a specific issuer."""

    attestation_type: str
    issuer_did: str | None = None
    max_age_days: int | None = None
    format: str | None = None

    @property
    def name(self) -> str:
        return "require_attestation"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        matching = list(metadata.attestations)

        if self.attestation_type:
            matching = [a for a in matching if a.type == self.attestation_type]
            if not matching:
                return PolicyResult(
                    passed=False,
                    violations=[
                        PolicyViolation(
                            rule=self.name,
                            message=f"No attestation of type '{self.attestation_type}' found",
                        )
                    ],
                )

        if self.format:
            matching = [a for a in matching if a.format == self.format]
            if not matching:
                type_hint = (
                    f" of type '{self.attestation_type}'"
                    if self.attestation_type
                    else ""
                )
                return PolicyResult(
                    passed=False,
                    violations=[
                        PolicyViolation(
                            rule=self.name,
                            message=f"No attestation{type_hint} with format '{self.format}' found",
                        )
                    ],
                )

        # Filter by issuer if specified
        if self.issuer_did:
            matching = [a for a in matching if a.issuer == self.issuer_did]
            if not matching:
                type_hint = (
                    f" of type '{self.attestation_type}'"
                    if self.attestation_type
                    else ""
                )
                return PolicyResult(
                    passed=False,
                    violations=[
                        PolicyViolation(
                            rule=self.name,
                            message=f"No attestation{type_hint} from issuer '{self.issuer_did}'",
                        )
                    ],
                )

        # Check age if specified
        if self.max_age_days:
            cutoff = _utcnow() - timedelta(days=self.max_age_days)
            recent = [a for a in matching if a.issued_at >= cutoff]
            if not recent:
                type_hint = (
                    f" of type '{self.attestation_type}'"
                    if self.attestation_type
                    else ""
                )
                return PolicyResult(
                    passed=False,
                    violations=[
                        PolicyViolation(
                            rule=self.name,
                            message=(
                                f"Attestation{type_hint} is older than "
                                f"{self.max_age_days} days"
                            ),
                        )
                    ],
                )

        return PolicyResult(passed=True)


@dataclass
class RequireCapability(PolicyRule):
    """Require that the agent has specific capability(ies)."""

    required_capabilities: list[str]

    @property
    def name(self) -> str:
        return "require_capability"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        agent_caps = {c.name.lower() for c in metadata.agent.capabilities}
        missing = []

        for required in self.required_capabilities:
            if required.lower() not in agent_caps:
                missing.append(required)

        if missing:
            return PolicyResult(
                passed=False,
                violations=[
                    PolicyViolation(
                        rule=self.name,
                        message=f"Missing required capabilities: {missing}",
                    )
                ],
            )

        return PolicyResult(passed=True)


@dataclass
class DenyCapability(PolicyRule):
    """Deny agents with specific capability(ies)."""

    denied_capabilities: list[str]

    @property
    def name(self) -> str:
        return "deny_capability"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        agent_caps = {c.name.lower() for c in metadata.agent.capabilities}
        violations = []

        for denied in self.denied_capabilities:
            if denied.lower() in agent_caps:
                violations.append(
                    PolicyViolation(
                        rule=self.name,
                        message=f"Agent has denied capability: '{denied}'",
                    )
                )

        if violations:
            return PolicyResult(passed=False, violations=violations)

        return PolicyResult(passed=True)


@dataclass
class RequireCompliance(PolicyRule):
    """Require compliance with specific framework(s)."""

    required_frameworks: list[str]

    @property
    def name(self) -> str:
        return "require_compliance"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        agent_frameworks = {f.lower() for f in metadata.policy.compliance.frameworks}
        missing = []

        for required in self.required_frameworks:
            if required.lower() not in agent_frameworks:
                missing.append(required)

        if missing:
            return PolicyResult(
                passed=False,
                violations=[
                    PolicyViolation(
                        rule=self.name,
                        message=f"Missing required compliance frameworks: {missing}",
                    )
                ],
            )

        return PolicyResult(passed=True)


@dataclass
class RequireRiskLevel(PolicyRule):
    """Require that all capabilities are at or below a maximum risk level."""

    max_risk_level: str  # low, medium, high

    RISK_LEVELS = {"low": 0, "medium": 1, "high": 2}

    @property
    def name(self) -> str:
        return "require_risk_level"

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        max_level = self.RISK_LEVELS.get(self.max_risk_level.lower(), 2)
        violations = []

        for cap in metadata.agent.capabilities:
            if cap.risk_level:
                cap_level = self.RISK_LEVELS.get(cap.risk_level.lower(), 2)
                if cap_level > max_level:
                    violations.append(
                        PolicyViolation(
                            rule=self.name,
                            message=(
                                f"Capability '{cap.name}' has risk level "
                                f"'{cap.risk_level}' (max: {self.max_risk_level})"
                            ),
                        )
                    )

        if violations:
            return PolicyResult(passed=False, violations=violations)

        return PolicyResult(passed=True)


@dataclass
class Policy:
    """
    A collection of rules that an agent must satisfy.

    Policies can be combined with AND/OR logic and support
    strict mode for rejecting agents with any violations.
    """

    name: str
    rules: list[PolicyRule] = field(default_factory=list)
    require_all: bool = True  # AND vs OR logic
    strict: bool = False  # Treat warnings as errors

    def add_rule(self, rule: PolicyRule) -> "Policy":
        """Add a rule to this policy."""
        self.rules.append(rule)
        return self

    def evaluate(self, metadata: AgentFactsCard) -> PolicyResult:
        """
        Evaluate all rules against the AgentFacts Card.

        Args:
            metadata: AgentFacts Card to evaluate

        Returns:
            Combined PolicyResult
        """
        all_violations: list[PolicyViolation] = []
        all_warnings: list[PolicyViolation] = []
        passed_count = 0

        for rule in self.rules:
            result = rule.evaluate(metadata)
            if result.passed:
                passed_count += 1
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)

        passed = (
            passed_count == len(self.rules) if self.require_all else passed_count > 0
        )

        if self.strict and all_warnings:
            # Promote warnings to violations in strict mode
            all_violations.extend(all_warnings)
            all_warnings = []
            passed = False

        return PolicyResult(
            passed=passed,
            violations=all_violations,
            warnings=all_warnings,
        )

    @classmethod
    def strict_enterprise(cls) -> "Policy":
        """
        Create a strict enterprise policy.

        Requires:
        - Signed metadata
        - Known provider (OpenAI or Anthropic)
        - Security attestation within 90 days
        """
        return cls(
            name="strict_enterprise",
            rules=[
                RequireSignature(),
                RequireProvider([ModelProvider.OPENAI, ModelProvider.ANTHROPIC]),
                RequireAttestation("security_audit", max_age_days=90),
            ],
            strict=True,
        )

    @classmethod
    def basic_trust(cls) -> "Policy":
        """
        Create a basic trust policy.

        Requires only signed metadata.
        """
        return cls(
            name="basic_trust",
            rules=[RequireSignature()],
        )


class PolicyBuilder:
    """
    Fluent builder for creating policies.

    Example:
        policy = (PolicyBuilder("my_policy")
            .require_signature()
            .require_provider([ModelProvider.OPENAI, ModelProvider.ANTHROPIC])
            .require_capability(["web_search"])
            .deny_capability(["shell", "code_executor"])
            .require_attestation("security_audit", max_age_days=90)
            .max_risk_level("medium")
            .strict()
            .build())
    """

    def __init__(self, name: str):
        """
        Initialize the builder with a policy name.

        Args:
            name: Human-readable name for the policy
        """
        self._name = name
        self._rules: list[PolicyRule] = []
        self._require_all = True
        self._strict = False

    def require_signature(self) -> "PolicyBuilder":
        """Add a rule requiring signed metadata."""
        self._rules.append(RequireSignature())
        return self

    def require_provider(self, providers: list[ModelProvider]) -> "PolicyBuilder":
        """Add a rule requiring specific model providers."""
        self._rules.append(RequireProvider(allowed_providers=providers))
        return self

    def require_model(
        self, models: list[str], allow_prefix_match: bool = True
    ) -> "PolicyBuilder":
        """Add a rule requiring specific model names."""
        self._rules.append(
            RequireModel(
                allowed_models=models,
                allow_prefix_match=allow_prefix_match,
            )
        )
        return self

    def require_attestation(
        self,
        attestation_type: str,
        issuer_did: str | None = None,
        max_age_days: int | None = None,
        format: str | None = None,
    ) -> "PolicyBuilder":
        """Add a rule requiring a specific attestation type."""
        self._rules.append(
            RequireAttestation(
                attestation_type=attestation_type,
                issuer_did=issuer_did,
                max_age_days=max_age_days,
                format=format,
            )
        )
        return self

    def require_capability(self, capabilities: list[str]) -> "PolicyBuilder":
        """Add a rule requiring specific capabilities."""
        self._rules.append(RequireCapability(required_capabilities=capabilities))
        return self

    def deny_capability(self, capabilities: list[str]) -> "PolicyBuilder":
        """Add a rule denying specific capabilities."""
        self._rules.append(DenyCapability(denied_capabilities=capabilities))
        return self

    def require_compliance(self, frameworks: list[str]) -> "PolicyBuilder":
        """Add a rule requiring compliance with specific frameworks."""
        self._rules.append(RequireCompliance(required_frameworks=frameworks))
        return self

    def max_risk_level(self, level: str) -> "PolicyBuilder":
        """Add a rule limiting maximum risk level (low, medium, high)."""
        self._rules.append(RequireRiskLevel(max_risk_level=level))
        return self

    def add_rule(self, rule: PolicyRule) -> "PolicyBuilder":
        """Add a custom rule."""
        self._rules.append(rule)
        return self

    def any_of(self) -> "PolicyBuilder":
        """Set policy to pass if ANY rule passes (OR logic)."""
        self._require_all = False
        return self

    def all_of(self) -> "PolicyBuilder":
        """Set policy to pass only if ALL rules pass (AND logic). This is the default."""
        self._require_all = True
        return self

    def strict(self, enabled: bool = True) -> "PolicyBuilder":
        """Enable strict mode (treat warnings as errors)."""
        self._strict = enabled
        return self

    def build(self) -> Policy:
        """Build and return the Policy."""
        return Policy(
            name=self._name,
            rules=self._rules,
            require_all=self._require_all,
            strict=self._strict,
        )
