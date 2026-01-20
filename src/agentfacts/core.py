"""
Core AgentFacts class - the main entry point for the SDK.

Provides factory methods for creating AgentFacts Cards from agent
frameworks and methods for signing, verification, and evidence logging.

Pythonic Features:
- Fluent API with Self return types for method chaining
- __len__ returns capability count
- __iter__ iterates over capabilities
- __contains__ checks for capability by name
- __getitem__ accesses capabilities by index or name
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from agentfacts.crypto.canonicalization import canonicalize_json, compute_hash
from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair
from agentfacts.logging import warn_default
from agentfacts.merkle.log import TransparencyLog
from agentfacts.merkle.tree import MerkleProof, MerkleTree
from agentfacts.models import (
    AgentFactsCard,
    AgentInfo,
    AgentRole,
    Attestation,
    BaselineModel,
    Capability,
    ComplianceInfo,
    DelegationPolicy,
    HandshakeChallenge,
    HandshakeResponse,
    LogProof,
    LogProofEntry,
    ModelProvider,
    OperationalConstraints,
    Policy,
    Publisher,
    PublisherKey,
    SignatureBlock,
    Tool,
    VerificationResult,
)
from agentfacts.utils import utcnow as _utcnow

if TYPE_CHECKING:
    from agentfacts.plugins import VerificationContext


class AgentFacts:
    """
    The main AgentFacts class - creates and manages verifiable agent identity.

    This is the AgentFacts Card (v0.1) for an AI agent, providing
    cryptographically verifiable metadata about its capabilities,
    model, publisher, and policy status.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        baseline_model: BaselineModel | None = None,
        capabilities: list[Capability] | None = None,
        tools: list[Tool] | None = None,
        constraints: OperationalConstraints | None = None,
        compliance: ComplianceInfo | None = None,
        policy: Policy | None = None,
        publisher: Publisher | None = None,
        key_pair: KeyPair | None = None,
        did: DID | None = None,
        version: str = "1.0.0",
        framework: str | None = None,
        role: AgentRole | None = None,
        delegation: DelegationPolicy | None = None,
        group_memberships: list[str] | None = None,
        context: dict[str, Any] | None = None,
        log_id: str = "local",
    ) -> None:
        """
        Initialize AgentFacts.

        Args:
            name: Human-readable agent name
            description: Agent description
            baseline_model: The underlying LLM information
            capabilities: List of agent capabilities/tools
            tools: List of tool objects
            constraints: Operational constraints
            compliance: Compliance information
            policy: Full policy object (overrides constraints/compliance)
            publisher: Publisher identity (defaults to agent DID)
            key_pair: Ed25519 key pair for signing (generated if not provided)
            did: Decentralised Identifier (derived from key_pair if not provided)
            version: Agent version string
            framework: Framework name (langchain, crewai, autogen)
            role: Role metadata for multi-agent systems
            delegation: Delegation policy for multi-agent systems
            group_memberships: Group/crew DIDs for this agent
            context: Additional metadata
            log_id: Transparency log identifier for log_proof
        """
        key_pair_value = key_pair or KeyPair.generate()
        self._key_pair: KeyPair | None = key_pair_value
        self._did: DID | None = did or DID.from_key_pair(key_pair_value)
        self._log_id = log_id

        agent_id = self._did.uri
        model = baseline_model or BaselineModel(
            name="unknown", provider=ModelProvider.UNKNOWN
        )
        caps = capabilities or []
        tool_list = tools or []
        policy_obj = policy or Policy(
            compliance=compliance or ComplianceInfo(),
            constraints=constraints or OperationalConstraints(),
        )
        publisher_obj = publisher or Publisher(id=agent_id)

        signature_placeholder = SignatureBlock(
            alg="ed25519",
            key_id=f"{publisher_obj.id}#sig-1",
            value="",
        )
        log_proof_placeholder = LogProof(
            log_id=self._log_id,
            leaf_hash="",
            root_hash="",
            inclusion=[],
        )

        self._card = AgentFactsCard(
            agent=AgentInfo(
                id=agent_id,
                name=name,
                description=description,
                version=version,
                model=model,
                capabilities=caps,
                tools=tool_list,
                framework=framework,
                role=role,
                delegation=delegation or DelegationPolicy(),
                group_memberships=group_memberships or [],
                context=context or {},
            ),
            publisher=publisher_obj,
            policy=policy_obj,
            signature=signature_placeholder,
            log_proof=log_proof_placeholder,
        )

        self._evidence_log = TransparencyLog(agent_did=agent_id)
        self._card_log = MerkleTree()
        self._signed = False

    @classmethod
    def from_langchain(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """
        Create AgentFacts from a LangChain agent or chain.

        This is the primary factory method that performs automatic introspection
        of the LangChain object to extract metadata.
        """
        try:
            from agentfacts.integrations.langchain.introspector import introspect_any
        except ImportError as err:
            raise ImportError(
                "LangChain introspection requires langchain. "
                "Install with: pip install langchain langchain-core"
            ) from err

        baseline_model, capabilities, constraints = introspect_any(agent)

        instance = cls(
            name=name,
            description=description,
            baseline_model=baseline_model,
            capabilities=capabilities,
            constraints=constraints,
            key_pair=key_pair,
            version=version,
            framework="langchain",
            **kwargs,
        )
        return instance

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        framework: str | None = None,
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """
        Create AgentFacts from any supported agent framework.

        This is the universal factory method that auto-detects the framework
        and performs appropriate introspection.
        """
        from agentfacts.integrations import get_registry

        registry = get_registry()
        result = registry.introspect(agent, framework=framework)

        if result.baseline_model is None:
            warn_default(
                "baseline_model",
                "unknown",
                f"No model found for {result.framework} agent '{name}'",
            )
        if not result.capabilities:
            warn_default(
                "capabilities",
                "[]",
                f"No tools/capabilities found for {result.framework} agent '{name}'",
            )

        instance = cls(
            name=name,
            description=description,
            baseline_model=result.baseline_model,
            capabilities=result.capabilities,
            constraints=result.constraints,
            key_pair=key_pair,
            version=version,
            framework=result.framework,
            **kwargs,
        )

        instance._card.agent.context.update(result.context)

        if "role" in result.context:
            instance._card.agent.role = AgentRole.model_validate(result.context["role"])
        if "delegation" in result.context:
            instance._card.agent.delegation = DelegationPolicy.model_validate(
                result.context["delegation"]
            )

        return instance

    @classmethod
    def from_agent_signed(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        framework: str | None = None,
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """Create and immediately sign AgentFacts from any agent."""
        facts = cls.from_agent(
            agent, name, description, framework, key_pair, version, **kwargs
        )
        facts.sign()
        return facts

    @classmethod
    def from_langchain_signed(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """Create and immediately sign AgentFacts from a LangChain agent."""
        facts = cls.from_langchain(
            agent, name, description, key_pair, version, **kwargs
        )
        facts.sign()
        return facts

    @classmethod
    def from_crewai_signed(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """Create and immediately sign AgentFacts from a CrewAI agent."""
        return cls.from_agent_signed(
            agent,
            name,
            description,
            framework="crewai",
            key_pair=key_pair,
            version=version,
            **kwargs,
        )

    @classmethod
    def from_autogen_signed(
        cls,
        agent: Any,
        name: str,
        description: str = "",
        key_pair: KeyPair | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> AgentFacts:
        """Create and immediately sign AgentFacts from an AutoGen agent."""
        return cls.from_agent_signed(
            agent,
            name,
            description,
            framework="autogen",
            key_pair=key_pair,
            version=version,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], key_pair: KeyPair | None = None
    ) -> AgentFacts:
        """
        Create AgentFacts from a dictionary (e.g., loaded from JSON).

        Args:
            data: Dictionary containing an AgentFacts Card
            key_pair: Optional key pair for signing new data
        """
        card = AgentFactsCard.model_validate(data)

        if key_pair is None:
            key_pair = cls._key_pair_from_card(card)

        facts = cls.__new__(cls)
        facts._key_pair = key_pair
        facts._did = DID.parse(card.agent.id) if card.agent.id else None
        if facts._did is None and key_pair is not None:
            facts._did = DID.from_key_pair(key_pair)
        facts._log_id = card.log_proof.log_id if card.log_proof else "local"
        facts._evidence_log = TransparencyLog(agent_did=card.agent.id)
        facts._card_log = MerkleTree()
        facts._card = card
        facts._signed = bool(
            card.signature.value
            and card.log_proof.root_hash
            and card.log_proof.leaf_hash
        )

        return facts

    @classmethod
    def from_json(cls, json_str: str, key_pair: KeyPair | None = None) -> AgentFacts:
        """Load AgentFacts from a JSON string."""
        import json

        data = json.loads(json_str)
        return cls.from_dict(data, key_pair)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def did(self) -> str:
        """Get the agent's DID."""
        return self._card.agent.id

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._card.agent.name

    @property
    def metadata(self) -> AgentFactsCard:
        """Get the full AgentFacts Card."""
        return self._card

    @property
    def card(self) -> AgentFactsCard:
        """Alias for metadata."""
        return self._card

    @property
    def key_pair(self) -> KeyPair:
        """Get the key pair."""
        if self._key_pair is None:
            raise ValueError("No key pair available for this AgentFacts instance")
        return self._key_pair

    @property
    def public_key(self) -> str:
        """Get the public key (base64)."""
        if self._key_pair is not None:
            return self._key_pair.public_key_base64
        key = self._select_publisher_key()
        if key is None:
            raise ValueError("No public key available for this AgentFacts instance")
        return key.public_key

    def _has_signature(self) -> bool:
        return bool(
            self._card.signature.value
            and self._card.log_proof.root_hash
            and self._card.log_proof.leaf_hash
        )

    @property
    def is_signed(self) -> bool:
        """Check if the card is signed and has a log proof."""
        return self._signed and self._has_signature()

    @property
    def merkle_root(self) -> str | None:
        """Get the current Merkle root of the evidence log."""
        return self._evidence_log.root

    @property
    def evidence_log(self) -> TransparencyLog:
        """Get the evidence transparency log."""
        return self._evidence_log

    @property
    def signature(self) -> str | None:
        """Get the signature value (base64-encoded) if signed."""
        if self._card.signature and self._card.signature.value:
            return self._card.signature.value
        return None

    # -------------------------------------------------------------------------
    # Signing & Verification
    # -------------------------------------------------------------------------

    def sign(
        self,
        key_pair: KeyPair | None = None,
        key_id: str | None = None,
        algorithm: str = "ed25519",
        log_id: str | None = None,
    ) -> AgentFacts:
        """
        Sign the AgentFacts Card.

        Args:
            key_pair: Optional key pair to use (uses instance key pair if not provided)
            key_id: Optional key identifier (defaults to publisher.id#sig-1)
            algorithm: Signature algorithm label
            log_id: Optional log identifier override
        """
        kp = key_pair or self.key_pair

        if not kp.can_sign():
            raise ValueError("Cannot sign: key pair is verification-only")

        if log_id:
            self._log_id = log_id

        self._card.issued_at = _utcnow()

        resolved_key_id = key_id or f"{self._card.publisher.id}#sig-1"

        self._ensure_publisher_key(kp, resolved_key_id)

        signable = self._get_signable_data()
        canonical = canonicalize_json(signable)
        signature_value = kp.sign_base64(canonical)

        self._card.signature = SignatureBlock(
            alg=algorithm,
            key_id=resolved_key_id,
            value=signature_value,
        )

        leaf_hash_bytes = self._compute_leaf_hash_bytes()
        index = self._card_log.append_hash(leaf_hash_bytes)
        proof = self._card_log.get_proof(index)

        self._card.log_proof = LogProof(
            log_id=self._log_id,
            leaf_hash=leaf_hash_bytes.hex(),
            root_hash=proof.root,
            inclusion=self._format_inclusion(proof),
        )

        self._signed = True

        return self

    def verify(
        self,
        key_pair: KeyPair | None = None,
        context: VerificationContext | None = None,
    ) -> VerificationResult:
        """
        Verify the AgentFacts Card signature and log proof.

        When a VerificationContext is provided, the verification pipeline
        is enhanced to support:
        - External DID resolution (did:web, did:ion, etc.)
        - Attestation signature verification (SD-JWT-VC, etc.)
        - Revocation/status checking
        - Governance policy evaluation
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self._has_signature():
            errors.append("Card is not signed")
            return VerificationResult(valid=False, did=self.did, errors=errors)

        verify_key = self._resolve_verification_key(key_pair, context, errors)
        if verify_key is None:
            return VerificationResult(valid=False, did=self.did, errors=errors)

        signable = self._get_signable_data()
        canonical = canonicalize_json(signable)

        if not verify_key.verify_base64(canonical, self._card.signature.value):
            errors.append("Signature verification failed")
            return VerificationResult(valid=False, did=self.did, errors=errors)

        publisher_did = self._card.publisher.id
        if publisher_did:
            try:
                expected_did = DID.from_public_key(verify_key.public_key_bytes)
                if expected_did.uri != publisher_did:
                    message = (
                        "Publisher DID mismatch: "
                        f"expected {expected_did.uri}, got {publisher_did}"
                    )
                    if context and context.strict_publisher_did_match:
                        errors.append(message)
                    else:
                        warnings.append(message)
            except Exception:
                pass

        if not self._verify_log_proof(errors):
            return VerificationResult(valid=False, did=self.did, errors=errors)

        if context:
            log_errors, log_warnings = self._verify_log_checkpoint(context)
            errors.extend(log_errors)
            warnings.extend(log_warnings)

        if context and context.verify_attestation_signatures:
            attestation_errors, attestation_warnings = self._verify_attestations(
                context
            )
            errors.extend(attestation_errors)
            warnings.extend(attestation_warnings)

        if context and context.check_revocation_status:
            status_errors, status_warnings = self._check_attestation_status(context)
            errors.extend(status_errors)
            warnings.extend(status_warnings)

        policy_violations: list[str] = []
        if context:
            policy_ir = context.get_policy_ir()
            if policy_ir:
                policy_violations = self._evaluate_policy_ir(policy_ir)

        valid = len(errors) == 0

        return VerificationResult(
            valid=valid,
            did=self.did,
            errors=errors,
            warnings=warnings,
            policy_violations=policy_violations,
        )

    def _verify_log_proof(self, errors: list[str]) -> bool:
        if not self._card.log_proof:
            errors.append("Missing log_proof")
            return False

        leaf_hash_bytes = self._compute_leaf_hash_bytes()
        leaf_hash_hex = leaf_hash_bytes.hex()
        if leaf_hash_hex != self._card.log_proof.leaf_hash:
            errors.append("log_proof leaf_hash mismatch")
            return False

        try:
            siblings = self._parse_inclusion(self._card.log_proof.inclusion)
        except ValueError as exc:
            errors.append(str(exc))
            return False

        proof = MerkleProof(
            leaf_index=0,
            leaf_hash=self._card.log_proof.leaf_hash,
            siblings=siblings,
            root=self._card.log_proof.root_hash,
        )

        if not MerkleTree.verify_proof(proof):
            errors.append("log_proof verification failed")
            return False

        return True

    def _verify_log_checkpoint(
        self,
        context: VerificationContext,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        if not self._card.log_proof:
            return errors, warnings

        provider = context.log_root_provider
        log_id = self._card.log_proof.log_id

        if provider:
            try:
                expected_root = provider(log_id)
            except Exception as exc:
                message = f"Log root provider error for log_id '{log_id}': {exc}"
                if context.strict_log_checkpoint:
                    errors.append(message)
                else:
                    warnings.append(message)
                return errors, warnings

            if expected_root:
                expected = str(expected_root).lower()
                actual = self._card.log_proof.root_hash.lower()
                if expected != actual:
                    errors.append(f"Log checkpoint root mismatch for log_id '{log_id}'")
            else:
                message = f"No log checkpoint available for log_id '{log_id}'"
                if context.strict_log_checkpoint:
                    errors.append(message)
                else:
                    warnings.append(message)
        elif context.strict_log_checkpoint:
            errors.append(
                "Log checkpoint verification required but no log_root_provider configured"
            )

        return errors, warnings

    def _resolve_verification_key(
        self,
        key_pair: KeyPair | None,
        context: VerificationContext | None,
        errors: list[str],
    ) -> KeyPair | None:
        if key_pair:
            return key_pair

        signature = self._card.signature
        publisher = self._card.publisher

        if signature:
            matching_key = self._find_publisher_key(signature.key_id)
            if matching_key:
                try:
                    resolved = KeyPair.from_public_key_base64(matching_key.public_key)
                except Exception as e:
                    errors.append(f"Invalid publisher key: {e}")
                    return None

                if context and context.did_resolver:
                    resolver = context.did_resolver
                    if resolver.supports(publisher.id):
                        try:
                            resolved_did = resolver.resolve(publisher.id)
                            if (
                                resolved_did.public_key_base64
                                != matching_key.public_key
                            ):
                                errors.append(
                                    "Publisher key mismatch with DID resolution"
                                )
                                return None
                        except Exception as e:
                            if context.strict_did_verification:
                                errors.append(f"DID resolution failed: {e}")
                                return None
                return resolved

        if context and context.did_resolver:
            resolver = context.did_resolver
            if resolver.supports(publisher.id):
                try:
                    resolved_did = resolver.resolve(publisher.id)
                    return KeyPair.from_public_key_base64(
                        resolved_did.public_key_base64
                    )
                except Exception as e:
                    if context.strict_did_verification:
                        errors.append(f"DID resolution failed: {e}")
                        return None

        if self._key_pair is not None:
            return self._key_pair

        if publisher.id:
            try:
                did = DID.parse(publisher.id)
                if did.method == "key":
                    return did.to_key_pair()
            except Exception as e:
                errors.append(f"DID key resolution failed: {e}")
                return None

        errors.append("No public key available for verification")
        return None

    def _find_publisher_key(self, key_id: str) -> PublisherKey | None:
        for key in self._card.publisher.keys:
            if key.id == key_id:
                return key
        return None

    def _select_publisher_key(self) -> PublisherKey | None:
        signature = self._card.signature
        if signature:
            match = self._find_publisher_key(signature.key_id)
            if match:
                return match
        if self._card.publisher.keys:
            return self._card.publisher.keys[0]
        return None

    def _ensure_publisher_key(self, key_pair: KeyPair, key_id: str) -> None:
        existing = self._find_publisher_key(key_id)
        if existing:
            if existing.public_key != key_pair.public_key_base64:
                existing.public_key = key_pair.public_key_base64
                existing.controller = self._card.publisher.id
                existing.type = "Ed25519VerificationKey2020"
                existing.purpose = "assertionMethod"
            return
        self._card.publisher.keys.append(
            PublisherKey(
                id=key_id,
                type="Ed25519VerificationKey2020",
                public_key=key_pair.public_key_base64,
                controller=self._card.publisher.id,
                purpose="assertionMethod",
            )
        )

    def _empty_signature(self) -> SignatureBlock:
        return SignatureBlock(
            alg="ed25519",
            key_id=f"{self._card.publisher.id}#sig-1",
            value="",
        )

    def _empty_log_proof(self) -> LogProof:
        return LogProof(
            log_id=self._log_id,
            leaf_hash="",
            root_hash="",
            inclusion=[],
        )

    def _verify_attestations(
        self,
        context: VerificationContext,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        for attestation in self._card.attestations:
            if not attestation.format:
                continue

            verifier = context.get_verifier_for_format(attestation.format)
            if verifier is None:
                warnings.append(
                    f"No verifier for attestation format '{attestation.format}' "
                    f"(attestation {attestation.id})"
                )
                continue

            try:
                result = verifier.verify(attestation, context)
                if not result.valid:
                    errors.extend(
                        f"Attestation {attestation.id}: {e}" for e in result.errors
                    )
                warnings.extend(
                    f"Attestation {attestation.id}: {w}" for w in result.warnings
                )
            except Exception as e:
                errors.append(f"Attestation {attestation.id} verification error: {e}")

        return errors, warnings

    def _check_attestation_status(
        self,
        context: VerificationContext,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        for attestation in self._card.attestations:
            if not attestation.status_ref:
                continue

            checker = context.get_status_checker(attestation.status_ref)
            if checker is None:
                warnings.append(
                    f"No status checker for '{attestation.status_ref}' "
                    f"(attestation {attestation.id})"
                )
                continue

            try:
                result = checker.check(attestation.status_ref, context)
                if not result.valid:
                    errors.append(
                        f"Attestation {attestation.id} status check failed: {result.status}"
                    )
                    errors.extend(result.errors)
            except Exception as e:
                warnings.append(f"Attestation {attestation.id} status check error: {e}")

        return errors, warnings

    def _evaluate_policy_ir(self, policy_ir: list[Any]) -> list[str]:
        from agentfacts.plugins import (
            DenyCapabilityIR,
            RequireAttestationIR,
            RequireCapabilityIR,
            RequireComplianceIR,
            RequireIssuerIR,
            RequireStatusNotRevokedIR,
        )

        violations: list[str] = []

        for rule in policy_ir:
            if isinstance(rule, RequireIssuerIR):
                if not any(a.issuer == rule.did for a in self._card.attestations):
                    violations.append(
                        f"No attestation from required issuer: {rule.did}"
                    )

            elif isinstance(rule, RequireAttestationIR):
                matching = self._card.attestations
                if rule.attestation_type:
                    matching = [a for a in matching if a.type == rule.attestation_type]
                if rule.format:
                    matching = [a for a in matching if a.format == rule.format]
                if rule.issuer:
                    matching = [a for a in matching if a.issuer == rule.issuer]
                if not matching:
                    violations.append(
                        f"Missing required attestation: type={rule.attestation_type}, "
                        f"format={rule.format}, issuer={rule.issuer}"
                    )

            elif isinstance(rule, RequireCapabilityIR):
                if rule.capability:
                    caps = [c.name.lower() for c in self._card.agent.capabilities]
                    if rule.capability.lower() not in caps:
                        violations.append(
                            f"Missing required capability: {rule.capability}"
                        )
                if rule.max_risk_level:
                    risk_levels = {"low": 0, "medium": 1, "high": 2}
                    max_level = risk_levels.get(rule.max_risk_level.lower(), 2)
                    for cap in self._card.agent.capabilities:
                        if cap.risk_level:
                            cap_level = risk_levels.get(cap.risk_level.lower(), 2)
                            if cap_level > max_level:
                                violations.append(
                                    f"Capability '{cap.name}' exceeds max risk level "
                                    f"({cap.risk_level} > {rule.max_risk_level})"
                                )

            elif isinstance(rule, RequireStatusNotRevokedIR):
                for attestation in self._card.attestations:
                    if attestation.status_ref:
                        pass

            elif isinstance(rule, RequireComplianceIR):
                frameworks = [
                    f.lower() for f in self._card.policy.compliance.frameworks
                ]
                if rule.framework.lower() not in frameworks:
                    violations.append(
                        f"Missing required compliance framework: {rule.framework}"
                    )

            elif isinstance(rule, DenyCapabilityIR):
                caps = [c.name.lower() for c in self._card.agent.capabilities]
                if rule.capability.lower() in caps:
                    violations.append(f"Agent has denied capability: {rule.capability}")

        return violations

    def _get_signable_data(self) -> dict[str, Any]:
        data = self._card.model_dump(mode="json", exclude_none=True)
        data.pop("signature", None)
        data.pop("log_proof", None)
        return data

    def _get_leaf_data(self) -> dict[str, Any]:
        data = self._card.model_dump(mode="json", exclude_none=True)
        data.pop("log_proof", None)
        return data

    def _compute_leaf_hash_bytes(self) -> bytes:
        import hashlib

        canonical = canonicalize_json(self._get_leaf_data())
        return hashlib.sha256(canonical).digest()

    def _format_inclusion(self, proof: MerkleProof) -> list[LogProofEntry]:
        return [
            LogProofEntry(
                hash=hash_hex,
                position=cast(Literal["left", "right"], position),
            )
            for hash_hex, position in proof.siblings
        ]

    def _parse_inclusion(self, inclusion: list[LogProofEntry]) -> list[tuple[str, str]]:
        siblings: list[tuple[str, str]] = []
        for entry in inclusion:
            position = entry.position
            hash_hex = entry.hash
            if position not in ("left", "right"):
                raise ValueError("Invalid inclusion entry position")
            try:
                sibling_bytes = bytes.fromhex(hash_hex)
            except ValueError as exc:
                raise ValueError("Invalid inclusion entry hash") from exc
            if len(sibling_bytes) != 32:
                raise ValueError("Invalid inclusion entry hash length")
            siblings.append((hash_hex, position))
        return siblings

    @staticmethod
    def _key_pair_from_card(card: AgentFactsCard) -> KeyPair | None:
        if card.signature and card.publisher.keys:
            for key in card.publisher.keys:
                if key.id == card.signature.key_id:
                    return KeyPair.from_public_key_base64(key.public_key)
            return KeyPair.from_public_key_base64(card.publisher.keys[0].public_key)

        if card.publisher.id:
            try:
                did = DID.parse(card.publisher.id)
                if did.method == "key":
                    return did.to_key_pair()
            except Exception:
                return None
        return None

    # -------------------------------------------------------------------------
    # Evidence Logging
    # -------------------------------------------------------------------------

    def log_evidence(self, evidence_type: str, data: dict[str, Any]) -> None:
        """
        Log evidence to the transparency log.

        Evidence logs are separate from the card log used for log_proof.
        """
        self._evidence_log.log_evidence(evidence_type, data, self.key_pair)

    def add_attestation(self, attestation: Attestation) -> None:
        """Add an attestation to the card and invalidate the signature."""
        self._card.attestations.append(attestation)
        self._card.signature = self._empty_signature()
        self._card.log_proof = self._empty_log_proof()
        self._signed = False

    # -------------------------------------------------------------------------
    # Handshake Protocol
    # -------------------------------------------------------------------------

    def create_challenge(
        self, challenger_did: str | None = None, ttl_seconds: int = 300
    ) -> HandshakeChallenge:
        """Create a handshake challenge for verifying a peer agent."""
        nonce = KeyPair.generate_nonce()
        now = _utcnow()
        expires = now + timedelta(seconds=ttl_seconds)

        return HandshakeChallenge(
            nonce=nonce,
            timestamp=now,
            challenger_did=challenger_did or self.did,
            expires_at=expires,
        )

    def respond_to_challenge(self, challenge: HandshakeChallenge) -> HandshakeResponse:
        """Respond to a handshake challenge by signing the nonce."""
        if _utcnow() > challenge.expires_at:
            raise ValueError("Challenge has expired")

        signature = self.key_pair.sign_base64(challenge.nonce.encode())

        return HandshakeResponse(
            nonce=challenge.nonce,
            responder_did=self.did,
            signature=signature,
            public_key=self.public_key,
            metadata_hash=compute_hash(self._get_signable_data()),
        )

    def verify_response(
        self,
        challenge: HandshakeChallenge,
        response: HandshakeResponse,
        expected_metadata_hash: str | None = None,
    ) -> VerificationResult:
        """Verify a handshake response."""
        errors: list[str] = []

        if challenge.nonce != response.nonce:
            errors.append("Nonce mismatch")
            return VerificationResult(valid=False, errors=errors)

        if _utcnow() > challenge.expires_at:
            errors.append("Challenge has expired")
            return VerificationResult(valid=False, errors=errors)

        try:
            verify_key = KeyPair.from_public_key_base64(response.public_key)
            if not verify_key.verify_base64(
                response.nonce.encode(), response.signature
            ):
                errors.append("Signature verification failed")
                return VerificationResult(valid=False, errors=errors)
        except Exception as e:
            errors.append(f"Verification error: {e}")
            return VerificationResult(valid=False, errors=errors)

        expected_did = DID.from_public_key(verify_key.public_key_bytes)
        if expected_did.uri != response.responder_did:
            errors.append("DID does not match public key")
            return VerificationResult(valid=False, errors=errors)

        if expected_metadata_hash is not None:
            if not response.metadata_hash:
                errors.append("Missing metadata hash in response")
                return VerificationResult(valid=False, errors=errors)
            if response.metadata_hash != expected_metadata_hash:
                errors.append("Metadata hash mismatch")
                return VerificationResult(valid=False, errors=errors)

        return VerificationResult(valid=True, did=response.responder_did)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export card as a dictionary."""
        if not self.is_signed:
            raise ValueError("Card is not signed")
        return self._card.model_dump(mode="json", exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """Export card as a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def get_callback_handler(self) -> Any:
        """Get a LangChain callback handler that logs events to this AgentFacts instance."""
        from agentfacts.integrations.langchain.callback import AgentFactsCallbackHandler

        return AgentFactsCallbackHandler(agent_facts=self)

    def __repr__(self) -> str:
        signed_status = "signed" if self.is_signed else "unsigned"
        short_did = self._did.short_id() if self._did else "unknown"
        return f"AgentFacts(name={self.name!r}, did={short_did}, {signed_status})"

    def __str__(self) -> str:
        short_did = self._did.short_id() if self._did else "unknown"
        return f"{self.name} ({short_did})"

    # -------------------------------------------------------------------------
    # Collection-like Interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of capabilities."""
        return len(self._card.agent.capabilities)

    def __iter__(self) -> Iterator[Capability]:
        """Iterate over capabilities."""
        return iter(self._card.agent.capabilities)

    def __contains__(self, item: str | Capability) -> bool:
        """Check if a capability exists by name or instance."""
        if isinstance(item, str):
            return any(c.name == item for c in self._card.agent.capabilities)
        if isinstance(item, Capability):
            return item in self._card.agent.capabilities
        return False

    @overload
    def __getitem__(self, key: int) -> Capability: ...
    @overload
    def __getitem__(self, key: str) -> Capability: ...
    @overload
    def __getitem__(self, key: slice) -> list[Capability]: ...

    def __getitem__(self, key: int | str | slice) -> Capability | list[Capability]:
        """Access capabilities by index, name, or slice."""
        if isinstance(key, int):
            return self._card.agent.capabilities[key]
        if isinstance(key, slice):
            return self._card.agent.capabilities[key]
        if isinstance(key, str):
            for cap in self._card.agent.capabilities:
                if cap.name == key:
                    return cap
            raise KeyError(f"No capability named {key!r}")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __eq__(self, other: object) -> bool:
        """Compare by DID."""
        if isinstance(other, AgentFacts):
            return self.did == other.did
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by DID."""
        return hash(self.did)
