"""
GroupFacts - Identity and verification for multi-agent groups.

Provides cryptographic identity for agent groups/crews such as
CrewAI Crews or AutoGen GroupChats.

Pythonic Features:
- Fluent API with Self return types
- __len__ returns member count
- __iter__ iterates over member AgentFacts
- __contains__ checks for member by DID
- __getitem__ accesses members by index or DID
"""

from __future__ import annotations

from collections.abc import Iterator

# Import VerificationContext with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, cast, overload

from agentfacts.crypto.canonicalization import canonicalize_json
from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair
from agentfacts.models import (
    GroupMetadata,
    ProcessType,
    VerificationResult,
)
from agentfacts.utils import utcnow as _utcnow

if TYPE_CHECKING:
    from agentfacts.plugins import VerificationContext


class GroupFacts:
    """
    AgentFacts for multi-agent groups/crews.

    Manages cryptographic identity for a collection of agents working
    together, providing group-level signing and verification.

    Attributes:
        did: The group's Decentralised Identifier (DID:key format).
        name: Human-readable group name.
        metadata: Full GroupMetadata object with all group information.
        members: List of AgentFacts instances in this group.
        member_dids: List of DIDs for all member agents.
        is_signed: Whether the group metadata has been signed.
        key_pair: The Ed25519 key pair used for signing.

    Process Types:
        - SEQUENTIAL: Agents execute in order (default).
        - PARALLEL: Agents execute simultaneously.
        - HIERARCHICAL: Manager agent delegates to workers.
        - EVENT_DRIVEN: Agents respond to events (AutoGen style).
        - CUSTOM: User-defined topology.

    Example:
        ```python
        from agentfacts import AgentFacts, GroupFacts
        from agentfacts.models import ProcessType, BaselineModel, ModelProvider

        # Create individual agents with baseline models
        researcher = AgentFacts(
            name="Researcher",
            description="Finds relevant information",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        writer = AgentFacts(
            name="Writer",
            description="Writes content based on research",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        # Create a group with sequential execution
        crew = GroupFacts(
            name="Research Crew",
            members=[researcher, writer],
            process_type=ProcessType.SEQUENTIAL,
            description="A crew that researches and writes content",
        )

        # Sign the group (also signs all members)
        signatures = crew.sign_all()
        print(f"Signed {len(signatures)} entities")

        # Verify all signatures
        results = crew.verify_all()
        assert crew.all_verified()

        # Export to JSON
        print(crew.to_json(include_members=True))
        ```
    """

    def __init__(
        self,
        name: str,
        members: list[Any] | None = None,  # List of AgentFacts
        description: str = "",
        process_type: ProcessType = ProcessType.SEQUENTIAL,
        key_pair: KeyPair | None = None,
        did: DID | None = None,
        version: str = "1.0.0",
        framework: str | None = None,
    ) -> None:
        """
        Initialize GroupFacts.

        Args:
            name: Human-readable group name
            members: List of AgentFacts instances
            description: Group description
            process_type: How agents execute (sequential, parallel, etc.)
            key_pair: Ed25519 key pair (generated if not provided)
            did: Decentralised Identifier (derived from key_pair if not provided)
            version: Group version string
            framework: Source framework (crewai, autogen, etc.)
        """
        self._key_pair = key_pair or KeyPair.generate()
        self._did = did or DID.from_key_pair(self._key_pair)
        self._members: list[Any] = members or []
        self._signed = False

        # Build member DIDs
        member_dids = [m.did for m in self._members]

        # Build topology (default: sequential chain)
        topology: dict[str, list[str]] = {}
        if process_type == ProcessType.SEQUENTIAL and len(member_dids) > 1:
            for i, did_uri in enumerate(member_dids[:-1]):
                topology[did_uri] = [member_dids[i + 1]]
        elif (
            process_type in (ProcessType.PARALLEL, ProcessType.HIERARCHICAL)
            and member_dids
        ):
            # Parallel: entry agent can reach all others
            topology[member_dids[0]] = member_dids[1:]

        # Build metadata
        self._metadata = GroupMetadata(
            did=self._did.uri,
            name=name,
            description=description,
            version=version,
            member_dids=member_dids,
            entry_agent_did=member_dids[0] if member_dids else None,
            process_type=process_type,
            topology=topology,
            framework=framework,
            public_key=self._key_pair.public_key_base64,
        )

        # Register group membership on members
        for member in self._members:
            if (
                hasattr(member, "metadata")
                and hasattr(member.metadata, "agent")
                and self._did.uri not in member.metadata.agent.group_memberships
            ):
                member.metadata.agent.group_memberships.append(self._did.uri)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def did(self) -> str:
        """Get the group's DID."""
        return self._did.uri

    @property
    def name(self) -> str:
        """Get the group name."""
        return self._metadata.name

    @property
    def metadata(self) -> GroupMetadata:
        """Get the full metadata object."""
        return self._metadata

    @property
    def members(self) -> list[Any]:
        """Get the member AgentFacts instances."""
        return self._members

    @property
    def member_dids(self) -> list[str]:
        """Get DIDs of all members."""
        return self._metadata.member_dids

    @property
    def is_signed(self) -> bool:
        """Check if the group metadata is signed."""
        return self._signed and self._metadata.signature is not None

    @property
    def key_pair(self) -> KeyPair:
        """Get the key pair."""
        return self._key_pair

    # -------------------------------------------------------------------------
    # Member Management
    # -------------------------------------------------------------------------

    def _refresh_entry_agent(self) -> None:
        """Ensure entry agent remains valid after membership changes."""
        if not self._metadata.member_dids:
            self._metadata.entry_agent_did = None
            return
        if self._metadata.entry_agent_did not in self._metadata.member_dids:
            self._metadata.entry_agent_did = self._metadata.member_dids[0]

    def _refresh_topology(self) -> None:
        """Rebuild default topology for non-custom process types."""
        if self._metadata.process_type == ProcessType.CUSTOM:
            return

        member_dids = self._metadata.member_dids
        topology: dict[str, list[str]] = {}

        if (
            self._metadata.process_type == ProcessType.SEQUENTIAL
            and len(member_dids) > 1
        ):
            for i, did_uri in enumerate(member_dids[:-1]):
                topology[did_uri] = [member_dids[i + 1]]
        elif (
            self._metadata.process_type
            in (
                ProcessType.PARALLEL,
                ProcessType.HIERARCHICAL,
            )
            and member_dids
        ):
            topology[member_dids[0]] = member_dids[1:]

        self._metadata.topology = topology

    def add_member(self, agent_facts: Any) -> None:
        """
        Add a member to the group.

        Args:
            agent_facts: AgentFacts instance to add
        """
        if agent_facts.did not in self._metadata.member_dids:
            self._members.append(agent_facts)
            self._metadata.member_dids.append(agent_facts.did)

            # Register group on member
            if hasattr(agent_facts.metadata, "agent"):
                agent_facts.metadata.agent.group_memberships.append(self._did.uri)

            self._refresh_entry_agent()
            self._refresh_topology()

            self._signed = False

    def remove_member(self, did: str) -> bool:
        """
        Remove a member by DID.

        Args:
            did: DID of the member to remove

        Returns:
            True if member was found and removed
        """
        if did in self._metadata.member_dids:
            self._metadata.member_dids.remove(did)

            # Find and remove from members list
            for i, m in enumerate(self._members):
                if m.did == did:
                    # Remove group from member
                    if (
                        hasattr(m.metadata, "agent")
                        and self._did.uri in m.metadata.agent.group_memberships
                    ):
                        m.metadata.agent.group_memberships.remove(self._did.uri)
                    self._members.pop(i)
                    break

            # Update topology
            self._metadata.topology.pop(did, None)
            for k in self._metadata.topology:
                if did in self._metadata.topology[k]:
                    self._metadata.topology[k].remove(did)

            self._refresh_entry_agent()
            self._refresh_topology()

            self._signed = False
            return True
        return False

    def get_member(self, did: str) -> Any | None:
        """
        Get a member AgentFacts by its DID.

        Args:
            did: The Decentralised Identifier of the member to find.

        Returns:
            The AgentFacts instance if found, None otherwise.
        """
        for m in self._members:
            if m.did == did:
                return m
        return None

    # -------------------------------------------------------------------------
    # Topology Management
    # -------------------------------------------------------------------------

    def set_topology(self, topology: dict[str, list[str]]) -> None:
        """
        Set custom topology.

        Args:
            topology: Adjacency list mapping agent DIDs to reachable DIDs
        """
        self._metadata.topology = topology
        self._signed = False

    def add_edge(self, from_did: str, to_did: str) -> None:
        """
        Add a communication edge between two agents.

        Args:
            from_did: DID of the source agent.
            to_did: DID of the target agent.
        """
        if from_did not in self._metadata.topology:
            self._metadata.topology[from_did] = []
        if to_did not in self._metadata.topology[from_did]:
            self._metadata.topology[from_did].append(to_did)
            self._signed = False

    # -------------------------------------------------------------------------
    # Signing & Verification
    # -------------------------------------------------------------------------

    def sign(self, key_pair: KeyPair | None = None) -> GroupFacts:
        """
        Sign the group metadata.

        Args:
            key_pair: Optional key pair to use for signing.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the key pair is verification-only.
        """
        kp = key_pair or self._key_pair

        if not kp.can_sign():
            raise ValueError("Cannot sign: key pair is verification-only")

        self._metadata.updated_at = _utcnow()

        signable = self._get_signable_data()
        canonical = canonicalize_json(signable)
        signature = kp.sign_base64(canonical)

        self._metadata.signature = signature
        self._signed = True

        return self

    @property
    def signature(self) -> str | None:
        """Get the signature (base64-encoded) if signed."""
        return self._metadata.signature

    def sign_all(self, key_pair: KeyPair | None = None) -> dict[str, str]:
        """
        Sign the group and all member agents.

        Args:
            key_pair: Optional key pair for the group signature.

        Returns:
            Dict mapping DIDs to their Base64-encoded signatures.
        """
        signatures: dict[str, str] = {}

        # Sign all members first
        for member in self._members:
            if hasattr(member, "sign"):
                member.sign()
                signatures[member.did] = member.signature or ""

        # Sign the group
        self.sign(key_pair)
        signatures[self.did] = self.signature or ""

        return signatures

    def verify(
        self,
        key_pair: KeyPair | None = None,
        context: VerificationContext | None = None,
    ) -> VerificationResult:
        """
        Verify the group metadata signature.

        Args:
            key_pair: Optional key pair to verify against
            context: Optional VerificationContext with resolvers

        Returns:
            VerificationResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self._metadata.signature:
            errors.append("Group metadata is not signed")
            return VerificationResult(valid=False, did=self.did, errors=errors)

        # Resolve verification key
        verify_key = self._resolve_verification_key(key_pair, context, errors)
        if verify_key is None:
            return VerificationResult(valid=False, did=self.did, errors=errors)

        # Verify signature
        signable = self._get_signable_data()
        canonical = canonicalize_json(signable)

        if not verify_key.verify_base64(canonical, self._metadata.signature):
            errors.append("Signature verification failed")
            return VerificationResult(valid=False, did=self.did, errors=errors)

        # Verify DID matches
        expected_did = DID.from_public_key(verify_key.public_key_bytes)
        if expected_did.uri != self._metadata.did:
            warnings.append(f"DID mismatch: expected {expected_did.uri}")

        return VerificationResult(
            valid=True, did=self.did, errors=errors, warnings=warnings
        )

    def _resolve_verification_key(
        self,
        key_pair: KeyPair | None,
        context: VerificationContext | None,
        errors: list[str],
    ) -> KeyPair | None:
        """Resolve the verification key using context resolver or fallback."""
        if key_pair:
            return key_pair

        if context and context.did_resolver:
            resolver = context.did_resolver
            if resolver.supports(self._metadata.did):
                try:
                    resolved = resolver.resolve(self._metadata.did)
                    return KeyPair.from_public_key_base64(resolved.public_key_base64)
                except Exception as e:
                    if context.strict_did_verification:
                        errors.append(f"DID resolution failed: {e}")
                        return None

        if self._metadata.public_key:
            try:
                return KeyPair.from_public_key_base64(self._metadata.public_key)
            except Exception as e:
                errors.append(f"Invalid public key: {e}")
                return None

        errors.append("No public key available for verification")
        return None

    def verify_all(self) -> dict[str, VerificationResult]:
        """
        Verify the group and all member agents.

        Returns:
            Dict mapping DIDs to VerificationResults
        """
        results: dict[str, VerificationResult] = {}

        for member in self._members:
            if hasattr(member, "verify"):
                results[member.did] = member.verify()

        results[self.did] = self.verify()

        return results

    def all_verified(self) -> bool:
        """
        Check if the group and all members pass verification.

        Returns:
            True if all signatures are valid, False if any fail.
        """
        results = self.verify_all()
        return all(r.valid for r in results.values())

    def _get_signable_data(self) -> dict[str, Any]:
        """Get data to be signed (metadata without signature)."""
        data = self._metadata.model_dump(mode="json")
        data.pop("signature", None)
        return data

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self, include_members: bool = False) -> dict[str, Any]:
        """
        Export group metadata as dictionary.

        Args:
            include_members: Whether to include full member metadata

        Returns:
            Dict representation
        """
        data = self._metadata.model_dump(mode="json")
        if include_members:
            data["_members"] = [
                m.to_dict() for m in self._members if hasattr(m, "to_dict")
            ]
        return data

    def to_json(self, indent: int = 2, include_members: bool = False) -> str:
        """
        Export group metadata as a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.
            include_members: If True, includes full metadata for all members.

        Returns:
            JSON string representation.
        """
        import json

        return json.dumps(self.to_dict(include_members), indent=indent)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], key_pair: KeyPair | None = None
    ) -> GroupFacts:
        """Create GroupFacts from dictionary."""
        metadata = GroupMetadata.model_validate(data)

        if key_pair is None and metadata.public_key:
            key_pair = KeyPair.from_public_key_base64(metadata.public_key)

        group = cls(
            name=metadata.name,
            description=metadata.description,
            process_type=metadata.process_type,
            key_pair=key_pair,
            did=DID.parse(metadata.did) if metadata.did else None,
            version=metadata.version,
            framework=metadata.framework,
        )

        group._metadata = metadata
        if metadata.signature:
            group._signed = True

        return group

    @classmethod
    def from_json(cls, json_str: str, key_pair: KeyPair | None = None) -> GroupFacts:
        """Load GroupFacts from a JSON string."""
        import json

        return cls.from_dict(json.loads(json_str), key_pair)

    # -------------------------------------------------------------------------
    # Framework Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_agent(
        cls,
        multi_agent: Any,
        name: str = "Agent Group",
        framework: str | None = None,
        key_pair: KeyPair | None = None,
        **kwargs: Any,
    ) -> GroupFacts:
        """
        Create GroupFacts with auto-detection.

        This is the universal factory method that auto-detects the framework
        and dispatches to the appropriate framework-specific factory.

        Args:
            multi_agent: Multi-agent object from any supported framework
            name: Group name
            framework: Explicit framework name (auto-detected if None)
            key_pair: Optional key pair for the group
            **kwargs: Additional arguments

        Returns:
            GroupFacts instance

        Supported frameworks:
            - crewai: Crew objects
            - autogen: GroupChat, GroupChatManager
            - llamaindex: QueryEngine, AgentRunner, Index
            - openagents: Network, AgentConfig, WorkerAgent
            - huggingface: smolagents, tiny-agents
        """
        from agentfacts.integrations import get_registry

        registry = get_registry()

        if framework is None:
            framework = registry.detect_framework(multi_agent)

        if framework is None:
            raise ValueError(
                f"Cannot detect framework for {type(multi_agent).__name__}. "
                "Please specify the framework explicitly."
            )

        # Import factory functions from integration modules
        factory_map = {
            "crewai": "agentfacts.integrations.crewai.factory",
            "autogen": "agentfacts.integrations.autogen.factory",
            "llamaindex": "agentfacts.integrations.llamaindex.factory",
            "openagents": "agentfacts.integrations.openagents.factory",
            "huggingface": "agentfacts.integrations.huggingface.factory",
        }

        factory_func_map = {
            "crewai": "create_group_from_crew",
            "autogen": "create_group_from_autogen",
            "llamaindex": "create_group_from_llamaindex",
            "openagents": "create_group_from_openagents",
            "huggingface": "create_group_from_huggingface",
        }

        module_name = factory_map.get(framework)
        func_name = factory_func_map.get(framework)

        if module_name is None or func_name is None:
            raise ValueError(
                f"No GroupFacts factory for framework '{framework}'. "
                f"Available: {list(factory_map.keys())}"
            )

        import importlib

        module = importlib.import_module(module_name)
        factory_func = getattr(module, func_name)

        return cast(GroupFacts, factory_func(multi_agent, name, key_pair, **kwargs))

    @classmethod
    def from_agent_signed(
        cls,
        multi_agent: Any,
        name: str = "Agent Group",
        framework: str | None = None,
        key_pair: KeyPair | None = None,
        **kwargs: Any,
    ) -> GroupFacts:
        """Create and sign GroupFacts with auto-detection."""
        group = cls.from_agent(multi_agent, name, framework, key_pair, **kwargs)
        group.sign_all()
        return group

    # Convenience aliases for common frameworks
    @classmethod
    def from_crewai(
        cls,
        crew: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create GroupFacts from a CrewAI Crew."""
        from agentfacts.integrations.crewai.factory import create_group_from_crew

        return create_group_from_crew(crew, name, key_pair)

    @classmethod
    def from_crewai_signed(
        cls,
        crew: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create and sign GroupFacts from a CrewAI Crew."""
        group = cls.from_crewai(crew, name, key_pair)
        group.sign_all()
        return group

    @classmethod
    def from_autogen(
        cls,
        group_chat: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create GroupFacts from an AutoGen GroupChat."""
        from agentfacts.integrations.autogen.factory import create_group_from_autogen

        return create_group_from_autogen(group_chat, name, key_pair)

    @classmethod
    def from_autogen_signed(
        cls,
        group_chat: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create and sign GroupFacts from an AutoGen GroupChat."""
        group = cls.from_autogen(group_chat, name, key_pair)
        group.sign_all()
        return group

    @classmethod
    def from_llamaindex(
        cls,
        query_engine_or_agent: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create GroupFacts from a LlamaIndex query engine or agent."""
        from agentfacts.integrations.llamaindex.factory import (
            create_group_from_llamaindex,
        )

        return create_group_from_llamaindex(query_engine_or_agent, name, key_pair)

    @classmethod
    def from_openagents(
        cls,
        network_or_agent: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create GroupFacts from an OpenAgents network or agent."""
        from agentfacts.integrations.openagents.factory import (
            create_group_from_openagents,
        )

        return create_group_from_openagents(network_or_agent, name, key_pair)

    @classmethod
    def from_huggingface(
        cls,
        agent_or_config: Any,
        name: str | None = None,
        key_pair: KeyPair | None = None,
    ) -> GroupFacts:
        """Create GroupFacts from a HuggingFace agent or config."""
        from agentfacts.integrations.huggingface.factory import (
            create_group_from_huggingface,
        )

        return create_group_from_huggingface(agent_or_config, name, key_pair)

    def __repr__(self) -> str:
        signed = "signed" if self.is_signed else "unsigned"
        return (
            f"GroupFacts(name={self.name!r}, "
            f"members={len(self._members)}, "
            f"process={self._metadata.process_type.value}, "
            f"{signed})"
        )

    def __str__(self) -> str:
        return f"{self.name} ({len(self._members)} members)"

    # -------------------------------------------------------------------------
    # Collection-like Interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of member agents."""
        return len(self._members)

    def __iter__(self) -> Iterator[Any]:  # Iterator[AgentFacts]
        """Iterate over member AgentFacts."""
        return iter(self._members)

    def __contains__(self, item: str | Any) -> bool:
        """
        Check if an agent is a member by DID or instance.

        Example:
            >>> "did:key:z6Mk..." in group
            True
            >>> agent_facts in group
            True
        """
        if isinstance(item, str):
            return item in self._metadata.member_dids
        if hasattr(item, "did"):
            return item.did in self._metadata.member_dids
        return False

    @overload
    def __getitem__(self, key: int) -> Any: ...  # AgentFacts
    @overload
    def __getitem__(self, key: str) -> Any: ...  # AgentFacts
    @overload
    def __getitem__(self, key: slice) -> list[Any]: ...  # list[AgentFacts]

    def __getitem__(self, key: int | str | slice) -> Any:
        """
        Access members by index, DID, or slice.

        Example:
            >>> group[0]                    # First member
            >>> group["did:key:z6Mk..."]    # Member by DID
            >>> group[1:3]                  # Slice of members
        """
        if isinstance(key, int):
            return self._members[key]
        if isinstance(key, slice):
            return self._members[key]
        if isinstance(key, str):
            for member in self._members:
                if hasattr(member, "did") and member.did == key:
                    return member
            raise KeyError(f"No member with DID {key!r}")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __eq__(self, other: object) -> bool:
        """Compare by DID."""
        if isinstance(other, GroupFacts):
            return self.did == other.did
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by DID."""
        return hash(self.did)
