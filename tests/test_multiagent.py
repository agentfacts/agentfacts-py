"""
Tests for multi-agent support: GroupFacts, AgentRole, DelegationPolicy.
"""

import json

from agentfacts import (
    AgentFacts,
    AgentRole,
    BaselineModel,
    Capability,
    DelegationPolicy,
    GroupFacts,
    GroupMetadata,
    ModelProvider,
    ProcessType,
)


class TestAgentRole:
    """Tests for AgentRole model."""

    def test_create_role(self):
        """Test creating an agent role."""
        role = AgentRole(
            role_name="Senior Researcher",
            goal="Find cutting-edge information",
            backstory="You have 20 years of research experience",
            hierarchy_level=1,
        )
        assert role.role_name == "Senior Researcher"
        assert role.goal == "Find cutting-edge information"
        assert role.backstory == "You have 20 years of research experience"
        assert role.hierarchy_level == 1

    def test_role_defaults(self):
        """Test role default values."""
        role = AgentRole(role_name="Worker")
        assert role.goal == ""
        assert role.backstory == ""
        assert role.hierarchy_level == 0

    def test_role_serialization(self):
        """Test role serialization to dict."""
        role = AgentRole(
            role_name="Manager",
            goal="Oversee the team",
            hierarchy_level=2,
        )
        data = role.model_dump()
        assert data["role_name"] == "Manager"
        assert data["hierarchy_level"] == 2


class TestDelegationPolicy:
    """Tests for DelegationPolicy model."""

    def test_create_delegation_policy(self):
        """Test creating a delegation policy."""
        policy = DelegationPolicy(
            can_delegate=True,
            delegatable_to=["did:key:abc", "did:key:xyz"],
            requires_approval_to_delegate=False,
            max_delegation_depth=3,
        )
        assert policy.can_delegate is True
        assert len(policy.delegatable_to) == 2
        assert policy.requires_approval_to_delegate is False
        assert policy.max_delegation_depth == 3

    def test_delegation_defaults(self):
        """Test delegation policy defaults."""
        policy = DelegationPolicy()
        assert policy.can_delegate is False
        assert policy.can_receive_delegation is True
        assert policy.delegatable_to == []
        assert policy.requires_approval_to_delegate is True
        assert policy.max_delegation_depth == 1


class TestCapabilityDelegation:
    """Tests for delegation fields on Capability."""

    def test_capability_delegation_fields(self):
        """Test capability delegation fields."""
        cap = Capability(
            name="web_search",
            description="Search the web",
            delegatable=True,
            delegatable_to=["did:key:worker1"],
        )
        assert cap.delegatable is True
        assert cap.delegatable_to == ["did:key:worker1"]

    def test_capability_delegation_defaults(self):
        """Test capability delegation defaults."""
        cap = Capability(name="calculator")
        assert cap.delegatable is False
        assert cap.delegatable_to == []


class TestAgentFactsMultiAgent:
    """Tests for multi-agent fields on AgentFacts cards."""

    def test_metadata_with_role(self):
        """Test AgentFacts card with role."""
        facts = AgentFacts(
            name="Researcher",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.agent.role = AgentRole(
            role_name="Senior Researcher",
            goal="Find information",
        )
        assert facts.metadata.agent.role is not None
        assert facts.metadata.agent.role.role_name == "Senior Researcher"

    def test_metadata_with_delegation(self):
        """Test AgentFacts card with delegation policy."""
        facts = AgentFacts(
            name="Manager",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.agent.delegation = DelegationPolicy(
            can_delegate=True,
            max_delegation_depth=2,
        )
        assert facts.metadata.agent.delegation.can_delegate is True

    def test_metadata_group_memberships(self):
        """Test group membership tracking."""
        facts = AgentFacts(
            name="Worker",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.agent.group_memberships.append("did:key:group1")
        assert "did:key:group1" in facts.metadata.agent.group_memberships

    def test_metadata_framework_field(self):
        """Test framework field."""
        facts = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.agent.framework = "crewai"
        assert facts.metadata.agent.framework == "crewai"


class TestGroupMetadata:
    """Tests for GroupMetadata model."""

    def test_create_group_metadata(self):
        """Test creating group metadata."""
        metadata = GroupMetadata(
            did="did:key:group123",
            name="Research Crew",
            member_dids=["did:key:agent1", "did:key:agent2"],
            process_type=ProcessType.SEQUENTIAL,
        )
        assert metadata.did == "did:key:group123"
        assert metadata.name == "Research Crew"
        assert len(metadata.member_dids) == 2
        assert metadata.process_type == ProcessType.SEQUENTIAL

    def test_group_topology(self):
        """Test group topology (communication graph)."""
        metadata = GroupMetadata(
            did="did:key:group",
            name="Team",
            topology={
                "did:key:manager": ["did:key:worker1", "did:key:worker2"],
                "did:key:worker1": ["did:key:worker2"],
            },
        )
        assert "did:key:manager" in metadata.topology
        assert len(metadata.topology["did:key:manager"]) == 2

    def test_process_types(self):
        """Test all process types."""
        for pt in ProcessType:
            metadata = GroupMetadata(
                did="did:key:test",
                name="Test",
                process_type=pt,
            )
            assert metadata.process_type == pt


class TestGroupFacts:
    """Tests for GroupFacts class."""

    def test_create_empty_group(self):
        """Test creating an empty group."""
        group = GroupFacts(name="Empty Crew")
        assert group.name == "Empty Crew"
        assert len(group.members) == 0
        assert group.did.startswith("did:key:")

    def test_create_group_with_members(self):
        """Test creating group with members."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group = GroupFacts(
            name="My Crew",
            members=[agent1, agent2],
            process_type=ProcessType.SEQUENTIAL,
        )

        assert len(group.members) == 2
        assert len(group.member_dids) == 2
        assert agent1.did in group.member_dids
        assert agent2.did in group.member_dids

    def test_group_registers_membership_on_agents(self):
        """Test that group membership is registered on member agents."""
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent])

        assert group.did in agent.metadata.agent.group_memberships

    def test_add_member(self):
        """Test adding member to group."""
        group = GroupFacts(name="Crew")
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group.add_member(agent)

        assert len(group.members) == 1
        assert agent.did in group.member_dids
        assert group.did in agent.metadata.agent.group_memberships

    def test_remove_member(self):
        """Test removing member from group."""
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent])

        result = group.remove_member(agent.did)

        assert result is True
        assert len(group.members) == 0
        assert agent.did not in group.member_dids
        assert group.did not in agent.metadata.agent.group_memberships

    def test_sign_group(self):
        """Test signing group metadata."""
        group = GroupFacts(name="Crew")
        result = group.sign()

        # sign() returns self for fluent API
        assert result is group
        assert group.is_signed
        assert group.signature is not None
        assert group.metadata.signature == group.signature

    def test_verify_group(self):
        """Test verifying group signature."""
        group = GroupFacts(name="Crew")
        group.sign()

        result = group.verify()

        assert result.valid is True
        assert result.did == group.did

    def test_sign_all_members(self):
        """Test signing group and all members."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent1, agent2])

        signatures = group.sign_all()

        assert len(signatures) == 3  # 2 agents + 1 group
        assert agent1.did in signatures
        assert agent2.did in signatures
        assert group.did in signatures
        assert agent1.is_signed
        assert agent2.is_signed
        assert group.is_signed

    def test_verify_all_members(self):
        """Test verifying group and all members."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent1, agent2])
        group.sign_all()

        results = group.verify_all()

        assert len(results) == 3
        assert all(r.valid for r in results.values())

    def test_all_verified(self):
        """Test all_verified helper."""
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent])

        # Not signed yet
        assert group.all_verified() is False

        group.sign_all()
        assert group.all_verified() is True

    def test_sequential_topology(self):
        """Test sequential process generates correct topology."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent3 = AgentFacts(
            name="Agent3",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group = GroupFacts(
            name="Crew",
            members=[agent1, agent2, agent3],
            process_type=ProcessType.SEQUENTIAL,
        )

        # Sequential: agent1 -> agent2 -> agent3
        assert agent1.did in group.metadata.topology
        assert agent2.did in group.metadata.topology[agent1.did]
        assert agent2.did in group.metadata.topology
        assert agent3.did in group.metadata.topology[agent2.did]

    def test_add_member_updates_topology_sequential(self):
        """Test adding a member refreshes sequential topology."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent3 = AgentFacts(
            name="Agent3",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group = GroupFacts(
            name="Crew",
            members=[agent1, agent2],
            process_type=ProcessType.SEQUENTIAL,
        )

        group.add_member(agent3)

        assert group.metadata.entry_agent_did == agent1.did
        assert group.metadata.topology[agent1.did] == [agent2.did]
        assert group.metadata.topology[agent2.did] == [agent3.did]

    def test_remove_member_updates_entry_and_topology(self):
        """Test removing a member refreshes entry agent and topology."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent3 = AgentFacts(
            name="Agent3",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group = GroupFacts(
            name="Crew",
            members=[agent1, agent2, agent3],
            process_type=ProcessType.SEQUENTIAL,
        )

        removed = group.remove_member(agent1.did)

        assert removed is True
        assert group.metadata.entry_agent_did == agent2.did
        assert agent1.did not in group.metadata.topology
        assert group.metadata.topology[agent2.did] == [agent3.did]

    def test_add_member_updates_topology_parallel(self):
        """Test adding a member refreshes parallel topology."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent3 = AgentFacts(
            name="Agent3",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )

        group = GroupFacts(
            name="Crew",
            members=[agent1, agent2],
            process_type=ProcessType.PARALLEL,
        )

        group.add_member(agent3)

        assert group.metadata.entry_agent_did == agent1.did
        assert set(group.metadata.topology[agent1.did]) == {agent2.did, agent3.did}

    def test_custom_topology(self):
        """Test setting custom topology."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent1, agent2])

        # Set hub-and-spoke topology
        group.set_topology(
            {
                agent1.did: [agent2.did],
                agent2.did: [agent1.did],
            }
        )

        assert agent2.did in group.metadata.topology[agent1.did]
        assert agent1.did in group.metadata.topology[agent2.did]

    def test_add_edge(self):
        """Test adding communication edge."""
        agent1 = AgentFacts(
            name="Agent1",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Agent2",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent1, agent2])
        group.set_topology({})  # Clear topology

        group.add_edge(agent1.did, agent2.did)

        assert agent2.did in group.metadata.topology[agent1.did]

    def test_group_serialization(self):
        """Test group serialization to dict/JSON."""
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(
            name="Crew",
            members=[agent],
            process_type=ProcessType.HIERARCHICAL,
            framework="crewai",
        )
        group.sign_all()

        # To dict
        data = group.to_dict()
        assert data["name"] == "Crew"
        assert data["framework"] == "crewai"
        assert data["process_type"] == "hierarchical"

        # To JSON
        json_str = group.to_json()
        parsed = json.loads(json_str)
        assert parsed["name"] == "Crew"

    def test_group_serialization_with_members(self):
        """Test group serialization including member data."""
        agent = AgentFacts(
            name="Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent])
        group.sign_all()

        data = group.to_dict(include_members=True)
        assert "_members" in data
        assert len(data["_members"]) == 1

    def test_group_from_dict(self):
        """Test creating group from dict."""
        group = GroupFacts(name="Original", framework="crewai")
        group.sign()
        data = group.to_dict()

        restored = GroupFacts.from_dict(data)
        assert restored.name == "Original"
        assert restored.metadata.framework == "crewai"
        assert restored.is_signed

    def test_group_from_json(self):
        """Test creating group from JSON."""
        group = GroupFacts(name="Original")
        group.sign()
        json_str = group.to_json()

        restored = GroupFacts.from_json(json_str)
        assert restored.name == "Original"
        assert restored.verify().valid

    def test_entry_agent(self):
        """Test entry agent is set correctly."""
        agent1 = AgentFacts(
            name="Entry",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        agent2 = AgentFacts(
            name="Worker",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        group = GroupFacts(name="Crew", members=[agent1, agent2])

        assert group.metadata.entry_agent_did == agent1.did

    def test_group_repr(self):
        """Test group string representation."""
        group = GroupFacts(
            name="My Crew",
            process_type=ProcessType.SEQUENTIAL,
        )
        repr_str = repr(group)
        assert "My Crew" in repr_str
        assert "sequential" in repr_str
        assert "unsigned" in repr_str

        group.sign()
        repr_str = repr(group)
        assert "signed" in repr_str
