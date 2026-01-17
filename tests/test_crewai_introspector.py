"""
Tests for CrewAI introspector.
"""

from agentfacts import AgentFacts, GroupFacts, ProcessType
from agentfacts.integrations.crewai.introspector import (
    CrewAIIntegration as CrewAIIntrospector,
)


class MockTool:
    """Mock CrewAI tool."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.args_schema = None


class MockLLM:
    """Mock LLM for CrewAI."""

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = 1000


class MockCrewAIAgent:
    """Mock CrewAI Agent."""

    __module__ = "crewai.agents"

    def __init__(
        self,
        role: str = "Researcher",
        goal: str = "Find information",
        backstory: str = "Expert researcher",
        tools: list = None,
        llm: MockLLM = None,
        allow_delegation: bool = False,
        verbose: bool = False,
        memory: bool = False,
        max_iter: int = 15,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = llm
        self.allow_delegation = allow_delegation
        self.verbose = verbose
        self.memory = memory if memory else None
        self.max_iter = max_iter


class MockTask:
    """Mock CrewAI Task."""

    def __init__(self, description: str = "Do something"):
        self.description = description


class MockProcess:
    """Mock CrewAI Process enum."""

    def __init__(self, value: str):
        self._value = value

    def __str__(self):
        return self._value


class MockCrew:
    """Mock CrewAI Crew."""

    __module__ = "crewai.crew"

    def __init__(
        self,
        agents: list = None,
        tasks: list = None,
        process: MockProcess = None,
        verbose: bool = False,
        memory: bool = False,
    ):
        self.agents = agents or []
        self.tasks = tasks or []
        self.process = process or MockProcess("sequential")
        self.verbose = verbose
        self.memory = memory if memory else None


class TestCrewAIIntrospector:
    """Tests for CrewAI introspector."""

    def test_framework_name(self):
        """Test framework name."""
        introspector = CrewAIIntrospector()
        assert introspector.framework_name == "crewai"

    def test_can_introspect_agent(self):
        """Test can_introspect with CrewAI agent."""
        introspector = CrewAIIntrospector()
        agent = MockCrewAIAgent()
        assert introspector.can_introspect(agent) is True

    def test_can_introspect_crew(self):
        """Test can_introspect with CrewAI crew."""
        introspector = CrewAIIntrospector()
        crew = MockCrew()
        assert introspector.can_introspect(crew) is True

    def test_cannot_introspect_random(self):
        """Test can_introspect rejects random objects."""
        introspector = CrewAIIntrospector()
        assert introspector.can_introspect("string") is False
        assert introspector.can_introspect(123) is False
        assert introspector.can_introspect({}) is False

    def test_introspect_agent_basic(self):
        """Test basic agent introspection."""
        introspector = CrewAIIntrospector()
        agent = MockCrewAIAgent(
            role="Data Analyst",
            goal="Analyze data patterns",
            backstory="Expert in data science",
        )

        result = introspector.introspect(agent)

        assert "role" in result.context
        assert result.context["role"]["role_name"] == "Data Analyst"
        assert result.context["role"]["goal"] == "Analyze data patterns"
        assert result.context["role"]["backstory"] == "Expert in data science"

    def test_introspect_agent_with_llm(self):
        """Test agent introspection with LLM."""
        introspector = CrewAIIntrospector()
        llm = MockLLM(model_name="gpt-4-turbo", temperature=0.5)
        agent = MockCrewAIAgent(llm=llm)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4-turbo"
        assert result.baseline_model.temperature == 0.5

    def test_introspect_agent_with_tools(self):
        """Test agent introspection with tools."""
        introspector = CrewAIIntrospector()
        tools = [
            MockTool("web_search", "Search the web"),
            MockTool("file_write", "Write files to disk"),
        ]
        agent = MockCrewAIAgent(tools=tools)

        result = introspector.introspect(agent)

        assert len(result.capabilities) == 2
        assert result.capabilities[0].name == "web_search"
        assert result.capabilities[1].name == "file_write"
        # file_write should be high risk
        assert result.capabilities[1].risk_level == "high"

    def test_introspect_agent_delegation(self):
        """Test agent introspection with delegation enabled."""
        introspector = CrewAIIntrospector()
        agent = MockCrewAIAgent(allow_delegation=True)

        result = introspector.introspect(agent)

        assert "delegation" in result.context
        assert result.context["delegation"]["can_delegate"] is True

    def test_introspect_agent_constraints(self):
        """Test agent introspection extracts constraints."""
        introspector = CrewAIIntrospector()
        agent = MockCrewAIAgent(max_iter=25)

        result = introspector.introspect(agent)

        assert result.constraints.max_iterations == 25

    def test_introspect_crew_basic(self):
        """Test basic crew introspection."""
        introspector = CrewAIIntrospector()
        agents = [
            MockCrewAIAgent(role="Researcher"),
            MockCrewAIAgent(role="Writer"),
        ]
        crew = MockCrew(agents=agents)

        result = introspector.introspect(crew)

        assert result.context["agent_count"] == 2
        assert "Researcher" in result.context["agent_roles"]
        assert "Writer" in result.context["agent_roles"]

    def test_introspect_crew_process_type(self):
        """Test crew introspection extracts process type."""
        introspector = CrewAIIntrospector()

        # Sequential
        crew_seq = MockCrew(process=MockProcess("sequential"))
        result_seq = introspector.introspect(crew_seq)
        assert result_seq.context["process_type"] == "sequential"

        # Hierarchical
        crew_hier = MockCrew(process=MockProcess("hierarchical"))
        result_hier = introspector.introspect(crew_hier)
        assert result_hier.context["process_type"] == "hierarchical"

    def test_introspect_crew_with_tasks(self):
        """Test crew introspection extracts tasks."""
        introspector = CrewAIIntrospector()
        tasks = [
            MockTask("Research the topic"),
            MockTask("Write the article"),
        ]
        crew = MockCrew(tasks=tasks)

        result = introspector.introspect(crew)

        assert result.context["task_count"] == 2
        assert len(result.context["task_descriptions"]) == 2

    def test_introspect_crew_collects_all_tools(self):
        """Test crew introspection collects tools from all agents."""
        introspector = CrewAIIntrospector()
        agents = [
            MockCrewAIAgent(tools=[MockTool("search")]),
            MockCrewAIAgent(tools=[MockTool("write"), MockTool("read")]),
        ]
        crew = MockCrew(agents=agents)

        result = introspector.introspect(crew)

        assert len(result.capabilities) == 3
        names = {c.name for c in result.capabilities}
        assert names == {"search", "write", "read"}

    def test_get_agent_type(self):
        """Test get_agent_type returns correct types."""
        introspector = CrewAIIntrospector()

        agent = MockCrewAIAgent()
        agent_type = introspector.get_agent_type(agent)
        # Mock classes return their class name
        assert agent_type is not None

        delegating_agent = MockCrewAIAgent(allow_delegation=True)
        assert introspector.get_agent_type(delegating_agent) is not None

        crew = MockCrew(process=MockProcess("HIERARCHICAL"))
        assert introspector.get_agent_type(crew) is not None


class TestGroupFactsFromCrewAI:
    """Tests for GroupFacts.from_crewai()."""

    def test_from_crewai_basic(self):
        """Test creating GroupFacts from CrewAI crew."""
        agents = [
            MockCrewAIAgent(
                role="Researcher",
                goal="Find information",
                llm=MockLLM("gpt-4"),
            ),
            MockCrewAIAgent(
                role="Writer",
                goal="Write articles",
                llm=MockLLM("gpt-4"),
            ),
        ]
        crew = MockCrew(agents=agents)

        group = GroupFacts.from_crewai(crew, name="Research Team")

        assert group.name == "Research Team"
        assert len(group.members) == 2
        assert group.metadata.framework == "crewai"
        assert group.metadata.process_type == ProcessType.SEQUENTIAL

    def test_from_crewai_hierarchical(self):
        """Test creating GroupFacts from hierarchical crew."""
        agents = [MockCrewAIAgent(), MockCrewAIAgent()]
        crew = MockCrew(agents=agents, process=MockProcess("hierarchical"))

        group = GroupFacts.from_crewai(crew)

        assert group.metadata.process_type == ProcessType.HIERARCHICAL

    def test_from_crewai_members_have_roles(self):
        """Test that member AgentFacts have role information."""
        agent = MockCrewAIAgent(
            role="Senior Researcher",
            goal="Lead research efforts",
            backstory="20 years experience",
        )
        crew = MockCrew(agents=[agent])

        group = GroupFacts.from_crewai(crew)

        member = group.members[0]
        assert member.metadata.agent.role is not None
        assert member.metadata.agent.role.role_name == "Senior Researcher"
        assert member.metadata.agent.role.goal == "Lead research efforts"

    def test_from_crewai_members_have_delegation(self):
        """Test that member AgentFacts have delegation policy."""
        agent = MockCrewAIAgent(allow_delegation=True)
        crew = MockCrew(agents=[agent])

        group = GroupFacts.from_crewai(crew)

        member = group.members[0]
        assert member.metadata.agent.delegation.can_delegate is True

    def test_from_crewai_shared_memory(self):
        """Test crew with shared memory."""
        crew = MockCrew(agents=[MockCrewAIAgent()], memory=True)

        group = GroupFacts.from_crewai(crew)

        assert group.metadata.shared_memory is True

    def test_from_crewai_sign_all(self):
        """Test signing CrewAI group and members."""
        agents = [MockCrewAIAgent(), MockCrewAIAgent()]
        crew = MockCrew(agents=agents)

        group = GroupFacts.from_crewai(crew)
        signatures = group.sign_all()

        assert len(signatures) == 3  # 2 agents + 1 group
        assert group.all_verified()


class TestAgentFactsFromCrewAI:
    """Tests for AgentFacts.from_agent with CrewAI."""

    def test_from_agent_crewai_auto_detect(self):
        """Test AgentFacts.from_agent auto-detects CrewAI."""
        agent = MockCrewAIAgent(
            role="Tester",
            goal="Test things",
            llm=MockLLM("gpt-4"),
        )

        # This should work because CrewAI introspector is registered
        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()

        # Manually register CrewAI introspector for test
        registry = get_registry()
        registry.register(CrewAIIntrospector())

        facts = AgentFacts.from_agent(agent, name="Test Agent")

        assert facts.name == "Test Agent"
        assert facts.metadata.agent.framework == "crewai"

    def test_from_agent_crewai_explicit(self):
        """Test AgentFacts.from_agent with explicit framework."""
        agent = MockCrewAIAgent(llm=MockLLM("claude-3"))

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(CrewAIIntrospector())

        facts = AgentFacts.from_agent(agent, name="Claude Agent", framework="crewai")

        assert facts.metadata.agent.framework == "crewai"
