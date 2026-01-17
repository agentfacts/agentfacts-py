#!/usr/bin/env python3
"""
AgentFacts Integration Test Runner

Tests AgentFacts SDK against real-world agent implementations
from various frameworks.

Usage:
    python run_tests.py                    # Run all frameworks
    python run_tests.py --framework crewai # Run specific framework
    python run_tests.py --list             # List available tests
"""

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

agentfacts_module = importlib.import_module("agentfacts")
AgentFacts = agentfacts_module.AgentFacts
ModelProvider = agentfacts_module.ModelProvider
Policy = importlib.import_module("agentfacts.policy").Policy

# ---------------------------------------------------------------------------
# Test Result Classes
# ---------------------------------------------------------------------------


@dataclass
class TestCaseResult:
    """Result of a single test case."""

    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class AgentTestResult:
    """Result of testing a single agent."""

    agent_name: str
    agent_type: str
    framework: str
    tests: list[TestCaseResult] = field(default_factory=list)
    metadata_file: str | None = None

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tests)

    @property
    def pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        return sum(1 for t in self.tests if t.passed) / len(self.tests)


@dataclass
class RepoTestResult:
    """Result of testing a repository."""

    repo_name: str
    repo_url: str
    framework: str
    agents: list[AgentTestResult] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.agents)


@dataclass
class FrameworkTestResult:
    """Result of testing a framework."""

    framework: str
    repos: list[RepoTestResult] = field(default_factory=list)

    @property
    def total_agents(self) -> int:
        return sum(len(r.agents) for r in self.repos)

    @property
    def passed_agents(self) -> int:
        return sum(1 for r in self.repos for a in r.agents if a.passed)

    @property
    def skipped_repos(self) -> int:
        return sum(1 for r in self.repos if r.skipped)


@dataclass
class RepoSpec:
    """Specification for a repository under test."""

    name: str
    url: str
    path: Path
    agents: list[tuple[str, Any]] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs all test cases for an agent."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

    def run_all_tests(
        self,
        agent: Any,
        name: str,
        framework: str,
        context: dict[str, Any] | None = None,
    ) -> AgentTestResult:
        """Run all test cases for an agent."""
        result = AgentTestResult(
            agent_name=name,
            agent_type=type(agent).__name__,
            framework=framework,
        )

        # Test 1: Identity Creation
        result.tests.append(self._test_identity(agent, name, context))

        # Test 2: Signing
        facts = self._get_facts(agent, name, context)
        if facts is not None:
            result.tests.append(self._test_signing(facts))

            # Test 3: Verification
            result.tests.append(self._test_verification(facts))

            # Test 4: Introspection Quality
            result.tests.append(self._test_introspection(facts))

            # Test 5: Handshake
            result.tests.append(self._test_handshake(facts))

            # Test 6: Policy Evaluation
            result.tests.append(self._test_policy(facts))

            # Test 7: JSON Round-trip (before transparency to preserve signature)
            result.tests.append(self._test_json_roundtrip(facts))

            # Test 8: Transparency Log (modifies merkle root, run after JSON test)
            result.tests.append(self._test_transparency(facts))

            # Generate artifacts
            result.metadata_file = self._generate_artifacts(facts, name, framework)

        return result

    def _get_facts(
        self,
        agent: Any,
        name: str,
        context: dict[str, Any] | None = None,
    ) -> AgentFacts | None:
        """Create AgentFacts from agent."""
        try:
            return AgentFacts.from_agent(agent, name, context=context or {})
        except Exception:
            return None

    def _test_identity(
        self,
        agent: Any,
        name: str,
        context: dict[str, Any] | None = None,
    ) -> TestCaseResult:
        """Test identity creation."""
        start = datetime.now(timezone.utc)
        try:
            facts = AgentFacts.from_agent(agent, name, context=context or {})
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not facts.did.startswith("did:key:"):
                return TestCaseResult(
                    name="identity",
                    passed=False,
                    message=f"Invalid DID format: {facts.did}",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="identity",
                passed=True,
                message=f"Created DID: {facts.did[:40]}...",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="identity",
                passed=False,
                message="Failed to create identity",
                duration_ms=duration,
                error=str(e),
            )

    def _test_signing(self, facts: AgentFacts) -> TestCaseResult:
        """Test metadata signing."""
        start = datetime.now(timezone.utc)
        try:
            facts.sign()
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not facts.is_signed:
                return TestCaseResult(
                    name="signing",
                    passed=False,
                    message="Sign completed but is_signed=False",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="signing",
                passed=True,
                message=f"Signature: {facts.signature[:30]}...",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="signing",
                passed=False,
                message="Failed to sign",
                duration_ms=duration,
                error=str(e),
            )

    def _test_verification(self, facts: AgentFacts) -> TestCaseResult:
        """Test signature verification."""
        start = datetime.now(timezone.utc)
        try:
            result = facts.verify()
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not result.valid:
                return TestCaseResult(
                    name="verification",
                    passed=False,
                    message=f"Verification failed: {result.errors}",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="verification",
                passed=True,
                message="Signature verified",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="verification",
                passed=False,
                message="Verification raised exception",
                duration_ms=duration,
                error=str(e),
            )

    def _test_introspection(self, facts: AgentFacts) -> TestCaseResult:
        """Test introspection quality."""
        start = datetime.now(timezone.utc)
        try:
            metadata = facts.metadata
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            issues = []

            # Check model extraction
            if metadata.agent.model.name == "unknown":
                issues.append("Model name not extracted")

            if metadata.agent.model.provider == ModelProvider.UNKNOWN:
                issues.append("Provider not detected")

            # Check capabilities
            if len(metadata.agent.capabilities) == 0:
                issues.append("No capabilities extracted")

            # Check context
            if not metadata.agent.framework:
                issues.append("Framework not detected")

            if issues:
                return TestCaseResult(
                    name="introspection",
                    passed=len(issues) <= 2,  # Allow up to 2 issues
                    message=f"Issues: {', '.join(issues)}",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="introspection",
                passed=True,
                message=f"Model: {metadata.agent.model.name}, "
                f"Caps: {len(metadata.agent.capabilities)}",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="introspection",
                passed=False,
                message="Introspection check failed",
                duration_ms=duration,
                error=str(e),
            )

    def _test_handshake(self, facts: AgentFacts) -> TestCaseResult:
        """Test handshake protocol."""
        start = datetime.now(timezone.utc)
        try:
            # Create a service agent
            service = AgentFacts(name="TestService")
            service.sign()

            # Create challenge
            challenge = service.create_challenge(ttl_seconds=60)

            # Client responds
            response = facts.respond_to_challenge(challenge)

            # Service verifies
            result = service.verify_response(challenge, response)

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not result.valid:
                return TestCaseResult(
                    name="handshake",
                    passed=False,
                    message=f"Handshake failed: {result.errors}",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="handshake",
                passed=True,
                message="Challenge-response verified",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="handshake",
                passed=False,
                message="Handshake protocol failed",
                duration_ms=duration,
                error=str(e),
            )

    def _test_policy(self, facts: AgentFacts) -> TestCaseResult:
        """Test policy evaluation."""
        start = datetime.now(timezone.utc)
        try:
            # Test basic trust policy
            policy = Policy.basic_trust()
            result = policy.evaluate(facts.metadata)

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not result.passed:
                violations = [str(v) for v in result.violations]
                return TestCaseResult(
                    name="policy",
                    passed=False,
                    message=f"Policy violations: {violations[:2]}",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="policy",
                passed=True,
                message="Basic trust policy passed",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="policy",
                passed=False,
                message="Policy evaluation failed",
                duration_ms=duration,
                error=str(e),
            )

    def _test_transparency(self, facts: AgentFacts) -> TestCaseResult:
        """Test transparency log."""
        start = datetime.now(timezone.utc)
        try:
            # Log some evidence
            facts.log_evidence("test_event", {"test": True})

            # Check merkle root exists
            if not facts.merkle_root:
                return TestCaseResult(
                    name="transparency",
                    passed=False,
                    message="No merkle root after logging",
                    duration_ms=0,
                )

            # Re-sign metadata after logging evidence to keep artifacts verifiable
            facts.sign()

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            return TestCaseResult(
                name="transparency",
                passed=True,
                message=f"Merkle root: {facts.merkle_root[:20]}...",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="transparency",
                passed=False,
                message="Transparency log failed",
                duration_ms=duration,
                error=str(e),
            )

    def _test_json_roundtrip(self, facts: AgentFacts) -> TestCaseResult:
        """Test JSON export/import."""
        start = datetime.now(timezone.utc)
        try:
            # Export
            json_str = facts.to_json()

            # Import
            restored = AgentFacts.from_json(json_str)

            # Verify
            result = restored.verify()

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if not result.valid:
                return TestCaseResult(
                    name="json_roundtrip",
                    passed=False,
                    message="Restored agent failed verification",
                    duration_ms=duration,
                )

            if restored.did != facts.did:
                return TestCaseResult(
                    name="json_roundtrip",
                    passed=False,
                    message="DID mismatch after roundtrip",
                    duration_ms=duration,
                )

            return TestCaseResult(
                name="json_roundtrip",
                passed=True,
                message=f"JSON size: {len(json_str)} bytes",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return TestCaseResult(
                name="json_roundtrip",
                passed=False,
                message="JSON roundtrip failed",
                duration_ms=duration,
                error=str(e),
            )

    def _generate_artifacts(self, facts: AgentFacts, name: str, framework: str) -> str:
        """Generate test artifacts."""
        safe_name = name.replace(" ", "_").replace("/", "_")

        # Metadata JSON
        metadata_dir = self.artifacts_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"{framework}_{safe_name}.json"
        metadata_file.write_text(facts.to_json())

        # Badge
        badge_dir = self.artifacts_dir / "badges"
        badge_dir.mkdir(parents=True, exist_ok=True)
        badge_file = badge_dir / f"{framework}_{safe_name}.md"
        badge_content = self._generate_badge(facts)
        badge_file.write_text(badge_content)

        return str(metadata_file)

    def _generate_badge(self, facts: AgentFacts) -> str:
        """Generate a trust badge."""
        status = "verified" if facts.is_signed else "unverified"
        color = "brightgreen" if facts.is_signed else "red"

        badge = f"![AgentFacts {status}](https://img.shields.io/badge/AgentFacts-{status}-{color})\n\n"
        badge += f"**Agent:** {facts.name}\n"
        badge += f"**DID:** `{facts.did}`\n"
        badge += f"**Model:** {facts.metadata.agent.model.name}\n"
        badge += f"**Provider:** {facts.metadata.agent.model.provider.value}\n"
        badge += f"**Capabilities:** {len(facts.metadata.agent.capabilities)} tools\n"

        return badge


# ---------------------------------------------------------------------------
# Agent Discovery
# ---------------------------------------------------------------------------


class AgentDiscovery:
    """Discovers agents in repositories."""

    @staticmethod
    def discover_repo_agents(
        framework: str,
        repos_root: Path,
        stub_deps: bool = False,
    ) -> list[RepoSpec]:
        """Create repo-backed test specs using cloned repositories."""
        framework_dir = repos_root / framework
        if not framework_dir.exists() or not framework_dir.is_dir():
            return []

        repo_dirs = sorted(
            [d for d in framework_dir.iterdir() if d.is_dir() and (d / ".git").exists()]
        )
        if not repo_dirs:
            return []

        loader = RepoAgentLoader(stub_deps=stub_deps)
        repo_specs: list[RepoSpec] = []
        for repo_dir in repo_dirs:
            agents, error, skipped = loader.load_agents(framework, repo_dir)
            repo_specs.append(
                RepoSpec(
                    name=repo_dir.name,
                    url=AgentDiscovery._read_repo_url(repo_dir),
                    path=repo_dir,
                    agents=agents,
                    error=error,
                    skipped=skipped,
                )
            )

        return repo_specs

    @staticmethod
    def _pick_mock_agent(framework: str, index: int) -> tuple[str, Any]:
        """Select a mock agent for a repo slot."""
        agents = AgentDiscovery.discover_mock_agents(framework)
        if not agents:
            raise ValueError(f"No mock agents available for framework '{framework}'")
        return agents[index % len(agents)]

    @staticmethod
    def _read_repo_url(repo_dir: Path) -> str:
        """Extract origin URL from a git clone."""
        config_path = repo_dir / ".git" / "config"
        if not config_path.exists():
            return f"local://{repo_dir}"

        try:
            for line in config_path.read_text().splitlines():
                if line.strip().startswith("url ="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            pass

        return f"local://{repo_dir}"

    @staticmethod
    def discover_mock_agents(framework: str) -> list[tuple[str, Any]]:
        """Create mock agents for testing without real repos."""
        agents = []

        if framework == "langchain":
            agents.extend(AgentDiscovery._create_langchain_mocks())
        elif framework == "crewai":
            agents.extend(AgentDiscovery._create_crewai_mocks())
        elif framework == "autogen":
            agents.extend(AgentDiscovery._create_autogen_mocks())
        elif framework == "llamaindex":
            agents.extend(AgentDiscovery._create_llamaindex_mocks())
        elif framework == "openagents":
            agents.extend(AgentDiscovery._create_openagents_mocks())
        elif framework == "huggingface":
            agents.extend(AgentDiscovery._create_huggingface_mocks())

        return agents

    @staticmethod
    def _create_langchain_mocks() -> list[tuple[str, Any]]:
        """Create mock LangChain agents."""

        class MockLLM:
            model_name = "gpt-4"
            temperature = 0.7
            max_tokens = 2048

        class MockTool:
            def __init__(self, name, desc):
                self.name = name
                self.description = desc
                self.args_schema = None

        class MockAgent:
            __module__ = "langchain.agents"

            def __init__(self):
                self.llm = MockLLM()

        class MockAgentExecutor:
            __module__ = "langchain.agents"

            def __init__(self, name, tools):
                self.agent = MockAgent()
                self.tools = tools
                self.max_iterations = 10

        return [
            (
                "LangChain_Search_Agent",
                MockAgentExecutor("Search", [MockTool("web_search", "Search the web")]),
            ),
            (
                "LangChain_Code_Agent",
                MockAgentExecutor(
                    "Code",
                    [
                        MockTool("python_repl", "Execute Python code"),
                        MockTool("file_reader", "Read files"),
                    ],
                ),
            ),
            (
                "LangChain_Research_Agent",
                MockAgentExecutor(
                    "Research",
                    [
                        MockTool("arxiv_search", "Search arXiv papers"),
                        MockTool("wikipedia", "Search Wikipedia"),
                        MockTool("calculator", "Perform calculations"),
                    ],
                ),
            ),
        ]

    @staticmethod
    def _create_crewai_mocks() -> list[tuple[str, Any]]:
        """Create mock CrewAI agents."""

        class MockTool:
            def __init__(self, name, desc):
                self.name = name
                self.description = desc

        class ChatOpenAI:
            def __init__(self, model_name: str = "gpt-4", temperature: float = 0.2):
                self.model_name = model_name
                self.temperature = temperature
                self.max_tokens = 2048

        class MockCrewAgent:
            __module__ = "crewai"

            def __init__(self, role, goal, backstory, tools=None, llm=None):
                self.role = role
                self.goal = goal
                self.backstory = backstory
                self.tools = tools or []
                self.allow_delegation = True
                self.verbose = True
                self.llm = llm or ChatOpenAI()

        return [
            (
                "CrewAI_Researcher",
                MockCrewAgent(
                    role="Senior Researcher",
                    goal="Find and analyze information",
                    backstory="Expert researcher with 10 years experience",
                    tools=[MockTool("search", "Web search")],
                ),
            ),
            (
                "CrewAI_Writer",
                MockCrewAgent(
                    role="Content Writer",
                    goal="Create engaging content",
                    backstory="Professional writer specializing in tech",
                    tools=[MockTool("write", "Write content")],
                ),
            ),
            (
                "CrewAI_Analyst",
                MockCrewAgent(
                    role="Data Analyst",
                    goal="Analyze data and find insights",
                    backstory="Data scientist with ML expertise",
                    tools=[
                        MockTool("sql_query", "Query databases"),
                        MockTool("chart", "Create charts"),
                    ],
                ),
            ),
        ]

    @staticmethod
    def _create_autogen_mocks() -> list[tuple[str, Any]]:
        """Create mock AutoGen agents."""

        class MockAutoGenAgent:
            __module__ = "autogen"

            def __init__(self, name, system_message, llm_config=None):
                self.name = name
                self.system_message = system_message
                self.llm_config = llm_config or {"model": "gpt-4"}
                self._function_map = {}

        return [
            (
                "AutoGen_Assistant",
                MockAutoGenAgent(
                    name="assistant",
                    system_message="You are a helpful AI assistant.",
                    llm_config={"model": "gpt-4", "temperature": 0.7},
                ),
            ),
            (
                "AutoGen_Coder",
                MockAutoGenAgent(
                    name="coder",
                    system_message="You are an expert programmer.",
                    llm_config={"model": "gpt-4", "temperature": 0},
                ),
            ),
            (
                "AutoGen_Reviewer",
                MockAutoGenAgent(
                    name="reviewer",
                    system_message="You review code for quality and security.",
                    llm_config={"model": "gpt-4-turbo"},
                ),
            ),
        ]

    @staticmethod
    def _create_llamaindex_mocks() -> list[tuple[str, Any]]:
        """Create mock LlamaIndex agents."""

        class MockLLM:
            model = "gpt-4"
            temperature = 0.1

        class MockTool:
            def __init__(self, name, desc):
                self.name = name
                self.description = desc
                self.metadata = type("M", (), {"name": name, "description": desc})()

        class MockQueryEngine:
            __module__ = "llama_index.core.query_engine"

            def __init__(self, name):
                self._llm = MockLLM()
                self._name = name

        class MockAgentRunner:
            __module__ = "llama_index.core.agent"

            def __init__(self, name, tools):
                self.agent_worker = type(
                    "W",
                    (),
                    {
                        "llm": MockLLM(),
                        "tools": tools,
                    },
                )()
                self._name = name

        return [
            ("LlamaIndex_RAG_Engine", MockQueryEngine("RAG")),
            (
                "LlamaIndex_Agent",
                MockAgentRunner(
                    "Research", [MockTool("vector_search", "Search vector store")]
                ),
            ),
            (
                "LlamaIndex_SQL_Agent",
                MockAgentRunner(
                    "SQL",
                    [
                        MockTool("sql_query", "Execute SQL"),
                        MockTool("schema_lookup", "Get table schema"),
                    ],
                ),
            ),
        ]

    @staticmethod
    def _create_openagents_mocks() -> list[tuple[str, Any]]:
        """Create mock OpenAgents."""

        class MockWorkerAgent:
            __module__ = "openagents"

            def __init__(self, name, role):
                self.name = name
                self.role = role
                self.model = "gpt-4"
                self.default_agent_id = name

            def run_agent(self, *args, **kwargs):
                return "ok"

        return [
            ("OpenAgents_Data_Worker", MockWorkerAgent("data_worker", "Data analysis")),
            (
                "OpenAgents_Plugin_Worker",
                MockWorkerAgent("plugin_worker", "Plugin execution"),
            ),
            ("OpenAgents_Web_Worker", MockWorkerAgent("web_worker", "Web browsing")),
        ]

    @staticmethod
    def _create_huggingface_mocks() -> list[tuple[str, Any]]:
        """Create mock HuggingFace agents."""

        class MockTool:
            def __init__(self, name, desc):
                self.name = name
                self.description = desc

        class MockCodeAgent:
            __module__ = "smolagents"

            def __init__(self, tools, model):
                self.tools = {t.name: t for t in tools}
                self.model = model
                self.max_steps = 10

        class MockToolCallingAgent:
            __module__ = "smolagents"

            def __init__(self, tools, model):
                self.tools = {t.name: t for t in tools}
                self.model = model

        return [
            (
                "HuggingFace_Code_Agent",
                MockCodeAgent(
                    tools=[
                        MockTool("python_interpreter", "Execute Python"),
                        MockTool("web_surfer", "Browse web"),
                    ],
                    model="Qwen/Qwen2.5-Coder-32B-Instruct",
                ),
            ),
            (
                "HuggingFace_Tool_Agent",
                MockToolCallingAgent(
                    tools=[
                        MockTool("calculator", "Math operations"),
                        MockTool("search", "Web search"),
                    ],
                    model="meta-llama/Llama-3.1-70B-Instruct",
                ),
            ),
        ]


class RepoAgentLoader:
    """Loads real agents from cloned repositories."""

    def __init__(self, stub_deps: bool = False) -> None:
        self._stub_deps = stub_deps

    def load_agents(
        self,
        framework: str,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        """Load agents for a specific framework from a repo."""
        try:
            if framework == "langchain":
                return self._load_langchain(repo_path)
            if framework == "crewai":
                return self._load_crewai(repo_path)
            if framework == "autogen":
                return self._load_autogen(repo_path)
            if framework == "llamaindex":
                return self._load_llamaindex(repo_path)
            if framework == "openagents":
                return self._load_openagents(repo_path)
            if framework == "huggingface":
                return self._load_huggingface(repo_path)
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            return [], f"Dependency required: {missing}", True
        except PermissionError as exc:
            return [], f"Dependency required: {exc}", True
        except ImportError as exc:
            return [], f"Dependency required: {exc}", True
        except Exception as exc:
            return [], f"Repo loader error: {exc}", False

        return [], f"No repo loader available for framework '{framework}'", False

    @staticmethod
    @contextlib.contextmanager
    def _temporary_sys_path(paths: list[Path]):
        original = list(sys.path)
        for path in paths:
            sys.path.insert(0, str(path))
        try:
            yield
        finally:
            sys.path = original

    @staticmethod
    def _purge_modules(prefixes: list[str]) -> None:
        for module_name in list(sys.modules.keys()):
            if any(
                module_name == prefix or module_name.startswith(f"{prefix}.")
                for prefix in prefixes
            ):
                sys.modules.pop(module_name, None)

    def _install_stub(self, module_name: str) -> None:
        if module_name in sys.modules:
            return

        stub = _StubModule(module_name)
        sys.modules[module_name] = stub

    def _import_with_stubs(
        self, module_name: str, max_attempts: int = 20
    ) -> types.ModuleType:
        for _ in range(max_attempts):
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if not self._stub_deps:
                    raise
                missing = exc.name
                if not missing:
                    raise
                self._install_stub(missing)
        raise RuntimeError(f"Exceeded stub limit importing {module_name}")

    def _try_import(self, module_name: str) -> types.ModuleType | None:
        root_name = module_name.split(".", 1)[0]
        try:
            return self._import_with_stubs(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith(root_name):
                return None
            raise

    @staticmethod
    def _find_package_roots(
        repo_path: Path, package_names: list[str], max_depth: int = 5
    ) -> list[Path]:
        roots: set[Path] = set()
        for root, dirs, _files in os.walk(repo_path):
            rel_depth = len(Path(root).relative_to(repo_path).parts)
            if rel_depth > max_depth:
                dirs[:] = []
                continue
            for name in package_names:
                if name in dirs:
                    roots.add(Path(root))
        return sorted(roots)

    def _load_langchain(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(
            repo_path,
            ["langchain_core", "langchain_classic"],
        )
        if not package_roots:
            return (
                [],
                "Package not found in repo: langchain_core/langchain_classic",
                True,
            )

        self._purge_modules(["langchain_core", "langchain_classic", "langchain"])
        with self._temporary_sys_path(package_roots):
            self._import_with_stubs("langchain_core")
            fake_list_llm_cls = importlib.import_module(
                "langchain_core.language_models.fake"
            ).FakeListLLM
            runnables = importlib.import_module("langchain_core.runnables")
            runnable_lambda = runnables.RunnableLambda
            runnable_sequence = runnables.RunnableSequence

            llm = fake_list_llm_cls(responses=["ok"])
            runnable = runnable_sequence(
                first=llm,
                last=runnable_lambda(lambda _x: "ok"),
            )

            return [("LangChain_RunnableSequence", runnable)], None, False

    def _load_crewai(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(repo_path, ["crewai"])
        if not package_roots:
            return [], "Package not found in repo: crewai", True

        self._purge_modules(["crewai"])
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("CREWAI_TELEMETRY_DISABLED", "true")
        if self._stub_deps:
            data_dir = repo_path / ".crewai_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            sys.modules["appdirs"] = _AppDirsStub(str(data_dir))

        with self._temporary_sys_path(package_roots):
            crewai = self._import_with_stubs("crewai")
            agent_cls = crewai.Agent

            agent = agent_cls(
                role="Researcher",
                goal="Find relevant information",
                backstory="Repo-based CrewAI agent",
                tools=[],
                llm=None,
            )
            return [("CrewAI_Agent", agent)], None, False

    def _load_autogen(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(
            repo_path,
            ["autogen_agentchat", "autogen_core", "autogen", "pyautogen"],
        )
        if not package_roots:
            return (
                [],
                "Package not found in repo: autogen_agentchat/autogen/pyautogen",
                True,
            )

        self._purge_modules(
            ["autogen_agentchat", "autogen_core", "autogen_ext", "autogen", "pyautogen"]
        )
        with self._temporary_sys_path(package_roots):
            assistant_cls = None
            candidates = [
                ("autogen_agentchat.agents", "AssistantAgent"),
                ("autogen_agentchat", "AssistantAgent"),
                ("autogen.agentchat.agents", "AssistantAgent"),
                ("autogen.agentchat", "AssistantAgent"),
                ("pyautogen.agentchat.agents", "AssistantAgent"),
                ("pyautogen.agentchat", "AssistantAgent"),
            ]
            for module_name, attr in candidates:
                module = self._try_import(module_name)
                if module and hasattr(module, attr):
                    assistant_cls = getattr(module, attr)
                    break

            if assistant_cls is None:
                return (
                    [],
                    "Package not found in repo: autogen_agentchat/agentchat",
                    True,
                )

            agent = assistant_cls(
                name="assistant",
                system_message="Repo-based AutoGen assistant",
                llm_config={"config_list": [{"model": "gpt-4"}], "temperature": 0},
            )
            return [("AutoGen_Assistant", agent)], None, False

    def _load_llamaindex(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(repo_path, ["llama_index"])
        if not package_roots:
            return [], "Package not found in repo: llama_index", True

        self._purge_modules(["llama_index"])
        with self._temporary_sys_path(package_roots):
            agent_runner_cls = importlib.import_module(
                "llama_index.core.agent"
            ).AgentRunner
            mock_llm_cls = importlib.import_module("llama_index.core.llms").MockLLM

            llm = mock_llm_cls()
            agent = agent_runner_cls(llm=llm, tools=[])
            return [("LlamaIndex_AgentRunner", agent)], None, False

    def _load_openagents(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(repo_path, ["openagents"])
        if not package_roots:
            return [], "Package not found in repo: openagents", True

        self._purge_modules(["openagents"])
        with self._temporary_sys_path(package_roots):
            worker_agent_cls = importlib.import_module("openagents").WorkerAgent

            agent = worker_agent_cls(
                name="worker", role="Repo-based worker", model="gpt-4"
            )
            return [("OpenAgents_Worker", agent)], None, False

    def _load_huggingface(
        self,
        repo_path: Path,
    ) -> tuple[list[tuple[str, Any]], str | None, bool]:
        package_roots = self._find_package_roots(repo_path, ["smolagents"])
        if not package_roots:
            return [], "Package not found in repo: smolagents", True

        self._purge_modules(["smolagents"])
        with self._temporary_sys_path(package_roots):
            smolagents = importlib.import_module("smolagents")
            tools_module = importlib.import_module("smolagents.tools")
            models_module = importlib.import_module("smolagents.models")

            code_agent_cls = smolagents.CodeAgent
            tool_cls = tools_module.Tool
            model_cls = models_module.Model
            chat_message_cls = models_module.ChatMessage
            message_role_cls = models_module.MessageRole

            class DemoTool(tool_cls):
                name = "calculator"
                description = "Return a canned result for repo-based testing."
                inputs = {
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate.",
                    }
                }
                output_type = "string"

                def forward(self, expression: str) -> str:
                    return "ok"

            class DummyModel(model_cls):
                def generate(self, messages, **_kwargs):
                    return chat_message_cls(
                        role=message_role_cls.ASSISTANT, content="ok"
                    )

            tool = DemoTool()
            agent = code_agent_cls(tools=[tool], model=DummyModel(model_id="repo"))
            return [("HuggingFace_CodeAgent", agent)], None, False


class _StubModule(types.ModuleType):
    """Placeholder module used to stub missing optional dependencies."""

    def __getattr__(self, name: str) -> Any:  # noqa: D401
        full_name = f"{self.__name__}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        stub = _StubModule(full_name)
        sys.modules[full_name] = stub
        return stub

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, _key: Any) -> Any:
        return None


class _AppDirsStub(types.ModuleType):
    """Minimal appdirs stub for CrewAI storage paths."""

    def __init__(self, data_dir: str) -> None:
        super().__init__("appdirs")
        self._data_dir = data_dir

    def user_data_dir(self, *_args: Any, **_kwargs: Any) -> str:
        return self._data_dir


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def run_framework_tests(
    framework: str,
    artifacts_dir: Path,
    use_mocks: bool = True,
    repos_dir: Path | None = None,
    stub_deps: bool = False,
) -> FrameworkTestResult:
    """Run tests for a framework."""
    result = FrameworkTestResult(framework=framework)
    runner = TestRunner(artifacts_dir)

    # Get agents
    if use_mocks:
        repo_specs = [
            RepoSpec(
                name=f"mock_{framework}",
                url="mock://",
                path=Path(""),
                agents=AgentDiscovery.discover_mock_agents(framework),
            )
        ]
    else:
        if repos_dir is None:
            raise ValueError("repos_dir must be provided when use_mocks=False")
        repo_specs = AgentDiscovery.discover_repo_agents(
            framework,
            repos_dir,
            stub_deps=stub_deps,
        )
        if not repo_specs:
            repo_result = RepoTestResult(
                repo_name=f"repos_{framework}",
                repo_url=f"local://{repos_dir}",
                framework=framework,
                error=f"No repositories found under {repos_dir / framework}",
            )
            result.repos.append(repo_result)
            return result

    for repo_spec in repo_specs:
        if repo_spec.error:
            status = "SKIP" if repo_spec.skipped else "ERROR"
            print(f"  [{status}] {repo_spec.name}: {repo_spec.error}")
            repo_result = RepoTestResult(
                repo_name=repo_spec.name,
                repo_url=repo_spec.url,
                framework=framework,
                error=repo_spec.error,
                skipped=repo_spec.skipped,
            )
            result.repos.append(repo_result)
            continue

        repo_result = RepoTestResult(
            repo_name=repo_spec.name,
            repo_url=repo_spec.url,
            framework=framework,
        )

        for name, agent in repo_spec.agents:
            context = None
            if not use_mocks:
                context = {
                    "source_repo": repo_spec.url,
                    "source_repo_name": repo_spec.name,
                    "source_repo_path": str(repo_spec.path),
                }

            print(f"  Testing: {name}")
            agent_result = runner.run_all_tests(agent, name, framework, context=context)
            repo_result.agents.append(agent_result)

            # Print results
            for test in agent_result.tests:
                status = "PASS" if test.passed else "FAIL"
                print(f"    [{status}] {test.name}: {test.message}")

        result.repos.append(repo_result)

    return result


def generate_summary_report(results: dict[str, FrameworkTestResult], output_dir: Path):
    """Generate summary report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frameworks": {},
        "totals": {
            "total_frameworks": len(results),
            "total_agents": 0,
            "passed_agents": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "skipped_repos": 0,
        },
    }

    for framework, result in results.items():
        framework_data = {
            "repos_tested": len(result.repos),
            "agents_tested": result.total_agents,
            "agents_passed": result.passed_agents,
            "pass_rate": result.passed_agents / max(result.total_agents, 1),
            "repos_skipped": result.skipped_repos,
            "tests": {},
        }

        # Aggregate test results
        for repo in result.repos:
            for agent in repo.agents:
                for test in agent.tests:
                    if test.name not in framework_data["tests"]:
                        framework_data["tests"][test.name] = {"passed": 0, "failed": 0}

                    if test.passed:
                        framework_data["tests"][test.name]["passed"] += 1
                        report["totals"]["passed_tests"] += 1
                    else:
                        framework_data["tests"][test.name]["failed"] += 1

                    report["totals"]["total_tests"] += 1

        report["frameworks"][framework] = framework_data
        report["totals"]["total_agents"] += result.total_agents
        report["totals"]["passed_agents"] += result.passed_agents
        report["totals"]["skipped_repos"] += result.skipped_repos

    # Write report
    report_file = output_dir / "test_summary.json"
    report_file.write_text(json.dumps(report, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total frameworks: {report['totals']['total_frameworks']}")
    print(f"Total agents: {report['totals']['total_agents']}")
    print(f"Passed agents: {report['totals']['passed_agents']}")
    print(f"Total tests: {report['totals']['total_tests']}")
    print(f"Passed tests: {report['totals']['passed_tests']}")
    print(f"Skipped repos: {report['totals']['skipped_repos']}")
    pass_rate = report["totals"]["passed_tests"] / max(
        report["totals"]["total_tests"], 1
    )
    print(f"Overall pass rate: {pass_rate:.1%}")

    return report


def _clone_repos(script_path: Path, framework: str | None) -> None:
    """Clone or update repositories for integration tests."""
    if not script_path.exists():
        raise FileNotFoundError(f"Clone script not found: {script_path}")

    cmd = ["bash", str(script_path)]
    if framework:
        cmd.append(framework)
    subprocess.run(cmd, check=False)


def _repos_present(repos_dir: Path, frameworks: list[str]) -> bool:
    """Check if repos exist for the selected frameworks."""
    for framework in frameworks:
        framework_dir = repos_dir / framework
        if framework_dir.exists() and any(d.is_dir() for d in framework_dir.iterdir()):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="AgentFacts Integration Test Runner")
    parser.add_argument("--framework", "-f", help="Test specific framework")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available frameworks"
    )
    parser.add_argument(
        "--use-repos",
        action="store_true",
        help="Use cloned repositories instead of mocks",
    )
    parser.add_argument(
        "--clone-repos",
        action="store_true",
        help="Clone/update repositories before running tests",
    )
    parser.add_argument(
        "--repos-dir",
        help="Path to cloned repositories (default: tests/integration/repos)",
    )
    parser.add_argument(
        "--stub-deps",
        action="store_true",
        help="Stub missing optional dependencies when importing repo packages",
    )
    args = parser.parse_args()

    frameworks = [
        "langchain",
        "crewai",
        "autogen",
        "llamaindex",
        "openagents",
        "huggingface",
    ]

    if args.list:
        print("Available frameworks:")
        for fw in frameworks:
            print(f"  - {fw}")
        return

    # Setup directories
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir.parent / "artifacts"
    results_dir = script_dir.parent / "results"
    repos_dir = Path(args.repos_dir) if args.repos_dir else script_dir.parent / "repos"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    results = {}
    frameworks_to_test = [args.framework] if args.framework else frameworks

    if args.use_repos and (
        args.clone_repos or not _repos_present(repos_dir, frameworks_to_test)
    ):
        _clone_repos(script_dir / "clone_repos.sh", args.framework)

    for framework in frameworks_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing {framework.upper()}")
        print("=" * 60)

        results[framework] = run_framework_tests(
            framework,
            artifacts_dir,
            use_mocks=not args.use_repos,
            repos_dir=repos_dir,
            stub_deps=args.stub_deps,
        )

    # Generate report
    generate_summary_report(results, results_dir)

    print(f"\nArtifacts saved to: {artifacts_dir}")
    print(f"Report saved to: {results_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
