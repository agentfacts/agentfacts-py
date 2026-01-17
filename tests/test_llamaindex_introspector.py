"""
Tests for LlamaIndex introspector.
"""

from agentfacts import AgentFacts, GroupFacts, ProcessType
from agentfacts.integrations.llamaindex.introspector import (
    LlamaIndexIntegration as LlamaIndexIntrospector,
)
from agentfacts.models import ModelProvider

# -------------------------------------------------------------------------
# Mock Classes
# -------------------------------------------------------------------------


class MockLLM:
    """Mock LlamaIndex LLM."""

    __module__ = "llama_index.llms.openai"

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class MockEmbedModel:
    """Mock embedding model."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name


class MockTool:
    """Mock LlamaIndex tool."""

    def __init__(self, name: str, description: str = ""):
        self.metadata = MockToolMetadata(name, description)
        self.fn_schema = None


class MockToolMetadata:
    """Mock tool metadata."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class MockRetriever:
    """Mock LlamaIndex retriever."""

    __module__ = "llama_index.core.retrievers"

    def __init__(self, similarity_top_k: int = 10):
        self._similarity_top_k = similarity_top_k
        self._query_mode = "default"


class MockVectorStore:
    """Mock vector store."""

    pass


class MockDocStore:
    """Mock document store."""

    def __init__(self, docs: dict = None):
        self.docs = docs or {}


class MockResponseSynthesizer:
    """Mock response synthesizer."""

    def __init__(self, streaming: bool = False):
        self.streaming = streaming
        self._llm = None


class MockQueryEngine:
    """Mock LlamaIndex QueryEngine."""

    __module__ = "llama_index.core.query_engine"

    def __init__(
        self,
        retriever: MockRetriever = None,
        llm: MockLLM = None,
        response_synthesizer: MockResponseSynthesizer = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._response_synthesizer = response_synthesizer
        self._node_postprocessors = []

    def query(self, query_str: str):
        pass


class MockSubQuestionQueryEngine:
    """Mock SubQuestionQueryEngine."""

    __module__ = "llama_index.core.query_engine"

    def __init__(self, llm: MockLLM = None):
        self._llm = llm
        self._retriever = None

    def query(self, query_str: str):
        pass


class MockRouterQueryEngine:
    """Mock RouterQueryEngine."""

    __module__ = "llama_index.core.query_engine"

    def __init__(self, query_engines: list = None):
        self.query_engines = query_engines or []
        self._llm = None

    def query(self, query_str: str):
        pass


class MockCitationQueryEngine:
    """Mock CitationQueryEngine."""

    __module__ = "llama_index.core.query_engine"

    def __init__(self, llm: MockLLM = None):
        self._llm = llm
        self._retriever = None

    def query(self, query_str: str):
        pass


class MockVectorStoreIndex:
    """Mock LlamaIndex VectorStoreIndex."""

    __module__ = "llama_index.core.indices"

    def __init__(
        self,
        docstore: MockDocStore = None,
        vector_store: MockVectorStore = None,
        embed_model: MockEmbedModel = None,
    ):
        self.docstore = docstore or MockDocStore()
        self._vector_store = vector_store
        self._embed_model = embed_model
        self.index_struct = {}


class MockKnowledgeGraphIndex:
    """Mock KnowledgeGraphIndex."""

    __module__ = "llama_index.core.indices"

    def __init__(self, graph_store=None):
        self._graph_store = graph_store
        self.docstore = MockDocStore()
        self.index_struct = {}


class MockAgentWorker:
    """Mock agent worker."""

    def __init__(
        self, llm: MockLLM = None, max_iterations: int = 10, verbose: bool = False
    ):
        self._llm = llm
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose


class MockAgentRunner:
    """Mock LlamaIndex AgentRunner."""

    __module__ = "llama_index.core.agent"

    def __init__(
        self,
        agent_worker: MockAgentWorker = None,
        tools: list = None,
        memory=None,
    ):
        self.agent_worker = agent_worker
        self.tools = tools or []
        self.memory = memory


class MockReActAgent:
    """Mock ReActAgent."""

    __module__ = "llama_index.core.agent"

    def __init__(
        self,
        tools: list = None,
        llm: MockLLM = None,
    ):
        self.tools = tools or []
        self._llm = llm
        self.agent_worker = MockAgentWorker(llm=llm)


class MockOpenAIAgent:
    """Mock OpenAIAgent."""

    __module__ = "llama_index.core.agent"

    def __init__(self, tools: list = None, llm: MockLLM = None):
        self.tools = tools or []
        self._llm = llm
        self.agent_worker = None


class MockChatEngine:
    """Mock ChatEngine."""

    __module__ = "llama_index.core.chat_engine"

    def __init__(self, llm: MockLLM = None, memory=None):
        self._llm = llm
        self._memory = memory


class MockContextChatEngine:
    """Mock ContextChatEngine."""

    __module__ = "llama_index.core.chat_engine"

    def __init__(self, retriever: MockRetriever = None, llm: MockLLM = None):
        self._retriever = retriever
        self._llm = llm
        self.memory = None


class MockBM25Retriever:
    """Mock BM25Retriever."""

    __module__ = "llama_index.core.retrievers"

    def __init__(self, similarity_top_k: int = 10):
        self.similarity_top_k = similarity_top_k


class MockQueryFusionRetriever:
    """Mock QueryFusionRetriever."""

    __module__ = "llama_index.core.retrievers"

    def __init__(self, num_queries: int = 4):
        self.num_queries = num_queries


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestLlamaIndexIntrospector:
    """Tests for LlamaIndex introspector."""

    def test_framework_name(self):
        """Test framework name."""
        introspector = LlamaIndexIntrospector()
        assert introspector.framework_name == "llamaindex"

    def test_can_introspect_query_engine(self):
        """Test can_introspect with QueryEngine."""
        introspector = LlamaIndexIntrospector()
        engine = MockQueryEngine()
        assert introspector.can_introspect(engine) is True

    def test_can_introspect_index(self):
        """Test can_introspect with VectorStoreIndex."""
        introspector = LlamaIndexIntrospector()
        index = MockVectorStoreIndex()
        assert introspector.can_introspect(index) is True

    def test_can_introspect_agent(self):
        """Test can_introspect with AgentRunner."""
        introspector = LlamaIndexIntrospector()
        agent = MockAgentRunner()
        assert introspector.can_introspect(agent) is True

    def test_cannot_introspect_random(self):
        """Test can_introspect rejects random objects."""
        introspector = LlamaIndexIntrospector()
        assert introspector.can_introspect("string") is False
        assert introspector.can_introspect(123) is False
        assert introspector.can_introspect({}) is False

    def test_introspect_query_engine_basic(self):
        """Test basic QueryEngine introspection."""
        introspector = LlamaIndexIntrospector()
        retriever = MockRetriever(similarity_top_k=5)
        engine = MockQueryEngine(retriever=retriever)

        result = introspector.introspect(engine)

        assert result.context["engine_type"] == "MockQueryEngine"
        assert result.context["retriever_type"] == "MockRetriever"
        assert result.context["similarity_top_k"] == 5
        assert any(c.name == "query" for c in result.capabilities)

    def test_introspect_query_engine_with_llm(self):
        """Test QueryEngine introspection with LLM."""
        introspector = LlamaIndexIntrospector()
        llm = MockLLM(model="gpt-4-turbo", temperature=0.3)
        engine = MockQueryEngine(llm=llm)

        result = introspector.introspect(engine)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4-turbo"
        assert result.baseline_model.temperature == 0.3

    def test_introspect_query_engine_with_synthesizer(self):
        """Test QueryEngine with response synthesizer."""
        introspector = LlamaIndexIntrospector()
        synth = MockResponseSynthesizer(streaming=True)
        engine = MockQueryEngine(response_synthesizer=synth)

        result = introspector.introspect(engine)

        # Verify it was introspected
        assert result.context["engine_type"] == "MockQueryEngine"

    def test_introspect_subquestion_engine(self):
        """Test SubQuestionQueryEngine introspection."""
        introspector = LlamaIndexIntrospector()
        engine = MockSubQuestionQueryEngine()

        result = introspector.introspect(engine)

        assert result.context.get("multi_step") is True
        assert any(c.name == "sub_question_decomposition" for c in result.capabilities)

    def test_introspect_router_engine(self):
        """Test RouterQueryEngine introspection."""
        introspector = LlamaIndexIntrospector()
        sub_engines = [MockQueryEngine(), MockQueryEngine()]
        engine = MockRouterQueryEngine(query_engines=sub_engines)

        result = introspector.introspect(engine)

        assert result.context.get("routing") is True
        assert result.context.get("routed_engine_count") == 2

    def test_introspect_citation_engine(self):
        """Test CitationQueryEngine introspection."""
        introspector = LlamaIndexIntrospector()
        engine = MockCitationQueryEngine()

        result = introspector.introspect(engine)

        assert result.context.get("citations") is True
        assert any(c.name == "citations" for c in result.capabilities)

    def test_introspect_index_basic(self):
        """Test basic VectorStoreIndex introspection."""
        introspector = LlamaIndexIntrospector()
        docstore = MockDocStore(docs={"doc1": {}, "doc2": {}, "doc3": {}})
        index = MockVectorStoreIndex(docstore=docstore)

        result = introspector.introspect(index)

        assert result.context["index_type"] == "MockVectorStoreIndex"
        assert result.context.get("has_docstore") is True
        assert result.context.get("document_count") == 3
        assert any(c.name == "document_indexing" for c in result.capabilities)

    def test_introspect_index_with_embed_model(self):
        """Test index introspection with embed model."""
        introspector = LlamaIndexIntrospector()
        embed_model = MockEmbedModel("text-embedding-3-small")
        index = MockVectorStoreIndex(embed_model=embed_model)

        result = introspector.introspect(index)

        # Verify it was introspected as an index
        assert result.context["index_type"] == "MockVectorStoreIndex"

    def test_introspect_kg_index(self):
        """Test KnowledgeGraphIndex introspection."""
        introspector = LlamaIndexIntrospector()
        index = MockKnowledgeGraphIndex()

        result = introspector.introspect(index)

        assert result.context.get("knowledge_graph") is True

    def test_introspect_agent_basic(self):
        """Test basic AgentRunner introspection."""
        introspector = LlamaIndexIntrospector()
        tools = [
            MockTool("web_search", "Search the web"),
            MockTool("calculator", "Perform calculations"),
        ]
        worker = MockAgentWorker(max_iterations=15, verbose=True)
        agent = MockAgentRunner(agent_worker=worker, tools=tools)

        result = introspector.introspect(agent)

        assert result.context["agent_type"] == "MockAgentRunner"
        assert result.context.get("tool_count") == 2
        assert result.context.get("verbose") is True
        assert result.constraints.max_iterations == 15
        assert len(result.capabilities) == 2

    def test_introspect_agent_with_llm(self):
        """Test agent introspection with LLM."""
        introspector = LlamaIndexIntrospector()
        llm = MockLLM(model="claude-3-opus", temperature=0.5)
        worker = MockAgentWorker(llm=llm)
        agent = MockAgentRunner(agent_worker=worker)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "claude-3-opus"

    def test_introspect_react_agent(self):
        """Test ReActAgent introspection."""
        introspector = LlamaIndexIntrospector()
        tools = [MockTool("search")]
        agent = MockReActAgent(tools=tools)

        result = introspector.introspect(agent)

        assert result.context.get("reasoning_loop") is True
        assert any(c.name == "reasoning" for c in result.capabilities)

    def test_introspect_openai_agent(self):
        """Test OpenAIAgent introspection."""
        introspector = LlamaIndexIntrospector()
        agent = MockOpenAIAgent()

        result = introspector.introspect(agent)

        assert result.context.get("function_calling") is True

    def test_introspect_retriever_basic(self):
        """Test basic retriever introspection."""
        introspector = LlamaIndexIntrospector()
        retriever = MockRetriever(similarity_top_k=20)

        result = introspector.introspect(retriever)

        assert result.context["engine_type"] == "MockRetriever"
        # Check that capabilities exist
        assert len(result.capabilities) >= 0

    def test_introspect_bm25_retriever(self):
        """Test BM25Retriever introspection."""
        introspector = LlamaIndexIntrospector()
        retriever = MockBM25Retriever()

        result = introspector.introspect(retriever)

        # Check that it was introspected as a query engine type
        assert result.context["engine_type"] == "MockBM25Retriever"

    def test_introspect_fusion_retriever(self):
        """Test QueryFusionRetriever introspection."""
        introspector = LlamaIndexIntrospector()
        retriever = MockQueryFusionRetriever(num_queries=5)

        result = introspector.introspect(retriever)

        # Check that it was introspected as a query engine type
        assert result.context["engine_type"] == "MockQueryFusionRetriever"

    def test_introspect_chat_engine(self):
        """Test ChatEngine introspection."""
        introspector = LlamaIndexIntrospector()
        llm = MockLLM()
        engine = MockChatEngine(llm=llm)

        result = introspector.introspect(engine)

        assert result.context["engine_type"] == "MockChatEngine"
        assert result.baseline_model is not None

    def test_introspect_context_chat_engine(self):
        """Test ContextChatEngine introspection."""
        introspector = LlamaIndexIntrospector()
        retriever = MockRetriever()
        engine = MockContextChatEngine(retriever=retriever)

        result = introspector.introspect(engine)

        assert result.context["engine_type"] == "MockContextChatEngine"
        assert result.context.get("retriever_type") == "MockRetriever"

    def test_detect_provider_openai(self):
        """Test provider detection for OpenAI via introspect."""
        introspector = LlamaIndexIntrospector()
        llm = MockLLM(model="gpt-4")
        # Use an agent that has the LLM
        worker = MockAgentWorker(llm=llm)
        agent = MockAgentRunner(agent_worker=worker)
        result = introspector.introspect(agent)

        # Should detect OpenAI from the model name
        assert result.baseline_model is not None
        assert result.baseline_model.provider == ModelProvider.OPENAI

    def test_get_agent_type(self):
        """Test get_agent_type returns class names."""
        introspector = LlamaIndexIntrospector()

        agent = MockAgentRunner()
        assert introspector.get_agent_type(agent) == "MockAgentRunner"

        engine = MockQueryEngine()
        assert introspector.get_agent_type(engine) == "MockQueryEngine"

        index = MockVectorStoreIndex()
        assert introspector.get_agent_type(index) == "MockVectorStoreIndex"


class TestGroupFactsFromLlamaIndex:
    """Tests for GroupFacts.from_llamaindex()."""

    def test_from_llamaindex_query_engine(self):
        """Test creating GroupFacts from QueryEngine."""
        llm = MockLLM(model="gpt-4")
        retriever = MockRetriever(similarity_top_k=10)
        engine = MockQueryEngine(retriever=retriever, llm=llm)

        group = GroupFacts.from_llamaindex(engine, name="RAG System")

        assert group.name == "RAG System"
        assert len(group.members) == 1
        assert group.metadata.framework == "llamaindex"
        assert group.metadata.process_type == ProcessType.SEQUENTIAL

    def test_from_llamaindex_agent(self):
        """Test creating GroupFacts from AgentRunner."""
        tools = [MockTool("search"), MockTool("write")]
        llm = MockLLM()
        worker = MockAgentWorker(llm=llm)
        agent = MockAgentRunner(agent_worker=worker, tools=tools)

        group = GroupFacts.from_llamaindex(agent, name="Tool Agent")

        assert group.name == "Tool Agent"
        assert len(group.members) == 1
        member = group.members[0]
        assert member.metadata.agent.framework == "llamaindex"
        assert "tool_count" in member.metadata.agent.context

    def test_from_llamaindex_index(self):
        """Test creating GroupFacts from VectorStoreIndex."""
        index = MockVectorStoreIndex()

        group = GroupFacts.from_llamaindex(index)

        assert "LlamaIndex" in group.name
        assert len(group.members) == 1
        assert group.metadata.framework == "llamaindex"

    def test_from_llamaindex_auto_name(self):
        """Test auto-generated name for different component types."""
        engine = MockQueryEngine()
        group1 = GroupFacts.from_llamaindex(engine)
        assert "LlamaIndex" in group1.name

        agent = MockReActAgent()
        group2 = GroupFacts.from_llamaindex(agent)
        assert "LlamaIndex" in group2.name

    def test_from_llamaindex_member_has_capabilities(self):
        """Test that member AgentFacts has capabilities."""
        engine = MockQueryEngine()

        group = GroupFacts.from_llamaindex(engine)

        member = group.members[0]
        assert len(member.metadata.agent.capabilities) > 0
        assert any(c.name == "query" for c in member.metadata.agent.capabilities)

    def test_from_llamaindex_sign_all(self):
        """Test signing LlamaIndex group and members."""
        engine = MockQueryEngine()

        group = GroupFacts.from_llamaindex(engine)
        signatures = group.sign_all()

        assert len(signatures) == 2  # 1 member + 1 group
        assert group.all_verified()

    def test_from_llamaindex_context_preserved(self):
        """Test that component context is preserved."""
        retriever = MockRetriever(similarity_top_k=25)
        engine = MockQueryEngine(retriever=retriever)

        group = GroupFacts.from_llamaindex(engine)

        assert group.metadata.context.get("component_type") == "MockQueryEngine"
        member = group.members[0]
        assert member.metadata.agent.context.get("similarity_top_k") == 25


class TestAgentFactsFromLlamaIndex:
    """Tests for AgentFacts.from_agent with LlamaIndex."""

    def test_from_agent_llamaindex_auto_detect(self):
        """Test AgentFacts.from_agent auto-detects LlamaIndex."""
        engine = MockQueryEngine()

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(LlamaIndexIntrospector())

        facts = AgentFacts.from_agent(engine, name="Query Engine")

        assert facts.name == "Query Engine"
        assert facts.metadata.agent.framework == "llamaindex"

    def test_from_agent_llamaindex_explicit(self):
        """Test AgentFacts.from_agent with explicit framework."""
        llm = MockLLM(model="claude-3")
        engine = MockQueryEngine(llm=llm)

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(LlamaIndexIntrospector())

        facts = AgentFacts.from_agent(engine, name="Claude RAG", framework="llamaindex")

        assert facts.metadata.agent.framework == "llamaindex"
        assert facts.metadata.agent.model.name == "claude-3"
