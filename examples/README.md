# AgentFacts SDK Examples

This folder contains practical examples demonstrating how to use the AgentFacts SDK.

## Quick Start

```bash
# Install the SDK with all dependencies
pip install -e ".[all]"

# Run any example
python examples/01_basic_usage.py
```

## Examples Overview

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [01_basic_usage.py](01_basic_usage.py) | Getting started with AgentFacts | Create, sign, verify, export |
| [02_langchain_integration.py](02_langchain_integration.py) | LangChain agent introspection | `from_langchain()`, callbacks |
| [03_verified_handshake.py](03_verified_handshake.py) | Agent-to-agent authentication | Challenge-response protocol |
| [04_policy_verification.py](04_policy_verification.py) | Zero Trust policies | Rules, policies, engine |
| [05_transparency_log.py](05_transparency_log.py) | Evidence logging & Merkle trees | Audit trails, proofs |
| [06_fastapi_middleware.py](06_fastapi_middleware.py) | HTTP API protection | Middleware, headers |
| [07_cli_usage.py](07_cli_usage.py) | Command-line interface | Keys, badges, verify |
| [08_complete_workflow.py](08_complete_workflow.py) | End-to-end enterprise scenario | Full integration |
| [09_full_verify_middleware.py](09_full_verify_middleware.py) | Full verification middleware | Metadata provider, policy |
| [10_logging_and_audit.py](10_logging_and_audit.py) | Structured logging and audit trail | JSON logs, evidence export |
| [11_custom_metadata_provider.py](11_custom_metadata_provider.py) | Custom metadata provider | Resolve by DID, storage |

## Example Details

### 01: Basic Usage
Learn the fundamentals:
- Create an agent identity with DID
- Add model and capability metadata
- Sign metadata cryptographically
- Verify signatures
- Export/import JSON

```python
from agentfacts import AgentFacts

facts = AgentFacts(name="My Agent")
facts.sign()
print(facts.to_json())
```

### 02: LangChain Integration
Automatic metadata extraction from LangChain:
- Introspect agents, chains, and tools
- Use callback handlers for runtime logging
- Works with AgentExecutor, LCEL, and more

```python
facts = AgentFacts.from_langchain(executor, name="Research Agent")
```

### 03: Verified Handshake
Secure agent-to-agent authentication:
- Challenge-response protocol
- Nonce signing for replay protection
- Mutual authentication

```python
challenge = service.create_challenge()
response = client.respond_to_challenge(challenge)
result = service.verify_response(challenge, response)
```

### 04: Policy Verification
Zero Trust access control:
- Built-in rules (RequireSignature, RequireProvider, etc.)
- Custom policy composition
- Enterprise security policies

```python
policy = Policy(rules=[
    RequireSignature(),
    RequireProvider([ModelProvider.OPENAI]),
    RequireAttestation("security_audit"),
])
result = policy.evaluate(agent.metadata)
```

### 05: Transparency Log
Immutable evidence logging:
- Merkle tree implementation
- Inclusion proofs
- Tamper detection
- Audit trail export

```python
facts.log_evidence("security_audit", {"result": "passed"})
print(facts.merkle_root)
```

### 06: FastAPI Middleware
Protect HTTP APIs:
- Automatic header extraction
- Agent verification middleware
- Dependency injection for routes

```python
app.add_middleware(FastAPIMiddleware, verify_peers=True)

@app.get("/secure")
def endpoint(agent=Depends(require_verified_agent())):
    return {"accessed_by": agent.did}
```

### 07: CLI Usage
Command-line tools:
- Key generation and management
- Metadata creation (signed by default)
- Trust badge generation
- Policy verification

```bash
agentfacts keys generate -o agent.pem
agentfacts agent create --name "Agent" -o agent.json
agentfacts badge agent.json --format markdown
```

### 08: Complete Workflow
Full enterprise deployment scenario:
1. Create agent identity
2. Pass security audit
3. Sign metadata
4. Service verification
5. Authenticated session
6. Export audit trail
7. Third-party verification

### 10: Logging and Audit Trail
Structured logging plus an auditable evidence log:
- JSON log output with contextual fields
- Signature + verification events
- Policy evaluation logging
- Evidence log export for audits

```python
from agentfacts.logging import configure_logging

logger = configure_logging(json_output=True)
logger.set_context(environment="local", run_id="demo-01")
```

### 11: Custom Metadata Provider
Build a provider that resolves metadata by DID:
- In-memory storage for signed profiles
- `resolve()` hook used by middleware
- Rehydration + verification from stored metadata
- Drop-in for `FastAPIMiddleware` / `FlaskMiddleware` full verification

```python
provider = InMemoryMetadataProvider()
provider.upsert(facts)
metadata = provider.resolve(facts.did)
```

## Running Examples

All examples are self-contained and can be run directly:

```bash
# Basic examples (no external dependencies)
python examples/01_basic_usage.py
python examples/03_verified_handshake.py
python examples/04_policy_verification.py
python examples/05_transparency_log.py
python examples/10_logging_and_audit.py
python examples/11_custom_metadata_provider.py

# LangChain example (requires langchain)
pip install langchain langchain-core
python examples/02_langchain_integration.py

# FastAPI example (requires fastapi, uvicorn)
pip install agentfacts[middleware] uvicorn
python examples/06_fastapi_middleware.py

# Full verification middleware example (requires fastapi and/or flask)
pip install agentfacts[middleware]
python examples/09_full_verify_middleware.py

# CLI example (requires click, rich)
pip install agentfacts[cli]
python examples/07_cli_usage.py

# Complete workflow
python examples/08_complete_workflow.py
```

## Output Files

Some examples generate output files:
- `agent_metadata.json` - Exported agent metadata
- `agent_final_metadata.json` - Signed metadata with attestations
- `agent_transparency_log.json` - Merkle tree evidence log

These are created in the current working directory.

## Next Steps

After exploring these examples:

1. **Read the API docs** - See the docstrings in `src/agentfacts/`
2. **Run the tests** - `pytest tests/` shows more usage patterns
3. **Build your own** - Start with example 01 and customize

## Questions?

Open an issue at https://github.com/agentfacts/agentfacts-py/issues
