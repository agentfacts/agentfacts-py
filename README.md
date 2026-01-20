<p align="center">
  <img src=".github/workflows/AgentFacts.svg" alt="AgentFacts Logo" width="140" />
</p>
<h1 align="center">AgentFacts</h1>

<p align="center"><strong>The open source SDK for AI agents Identity.</strong><br>

## Overview

AgentFacts is an open source SDK that creates a **signed, verifiable profile** for any AI agent. It captures identity, base model, tools, policy, and provenance in a tamper-evident JSON card.
Verify offline with Ed25519 + `did:key`, or auto-introspect popular frameworks for instant metadata. AgentFacts brings trust and transparency to AI agents without heavy lifting by developers.



## Quick Start 🚀

### Installation

```bash
pip install agentfacts
```

**Optional extras:**
```bash
pip install agentfacts[cli]         # CLI tools (agentfacts, af commands)
pip install agentfacts[middleware]  # FastAPI/Flask middleware
pip install agentfacts[all]         # All extras
```

*Requires Python 3.10+.*

### Quickstart Example

```python
from agentfacts import AgentFacts

# Suppose you have an existing LangChain agent (e.g., an AgentExecutor or chain)
agent = MyLangChainAgent(...)

# Generate an AgentFacts profile from the agent (introspects model, tools, etc.)
af = AgentFacts.from_agent(agent, name="CustomerSupportBot", description="Answers support FAQs")

# Sign the agent’s metadata (generates a DID and Ed25519 key pair under the hood)
af.sign()

# The AgentFacts metadata can be obtained as a dict or JSON
profile_json = af.to_json()
print("AgentFacts Profile:", profile_json)

# Verify the AgentFacts signature (returns a VerificationResult object)
result = af.verify()
assert result.valid, f"Verification failed: {result.errors}"
```

This will produce a signed JSON profile (you can save it as `agent_profile.json`) containing the agent’s details and a cryptographic signature. Anyone with the agent’s public DID can call `AgentFacts.verify(...)` to validate that profile. ✅

## Key Features ✨

- **Verifiable identity:** Ed25519 signatures with `did:key` DIDs.
- **Structured metadata:** Model, tools, policy, and compliance in one schema.
- **Tamper evidence:** Merkle log proofs for audit trails.
- **Framework introspection:** Auto-detects popular agent frameworks.
- **Policy checks:** Zero-trust rules for required capabilities and attestations.
- **Lightweight:** Fast signing and RFC 8785 canonical JSON for cross-language parity.

## Supported Frameworks

- LangChain
- LlamaIndex
- Hugging Face Agents
- OpenAgents
- CrewAI
- AutoGen
- Manual/Custom

## AgentFacts Schema

```json
{
  "spec_version": "v0.1",
  "agent": {"id": "did:key:...", "name": "Agent", "model": {"name": "gpt-4", "provider": "openai"}},
  "publisher": {"id": "did:key:...", "keys": [{"id": "did:key:...#sig-1", "type": "Ed25519VerificationKey2020", "public_key": "BASE64"}]},
  "policy": {"compliance": {"frameworks": []}, "constraints": {}},
  "issued_at": "2025-02-12T00:00:00Z",
  "signature": {"alg": "ed25519", "key_id": "did:key:...#sig-1", "value": "BASE64_SIGNATURE"},
  "log_proof": {"log_id": "local", "leaf_hash": "LEAF_HASH", "root_hash": "ROOT_HASH", "inclusion": [{"hash": "SIBLING_HASH_1", "position": "left"}]}
}
```

## Roadmap

- [x] Core SDK: Sign/Verify
- [x] LangChain/CrewAI/HuggingFace integrations
- [ ] CLI Wizard
- [ ] Attestation plugins
- [ ] Web playground
- [ ] Multi-party signing

## Contributing

We welcome issues and PRs. 
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
