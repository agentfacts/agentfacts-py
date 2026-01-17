# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Integration registry for framework introspection with built-in adapters for LangChain/LangGraph, CrewAI, AutoGen, LlamaIndex, OpenAgents, and HuggingFace (smolagents/tiny-agents)
- VerificationContext plugin hooks for DID resolvers, attestation verifiers, status checkers, and governance adapters
- GroupFacts topology refresh on membership changes

### Changed
- Policy IR mapping now supports attestation format and issuer constraints
- LangChain callback handler import path aligned with integrations package

### Fixed
- Integration discovery now registers introspectors after reset
- Verification falls back to did:key when publisher keys are missing
- Merkle log verification fails if a signature is missing
- LangChain callback timestamps now emit UTC ISO8601 with `Z`

## [0.1.0] - 2024-01-15

### Added
- Initial alpha release
- Core `AgentFacts` class with `from_agent()` and `from_langchain()` factories
- `KeyPair` for Ed25519 key management
- `DID` for decentralized identifiers
- `MerkleTree` and `TransparencyLog` for evidence
- Policy rules: `RequireSignature`, `RequireProvider`, `RequireModel`, `RequireAttestation`, `RequireCapability`, `DenyCapability`, `RequireCompliance`, `RequireRiskLevel`
- `PolicyEngine` for centralized policy management
- `AgentFactsCallbackHandler` for LangChain integration
- FastAPI `AgentFactsMiddleware` and `require_verified_agent` dependency
- Flask `AgentFactsMiddleware` and `flask_require_verified_agent` decorator
- CLI commands: `agentfacts keys`, `agentfacts agent`, `agentfacts badge`, `agentfacts verify`, `agentfacts inspect`

### Security
- Ed25519 signatures via `cryptography` library
- Merkle tree with domain separation (prevents second-preimage attacks)
- Secure nonce generation via `secrets.token_bytes()`

[Unreleased]: https://github.com/agentfacts/agentfacts-py/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agentfacts/agentfacts-py/releases/tag/v0.1.0
