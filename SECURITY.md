# Security Policy

## Reporting a Vulnerability

The AgentFacts team takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@agentfacts.ai**

Include the following information:
- Type of issue (e.g., buffer overflow, signature bypass, key leakage)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 90 days (depending on complexity)

### Disclosure Policy

- We will acknowledge receipt of your vulnerability report
- We will work with you to understand and validate the issue
- We will develop and test a fix
- We will publicly disclose the vulnerability after a fix is available
- We will credit you in our security advisory (unless you prefer anonymity)

## Security Assumptions

AgentFacts makes the following security assumptions:

### Cryptographic Primitives

1. **Ed25519 Signatures**: We rely on the security of Ed25519 as implemented in the `cryptography` library (which uses OpenSSL/BoringSSL).

2. **SHA-256 Hashing**: Used for Merkle tree nodes and content hashing.

3. **Random Number Generation**: We use `secrets.token_bytes()` for all cryptographic randomness.

### Trust Model

1. **Private Key Security**: The security of an agent's identity depends entirely on keeping the private key secret. AgentFacts does not protect against compromised private keys.

2. **DID Resolution**: The `did:key` method is self-certifying. Other DID methods can be verified via optional resolvers supplied through `VerificationContext`; resolution trust and any network dependencies are external to AgentFacts.

3. **Attestation Trust**: Attestations are only as trustworthy as their issuers. By default, AgentFacts treats attestation payloads as opaque; when `VerificationContext` provides verifiers and status checkers, `verify()` can validate attestation signatures and revocation status, but semantics remain issuer-defined.

4. **Clock Synchronization**: The handshake protocol assumes reasonable clock synchronization between agents (within the challenge TTL).

5. **External Plugins**: If you enable DID resolvers, attestation verifiers, or status checkers, security depends on those plugins and any upstream services they use.

## Known Limitations

### No Key Revocation

Currently, there is no mechanism to revoke a compromised `did:key`. If your agent's private key is compromised:
1. Generate a new key pair
2. Update all systems to use the new DID
3. Consider the old DID permanently compromised

### No Replay Protection for Signed Metadata

Signed metadata can be presented indefinitely. The `updated_at` timestamp provides some freshness indication, but there is no built-in expiration.

### Memory Safety

Private keys remain in memory for the lifetime of the `KeyPair` object. In high-security environments, consider:
- Minimizing key lifetime
- Using hardware security modules (HSMs) - not currently supported

## Secure Usage Guidelines

### Key Management

```python
# DO: Generate keys securely
key_pair = KeyPair.generate()

# DO: Save with restrictive permissions (automatic)
key_pair.save("agent.pem")  # Sets 0o600 permissions

# DON'T: Log or print private keys
print(key_pair.private_key_base64)  # NEVER DO THIS

# DON'T: Commit keys to version control
# Add *.pem to .gitignore
```

### Signature Verification

```python
# DO: Always verify signatures before trusting metadata
result = facts.verify()
if not result.valid:
    raise SecurityError("Invalid signature")

# DO: Check signature freshness
if facts.metadata.updated_at < datetime.now(timezone.utc) - timedelta(days=30):
    logging.warning("Stale metadata")
```

### Policy Enforcement

```python
# DO: Use strict policies in production
policy = Policy.strict_enterprise()

# DO: Deny dangerous capabilities
policy.add_rule(DenyCapability(["shell", "code_executor", "sudo"]))

# DON'T: Trust unverified agents
if not engine.is_trusted(facts.metadata):
    return Response(status_code=403)
```

## Dependency Security

AgentFacts depends on:

| Package | Purpose | Security Notes |
|---------|---------|----------------|
| `cryptography` | Ed25519 signatures | Actively maintained, audited |
| `pydantic` | Data validation | Type-safe, prevents injection |
| `httpx` | HTTP client | Modern, secure defaults |

We recommend:
- Regularly updating dependencies
- Using `pip-audit` or `safety` to check for vulnerabilities
- Pinning versions in production

## Security Changelog

### Unreleased
- Optional strict publisher DID matching (fail on mismatch in production)
- Optional log checkpoint verification hook for transparency logs
- Transparency log import now detects merkle root tampering

### v0.1.0 (Initial Release)
- Initial security model established
- Ed25519 signature scheme implemented
- Merkle tree transparency log with domain separation

---

For questions about this security policy, contact security@agentfacts.dev
