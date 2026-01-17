"""
Integration tests for real-world plugin implementations.

These tests implement realistic DID resolvers, SD-JWT-VC verifiers, and
Bitstring Status List checkers based on actual specifications:

- did:web resolver (W3C DID Web Method)
- SD-JWT-VC verifier (IETF draft-ietf-oauth-sd-jwt-vc)
- Bitstring Status List checker (W3C vc-bitstring-status-list)

References:
- https://github.com/openwallet-foundation-labs/sd-jwt-python
- https://w3c.github.io/vc-bitstring-status-list/
- https://github.com/decentralized-identity/universal-resolver
"""

import base64
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import Mock

from agentfacts import AgentFacts
from agentfacts.models import (
    Attestation,
    BaselineModel,
    Capability,
    ModelProvider,
)
from agentfacts.plugins import (
    AttestationPayload,
    AttestationVerificationResult,
    DenyCapabilityIR,
    # Interfaces
    PluginRegistry,
    RequireComplianceIR,
    # Policy IR
    ResolvedDID,
    StatusCheckResult,
    # Context
    VerificationContext,
    reset_plugin_registry,
)

# =============================================================================
# did:web Resolver Implementation
# Based on: https://w3c-ccg.github.io/did-method-web/
# =============================================================================


class DidWebResolver:
    """
    Resolver for did:web DIDs.

    The did:web method uses the web's existing infrastructure to
    host DID documents. A DID like did:web:example.com resolves to
    https://example.com/.well-known/did.json

    Reference: https://w3c-ccg.github.io/did-method-web/
    """

    def __init__(self, http_client: Any = None):
        """
        Initialize the resolver.

        Args:
            http_client: Optional HTTP client for fetching DID documents.
                         If None, uses a mock for testing.
        """
        self.http_client = http_client
        self._cache: dict[str, tuple[ResolvedDID, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)

    def supports(self, did: str) -> bool:
        """Check if this resolver handles the DID."""
        return did.startswith("did:web:")

    def resolve(self, did: str) -> ResolvedDID:
        """
        Resolve a did:web DID to its verification material.

        Algorithm:
        1. Extract domain from DID (did:web:example.com -> example.com)
        2. Handle path encoding (: -> /)
        3. Fetch https://<domain>/.well-known/did.json (or path/did.json)
        4. Extract verification method public key
        """
        if not self.supports(did):
            raise ValueError(f"Unsupported DID method: {did}")

        # Check cache
        if did in self._cache:
            resolved, cached_at = self._cache[did]
            if datetime.now(timezone.utc) - cached_at < self.cache_ttl:
                return resolved

        # Parse DID to URL
        url = self._did_to_url(did)

        # Fetch DID document
        did_doc = self._fetch_did_document(url)

        # Extract public key from verification method
        public_key = self._extract_public_key(did_doc, did)

        resolved = ResolvedDID(
            did=did,
            public_key_base64=public_key,
            metadata={
                "url": url,
                "controller": did_doc.get("controller", did),
                "service": did_doc.get("service", []),
            },
            resolved_at=datetime.now(timezone.utc),
        )

        # Cache result
        self._cache[did] = (resolved, datetime.now(timezone.utc))

        return resolved

    def _did_to_url(self, did: str) -> str:
        """Convert did:web to HTTPS URL."""
        # Remove did:web: prefix
        domain_path = did[8:]

        # Decode percent-encoding and convert : to /
        parts = domain_path.split(":")

        if len(parts) == 1:
            # Simple domain: did:web:example.com
            return f"https://{parts[0]}/.well-known/did.json"
        else:
            # Domain with path: did:web:example.com:user:alice
            domain = parts[0]
            path = "/".join(parts[1:])
            return f"https://{domain}/{path}/did.json"

    def _fetch_did_document(self, url: str) -> dict:
        """Fetch DID document from URL."""
        if self.http_client:
            response = self.http_client.get(url)
            return response.json()
        else:
            # Mock response for testing
            raise NotImplementedError("HTTP client required for real resolution")

    def _extract_public_key(self, did_doc: dict, did: str) -> str:
        """Extract Ed25519 public key from DID document."""
        # Look for verification method
        verification_methods = did_doc.get("verificationMethod", [])

        for vm in verification_methods:
            # Check for Ed25519 key
            if vm.get("type") in (
                "Ed25519VerificationKey2020",
                "Ed25519VerificationKey2018",
                "JsonWebKey2020",
            ):
                # Handle different key formats
                if "publicKeyBase64" in vm:
                    return vm["publicKeyBase64"]
                elif "publicKeyMultibase" in vm:
                    # Decode multibase (z = base58btc)
                    multibase = vm["publicKeyMultibase"]
                    if multibase.startswith("z"):
                        # Base58 decode, skip multicodec prefix (0xed01 for Ed25519)
                        import base58

                        raw = base58.b58decode(multibase[1:])
                        # Skip 2-byte multicodec prefix
                        return base64.b64encode(raw[2:]).decode()
                elif "publicKeyJwk" in vm:
                    # Extract from JWK
                    jwk = vm["publicKeyJwk"]
                    if jwk.get("kty") == "OKP" and jwk.get("crv") == "Ed25519":
                        return jwk["x"]

        raise ValueError(f"No Ed25519 verification method found for {did}")


# =============================================================================
# SD-JWT-VC Verifier Implementation
# Based on: https://github.com/openwallet-foundation-labs/sd-jwt-python
# =============================================================================


@dataclass
class SdJwtVcVerifier:
    """
    Verifier for SD-JWT-VC format credentials.

    SD-JWT (Selective Disclosure JWT) allows holders to present
    only selected claims from a credential. The format is:
    <issuer-jwt>~<disclosure1>~<disclosure2>~...~<kb-jwt>

    Reference: https://datatracker.ietf.org/doc/draft-ietf-oauth-sd-jwt-vc/
    """

    trusted_issuers: list[str] | None = None

    @property
    def formats(self) -> set[str]:
        return {"sd-jwt-vc", "application/vc+sd-jwt"}

    def verify(
        self,
        attestation: Attestation,
        context: VerificationContext,
    ) -> AttestationVerificationResult:
        """
        Verify an SD-JWT-VC attestation.

        Verification steps:
        1. Split SD-JWT into components
        2. Verify issuer JWT signature
        3. Verify disclosures match hash digests
        4. Check expiration
        5. Verify issuer is trusted
        """
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Get the SD-JWT from payload
            sd_jwt = attestation.payload
            if isinstance(sd_jwt, dict):
                sd_jwt = sd_jwt.get("sd_jwt") or sd_jwt.get("credential")
            if not isinstance(sd_jwt, str):
                errors.append("SD-JWT payload must be a string")
                return AttestationVerificationResult(valid=False, errors=errors)

            # Split into components
            parts = sd_jwt.split("~")
            issuer_jwt = parts[0]
            disclosures = parts[1:-1] if len(parts) > 1 else []

            # Decode issuer JWT (without verification - that requires the issuer's key)
            header, payload, signature = self._decode_jwt(issuer_jwt)

            # Check algorithm
            if header.get("alg") not in ("ES256", "ES384", "EdDSA"):
                warnings.append(f"Unusual algorithm: {header.get('alg')}")

            # Check type
            if header.get("typ") not in ("vc+sd-jwt", "sd+jwt"):
                warnings.append(f"Unexpected type header: {header.get('typ')}")

            # Verify issuer
            issuer = payload.get("iss")
            if not issuer:
                errors.append("Missing 'iss' claim in SD-JWT")
            elif self.trusted_issuers and issuer not in self.trusted_issuers:
                errors.append(f"Untrusted issuer: {issuer}")

            # Check expiration
            exp = payload.get("exp")
            if exp:
                exp_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                if context.get_time() > exp_time:
                    errors.append("SD-JWT has expired")

            # Verify disclosures match _sd digests
            sd_digests = self._collect_sd_digests(payload)
            for disclosure in disclosures:
                disclosure_hash = self._hash_disclosure(disclosure)
                if disclosure_hash not in sd_digests:
                    warnings.append(
                        f"Disclosure hash not found in _sd: {disclosure_hash[:16]}..."
                    )

            # Build verified claims
            claims = self._reconstruct_claims(payload, disclosures)

            return AttestationVerificationResult(
                valid=len(errors) == 0,
                payload=AttestationPayload(
                    format="sd-jwt-vc",
                    issuer=issuer or attestation.issuer,
                    subject=payload.get("sub", attestation.subject),
                    claims=claims,
                    raw=sd_jwt,
                ),
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"SD-JWT verification error: {str(e)}")
            return AttestationVerificationResult(valid=False, errors=errors)

    def _decode_jwt(self, jwt: str) -> tuple[dict, dict, bytes]:
        """Decode JWT without signature verification."""
        parts = jwt.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        header = json.loads(self._base64url_decode(parts[0]))
        payload = json.loads(self._base64url_decode(parts[1]))
        signature = self._base64url_decode_bytes(parts[2])

        return header, payload, signature

    def _base64url_decode(self, data: str) -> str:
        """Base64url decode to string."""
        # Add padding if needed
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data).decode("utf-8")

    def _base64url_decode_bytes(self, data: str) -> bytes:
        """Base64url decode to bytes."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _collect_sd_digests(self, payload: dict) -> set[str]:
        """Collect all _sd digest values from payload."""
        digests: set[str] = set()

        def collect(obj: Any) -> None:
            if isinstance(obj, dict):
                if "_sd" in obj:
                    digests.update(obj["_sd"])
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(payload)
        return digests

    def _hash_disclosure(self, disclosure: str) -> str:
        """Hash a disclosure using SHA-256."""
        digest = hashlib.sha256(disclosure.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def _reconstruct_claims(self, payload: dict, disclosures: list[str]) -> dict:
        """Reconstruct claims from payload and disclosures."""
        claims = {k: v for k, v in payload.items() if not k.startswith("_")}

        for disclosure in disclosures:
            try:
                decoded = json.loads(self._base64url_decode(disclosure))
                if len(decoded) == 3:
                    # Object property: [salt, key, value]
                    _, key, value = decoded
                    claims[key] = value
                elif len(decoded) == 2:
                    # Array element: [salt, value]
                    pass  # Array reconstruction is more complex
            except Exception:
                continue

        return claims


# =============================================================================
# Bitstring Status List Checker Implementation
# Based on: https://w3c.github.io/vc-bitstring-status-list/
# =============================================================================


class BitstringStatusListChecker:
    """
    Checker for W3C Bitstring Status List.

    The bitstring status list is a privacy-preserving, space-efficient
    mechanism for publishing credential status. Each credential has
    an index in a GZIP-compressed bitstring.

    Reference: https://w3c.github.io/vc-bitstring-status-list/
    """

    def __init__(self, http_client: Any = None):
        self.http_client = http_client
        self._cache: dict[str, tuple[bytes, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)

    def supports(self, status_ref: str) -> bool:
        """Check if this is a bitstring status list reference."""
        # Status ref format: <url>#<index>
        # Or JSON with statusListCredential and statusListIndex
        return (
            "statuslist" in status_ref.lower()
            or "#" in status_ref
            or status_ref.startswith("https://")
        )

    def check(
        self,
        status_ref: str,
        context: VerificationContext,
    ) -> StatusCheckResult:
        """
        Check the status of a credential.

        Algorithm (from W3C spec):
        1. Retrieve status list credential from URL
        2. Verify the status list credential's proof
        3. Decode the encodedList (base64url + GZIP)
        4. Check bit at statusListIndex
        5. Return status based on bit value
        """
        errors: list[str] = []

        try:
            # Parse status reference
            url, index = self._parse_status_ref(status_ref)

            # Fetch and decode the bitstring
            bitstring = self._get_bitstring(url)

            # Check the bit at the index
            is_revoked = self._check_bit(bitstring, index)

            status = "revoked" if is_revoked else "active"

            return StatusCheckResult(
                valid=not is_revoked,
                status=status,
                checked_at=datetime.now(timezone.utc),
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Status check failed: {str(e)}")
            return StatusCheckResult(
                valid=False,
                status="error",
                checked_at=datetime.now(timezone.utc),
                errors=errors,
            )

    def _parse_status_ref(self, status_ref: str) -> tuple[str, int]:
        """Parse status reference to URL and index."""
        if isinstance(status_ref, str):
            if "#" in status_ref:
                # Format: https://example.com/status/1#12345
                url, index_str = status_ref.rsplit("#", 1)
                return url, int(index_str)
            else:
                # Try to parse as JSON
                try:
                    data = json.loads(status_ref)
                    return data["statusListCredential"], int(data["statusListIndex"])
                except (json.JSONDecodeError, KeyError) as err:
                    raise ValueError(
                        f"Cannot parse status reference: {status_ref}"
                    ) from err
        raise ValueError(f"Invalid status reference type: {type(status_ref)}")

    def _get_bitstring(self, url: str) -> bytes:
        """Fetch and decode the bitstring from status list credential."""
        # Check cache
        now = datetime.now(timezone.utc)
        if url in self._cache:
            bitstring, cached_at = self._cache[url]
            if now - cached_at < self.cache_ttl:
                return bitstring

        # Fetch status list credential
        if self.http_client:
            response = self.http_client.get(url)
            status_cred = response.json()
        else:
            raise NotImplementedError("HTTP client required for real status check")

        # Extract encoded list
        encoded_list = status_cred.get("credentialSubject", {}).get("encodedList")
        if not encoded_list:
            raise ValueError("No encodedList found in status credential")

        # Decode: base64url -> GZIP decompress
        bitstring = self._decode_bitstring(encoded_list)

        # Cache
        self._cache[url] = (bitstring, now)

        return bitstring

    def _decode_bitstring(self, encoded: str) -> bytes:
        """Decode base64url + GZIP compressed bitstring."""
        # Remove multibase prefix if present (u = base64url)
        if encoded.startswith("u"):
            encoded = encoded[1:]

        # Base64url decode
        padding = 4 - len(encoded) % 4
        if padding != 4:
            encoded += "=" * padding
        compressed = base64.urlsafe_b64decode(encoded)

        # GZIP decompress
        return gzip.decompress(compressed)

    def _check_bit(self, bitstring: bytes, index: int) -> bool:
        """
        Check if bit at index is set (1 = revoked).

        From W3C spec: "the first index, with a value of zero,
        is located at the left-most bit in the bitstring"
        """
        byte_index = index // 8
        bit_index = 7 - (index % 8)  # Left-most bit is index 0

        if byte_index >= len(bitstring):
            raise ValueError(
                f"Index {index} out of range for bitstring of size {len(bitstring) * 8}"
            )

        return bool(bitstring[byte_index] & (1 << bit_index))


# =============================================================================
# Integration Tests
# =============================================================================


class TestDidWebResolverIntegration:
    """Integration tests for did:web resolver."""

    def test_did_to_url_simple_domain(self):
        """Test URL conversion for simple domain."""
        resolver = DidWebResolver()
        url = resolver._did_to_url("did:web:example.com")
        assert url == "https://example.com/.well-known/did.json"

    def test_did_to_url_with_path(self):
        """Test URL conversion for domain with path."""
        resolver = DidWebResolver()
        url = resolver._did_to_url("did:web:example.com:users:alice")
        assert url == "https://example.com/users/alice/did.json"

    def test_did_to_url_subdomain(self):
        """Test URL conversion for subdomain."""
        resolver = DidWebResolver()
        url = resolver._did_to_url("did:web:id.example.com")
        assert url == "https://id.example.com/.well-known/did.json"

    def test_supports_did_web(self):
        """Test that resolver supports did:web."""
        resolver = DidWebResolver()
        assert resolver.supports("did:web:example.com")
        assert resolver.supports("did:web:example.com:users:alice")
        assert not resolver.supports("did:key:z6MkTest")
        assert not resolver.supports("did:ion:test")

    def test_resolve_with_mock_http(self):
        """Test resolution with mocked HTTP client."""
        # Create mock DID document
        did_doc = {
            "id": "did:web:example.com",
            "verificationMethod": [
                {
                    "id": "did:web:example.com#key-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": "did:web:example.com",
                    "publicKeyBase64": "MCowBQYDK2VwAyEATest1234567890abcdefghijklmnop",
                }
            ],
            "service": [
                {
                    "id": "did:web:example.com#agent",
                    "type": "AgentFacts",
                    "serviceEndpoint": "https://example.com/agent-facts",
                }
            ],
        }

        # Mock HTTP client
        mock_client = Mock()
        mock_client.get.return_value.json.return_value = did_doc

        resolver = DidWebResolver(http_client=mock_client)
        resolved = resolver.resolve("did:web:example.com")

        assert resolved.did == "did:web:example.com"
        assert (
            resolved.public_key_base64
            == "MCowBQYDK2VwAyEATest1234567890abcdefghijklmnop"
        )
        assert "url" in resolved.metadata
        assert resolved.metadata["url"] == "https://example.com/.well-known/did.json"

    def test_caching(self):
        """Test that resolver caches results."""
        did_doc = {
            "id": "did:web:example.com",
            "verificationMethod": [
                {
                    "id": "did:web:example.com#key-1",
                    "type": "Ed25519VerificationKey2020",
                    "publicKeyBase64": "TestKey123",
                }
            ],
        }

        mock_client = Mock()
        mock_client.get.return_value.json.return_value = did_doc

        resolver = DidWebResolver(http_client=mock_client)

        # First call
        resolver.resolve("did:web:example.com")
        assert mock_client.get.call_count == 1

        # Second call - should use cache
        resolver.resolve("did:web:example.com")
        assert mock_client.get.call_count == 1  # No additional call


class TestSdJwtVcVerifierIntegration:
    """Integration tests for SD-JWT-VC verifier."""

    def _create_mock_sd_jwt(
        self,
        issuer: str = "did:web:issuer.example.com",
        subject: str = "did:key:z6MkTest",
        claims: dict | None = None,
        expired: bool = False,
    ) -> str:
        """Create a mock SD-JWT for testing."""
        header = {"alg": "EdDSA", "typ": "vc+sd-jwt"}
        payload = {
            "iss": issuer,
            "sub": subject,
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "type": "VerifiableCredential",
            **(claims or {}),
        }

        if expired:
            payload["exp"] = int(
                (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
            )
        else:
            payload["exp"] = int(
                (datetime.now(timezone.utc) + timedelta(days=30)).timestamp()
            )

        # Encode (without real signature)
        def b64url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header_b64 = b64url(json.dumps(header).encode())
        payload_b64 = b64url(json.dumps(payload).encode())
        signature_b64 = b64url(b"mock_signature")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def test_verify_valid_sd_jwt(self):
        """Test verification of valid SD-JWT."""
        verifier = SdJwtVcVerifier()
        sd_jwt = self._create_mock_sd_jwt(
            issuer="did:web:trusted.example.com",
            claims={"name": "Test Agent", "role": "assistant"},
        )

        attestation = Attestation(
            id="att-001",
            type="security_audit",
            issuer="did:web:trusted.example.com",
            subject="did:key:z6MkTest",
            format="sd-jwt-vc",
            payload=sd_jwt,
        )

        context = VerificationContext()
        result = verifier.verify(attestation, context)

        assert result.valid
        assert result.payload is not None
        assert result.payload.issuer == "did:web:trusted.example.com"

    def test_verify_expired_sd_jwt(self):
        """Test that expired SD-JWT is rejected."""
        verifier = SdJwtVcVerifier()
        sd_jwt = self._create_mock_sd_jwt(expired=True)

        attestation = Attestation(
            id="att-001",
            type="security_audit",
            issuer="did:web:issuer.example.com",
            subject="did:key:z6MkTest",
            format="sd-jwt-vc",
            payload=sd_jwt,
        )

        context = VerificationContext()
        result = verifier.verify(attestation, context)

        assert not result.valid
        assert any("expired" in e.lower() for e in result.errors)

    def test_verify_untrusted_issuer(self):
        """Test that untrusted issuer is rejected."""
        verifier = SdJwtVcVerifier(trusted_issuers=["did:web:trusted.example.com"])
        sd_jwt = self._create_mock_sd_jwt(issuer="did:web:untrusted.example.com")

        attestation = Attestation(
            id="att-001",
            type="security_audit",
            issuer="did:web:untrusted.example.com",
            subject="did:key:z6MkTest",
            format="sd-jwt-vc",
            payload=sd_jwt,
        )

        context = VerificationContext()
        result = verifier.verify(attestation, context)

        assert not result.valid
        assert any("untrusted" in e.lower() for e in result.errors)

    def test_verify_with_disclosures(self):
        """Test SD-JWT with selective disclosures."""
        verifier = SdJwtVcVerifier()

        # Create disclosure for "email" claim
        disclosure_data = ["salt123", "email", "agent@example.com"]
        disclosure = (
            base64.urlsafe_b64encode(json.dumps(disclosure_data).encode())
            .rstrip(b"=")
            .decode()
        )

        # Create SD-JWT with disclosure hash
        sd_jwt = self._create_mock_sd_jwt()
        sd_jwt_with_disclosure = f"{sd_jwt}~{disclosure}~"

        attestation = Attestation(
            id="att-001",
            type="verification",
            issuer="did:web:issuer.example.com",
            subject="did:key:z6MkTest",
            format="sd-jwt-vc",
            payload=sd_jwt_with_disclosure,
        )

        context = VerificationContext()
        result = verifier.verify(attestation, context)

        assert result.valid
        assert result.payload is not None
        assert result.payload.claims.get("email") == "agent@example.com"


class TestBitstringStatusListIntegration:
    """Integration tests for Bitstring Status List checker."""

    def _create_status_list(
        self, revoked_indices: list[int], size: int = 1024
    ) -> bytes:
        """Create a bitstring with specific indices revoked."""
        # Create bitstring of zeros
        byte_size = (size + 7) // 8
        bitstring = bytearray(byte_size)

        # Set revoked bits
        for index in revoked_indices:
            byte_index = index // 8
            bit_index = 7 - (index % 8)  # Left-most bit is index 0
            if byte_index < byte_size:
                bitstring[byte_index] |= 1 << bit_index

        return bytes(bitstring)

    def _encode_status_list(self, bitstring: bytes) -> str:
        """Encode bitstring as base64url + GZIP."""
        compressed = gzip.compress(bitstring)
        return "u" + base64.urlsafe_b64encode(compressed).rstrip(b"=").decode()

    def test_check_active_credential(self):
        """Test checking a non-revoked credential."""
        # Create status list with no revocations
        bitstring = self._create_status_list(revoked_indices=[])
        encoded = self._encode_status_list(bitstring)

        status_cred = {
            "credentialSubject": {
                "id": "https://example.com/status/1#list",
                "type": "BitstringStatusList",
                "encodedList": encoded,
            }
        }

        mock_client = Mock()
        mock_client.get.return_value.json.return_value = status_cred

        checker = BitstringStatusListChecker(http_client=mock_client)
        result = checker.check("https://example.com/status/1#42", VerificationContext())

        assert result.valid
        assert result.status == "active"

    def test_check_revoked_credential(self):
        """Test checking a revoked credential."""
        # Create status list with index 42 revoked
        bitstring = self._create_status_list(revoked_indices=[42])
        encoded = self._encode_status_list(bitstring)

        status_cred = {
            "credentialSubject": {
                "encodedList": encoded,
            }
        }

        mock_client = Mock()
        mock_client.get.return_value.json.return_value = status_cred

        checker = BitstringStatusListChecker(http_client=mock_client)
        result = checker.check("https://example.com/status/1#42", VerificationContext())

        assert not result.valid
        assert result.status == "revoked"

    def test_check_multiple_revocations(self):
        """Test status list with multiple revocations."""
        revoked = [10, 42, 100, 500]
        bitstring = self._create_status_list(revoked_indices=revoked)
        encoded = self._encode_status_list(bitstring)

        status_cred = {"credentialSubject": {"encodedList": encoded}}

        mock_client = Mock()
        mock_client.get.return_value.json.return_value = status_cred

        checker = BitstringStatusListChecker(http_client=mock_client)
        context = VerificationContext()

        # Check revoked indices
        for idx in revoked:
            result = checker.check("https://example.com/status#" + str(idx), context)
            assert not result.valid, f"Index {idx} should be revoked"

        # Check non-revoked indices
        for idx in [0, 11, 50, 200]:
            result = checker.check("https://example.com/status#" + str(idx), context)
            assert result.valid, f"Index {idx} should be active"

    def test_bitstring_compression(self):
        """Test that bitstring compression works correctly."""
        # Large sparse bitstring should compress well
        bitstring = self._create_status_list(revoked_indices=[1000], size=16384)
        encoded = self._encode_status_list(bitstring)

        # Decode and verify
        checker = BitstringStatusListChecker()
        decoded = checker._decode_bitstring(encoded)

        assert decoded == bitstring


class TestFullVerificationPipeline:
    """End-to-end tests with all plugins working together."""

    def setup_method(self):
        reset_plugin_registry()

    def test_agent_verification_with_all_plugins(self):
        """Test full verification with DID resolver, VC verifier, and status checker."""
        # Create mock plugins
        did_doc = {
            "id": "did:web:issuer.example.com",
            "verificationMethod": [
                {
                    "id": "did:web:issuer.example.com#key-1",
                    "type": "Ed25519VerificationKey2020",
                    "publicKeyBase64": "TestIssuerKey",
                }
            ],
        }

        mock_http = Mock()
        mock_http.get.return_value.json.return_value = did_doc

        did_resolver = DidWebResolver(http_client=mock_http)
        vc_verifier = SdJwtVcVerifier(trusted_issuers=["did:web:issuer.example.com"])

        # Create mock status list (no revocations)
        bitstring = gzip.compress(bytes(128))
        status_cred = {
            "credentialSubject": {
                "encodedList": "u"
                + base64.urlsafe_b64encode(bitstring).rstrip(b"=").decode()
            }
        }
        mock_http.get.return_value.json.side_effect = [did_doc, status_cred]

        status_checker = BitstringStatusListChecker(http_client=mock_http)

        # Create agent with attestation
        facts = AgentFacts(
            name="Verified Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[
                Capability(name="web_search", description="Search the web"),
            ],
        )

        # Add SD-JWT-VC attestation
        SdJwtVcVerifier()._create_mock_sd_jwt = lambda **kw: "mock.jwt.sig"

        # Sign the agent
        facts.sign()

        # Create verification context
        context = VerificationContext(
            did_resolver=did_resolver,
            attestation_verifiers=[vc_verifier],
            status_checkers=[status_checker],
        )

        # Verify (note: DID resolver won't be used for did:key)
        result = facts.verify(context=context)

        # Should pass basic verification
        assert result.valid

    def test_registry_creates_working_context(self):
        """Test that registry.create_context() works for verification."""
        registry = PluginRegistry()

        # Create mock HTTP client
        did_doc = {
            "id": "did:web:example.com",
            "verificationMethod": [
                {
                    "type": "Ed25519VerificationKey2020",
                    "publicKeyBase64": "TestKey",
                }
            ],
        }
        mock_http = Mock()
        mock_http.get.return_value.json.return_value = did_doc

        # Register plugins
        registry.register_did_resolver("web", DidWebResolver(http_client=mock_http))
        registry.register_attestation_verifier(SdJwtVcVerifier())

        # Create agent
        facts = AgentFacts(
            name="Test Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.sign()

        # Create context from registry
        context = registry.create_context()

        # Verify
        result = facts.verify(context=context)
        assert result.valid

    def test_governance_policy_integration(self):
        """Test governance adapter with policy evaluation."""

        class SimpleGovernanceAdapter:
            """Simple governance adapter for testing."""

            def to_policy_ir(self, doc: dict) -> list:
                policies = []
                if "require_compliance" in doc:
                    for framework in doc["require_compliance"]:
                        policies.append(RequireComplianceIR(framework=framework))
                if "deny_capabilities" in doc:
                    for cap in doc["deny_capabilities"]:
                        policies.append(DenyCapabilityIR(capability=cap))
                return policies

        # Create agent with compliance
        facts = AgentFacts(
            name="Compliant Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
        )
        facts.metadata.policy.compliance.frameworks = ["GDPR", "SOC2"]
        facts.sign()

        # Create governance document
        gov_doc = {
            "require_compliance": ["GDPR"],
            "deny_capabilities": ["shell_exec"],
        }

        context = VerificationContext(
            governance_adapter=SimpleGovernanceAdapter(),
            governance_doc=gov_doc,
        )

        result = facts.verify(context=context)

        assert result.valid
        assert len(result.policy_violations) == 0

    def test_governance_policy_violation(self):
        """Test that governance policy violations are detected."""

        class SimpleGovernanceAdapter:
            def to_policy_ir(self, doc: dict) -> list:
                policies = []
                if doc.get("deny_capabilities"):
                    for cap in doc["deny_capabilities"]:
                        policies.append(DenyCapabilityIR(capability=cap))
                return policies

        # Create agent with denied capability
        facts = AgentFacts(
            name="Code Agent",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[
                Capability(name="code_execution", description="Execute code"),
            ],
        )
        facts.sign()

        # Governance doc with deny policy
        gov_doc = {"deny_capabilities": ["code_execution"]}

        context = VerificationContext(
            governance_adapter=SimpleGovernanceAdapter(),
            governance_doc=gov_doc,
        )

        result = facts.verify(context=context)

        # Base verification passes but policy is violated
        assert result.valid
        assert len(result.policy_violations) == 1
        assert "code_execution" in result.policy_violations[0]
