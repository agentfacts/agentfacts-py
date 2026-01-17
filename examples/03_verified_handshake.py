"""
Example 3: Verified Handshake Protocol

This example demonstrates agent-to-agent authentication using
the challenge-response handshake protocol.

Use cases:
- Agent A wants to verify Agent B's identity before sharing data
- Service provider wants to authenticate client agents
- Multi-agent systems need mutual authentication
"""

from agentfacts import AgentFacts
from agentfacts.models import BaselineModel, ModelProvider


def main():
    print("=" * 60)
    print("AgentFacts SDK - Verified Handshake Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Setup: Create two agents that need to authenticate each other
    # -------------------------------------------------------------------------
    print("\n1. Setting up two agents...")

    # Agent A: A service provider
    service_agent = AgentFacts(
        name="Data Service Provider",
        description="Provides secure data access to verified agents",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )
    service_agent.sign()

    # Agent B: A client that wants to access the service
    client_agent = AgentFacts(
        name="Research Client",
        description="Needs access to data service for research",
        baseline_model=BaselineModel(name="claude-3-opus", provider=ModelProvider.ANTHROPIC),
    )
    client_agent.sign()

    print(f"   Service Agent: {service_agent.name}")
    print(f"   Service DID:   {service_agent.did}")
    print(f"   Client Agent:  {client_agent.name}")
    print(f"   Client DID:    {client_agent.did}")

    # -------------------------------------------------------------------------
    # Step 1: Service creates a challenge
    # -------------------------------------------------------------------------
    print("\n2. Service agent creates a challenge...")

    challenge = service_agent.create_challenge(ttl_seconds=300)  # 5 minute expiry

    print(f"   Nonce:        {challenge.nonce[:30]}...")
    print(f"   Challenger:   {challenge.challenger_did}")
    print(f"   Expires:      {challenge.expires_at}")

    # In a real scenario, this challenge would be sent over HTTP or A2A protocol

    # -------------------------------------------------------------------------
    # Step 2: Client responds by signing the nonce
    # -------------------------------------------------------------------------
    print("\n3. Client agent responds to challenge...")

    response = client_agent.respond_to_challenge(challenge)

    print(f"   Responder:    {response.responder_did}")
    print(f"   Signature:    {response.signature[:30]}...")
    print(f"   Public Key:   {response.public_key[:30]}...")
    print(f"   Metadata Hash: {response.metadata_hash}")

    # -------------------------------------------------------------------------
    # Step 3: Service verifies the response
    # -------------------------------------------------------------------------
    print("\n4. Service agent verifies the response...")

    result = service_agent.verify_response(challenge, response)

    print(f"   Valid:        {result.valid}")
    print(f"   Verified DID: {result.did}")

    if result.valid:
        print("\n   ✓ Authentication successful!")
        print(f"   ✓ Client {response.responder_did} is who they claim to be")
    else:
        print("\n   ✗ Authentication failed!")
        print(f"   Errors: {result.errors}")

    # -------------------------------------------------------------------------
    # Demonstration: What happens with a forged response?
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("5. Security Demo: Detecting a forged response")
    print("-" * 50)

    # Create a malicious agent trying to impersonate the client
    malicious_agent = AgentFacts(name="Impersonator")

    # The malicious agent creates its own response
    malicious_response = malicious_agent.respond_to_challenge(challenge)

    # But claims to be the legitimate client (spoofed DID)
    from agentfacts.models import HandshakeResponse
    forged_response = HandshakeResponse(
        nonce=challenge.nonce,
        responder_did=client_agent.did,  # Claiming to be the real client!
        signature=malicious_response.signature,  # But signed with wrong key
        public_key=malicious_response.public_key,
    )

    # Service tries to verify
    forged_result = service_agent.verify_response(challenge, forged_response)

    print(f"   Forged DID claims: {forged_response.responder_did}")
    print(f"   Verification:      {'✗ REJECTED' if not forged_result.valid else '✓ Accepted'}")
    if not forged_result.valid:
        print(f"   Reason:            {forged_result.errors}")

    # -------------------------------------------------------------------------
    # Mutual Authentication (both agents verify each other)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("6. Mutual Authentication")
    print("-" * 50)

    print("\n   Round 1: Service verifies Client")
    challenge1 = service_agent.create_challenge()
    response1 = client_agent.respond_to_challenge(challenge1)
    result1 = service_agent.verify_response(challenge1, response1)
    print(f"   Service → Client: {'✓ Verified' if result1.valid else '✗ Failed'}")

    print("\n   Round 2: Client verifies Service")
    challenge2 = client_agent.create_challenge()
    response2 = service_agent.respond_to_challenge(challenge2)
    result2 = client_agent.verify_response(challenge2, response2)
    print(f"   Client → Service: {'✓ Verified' if result2.valid else '✗ Failed'}")

    if result1.valid and result2.valid:
        print("\n   ✓ Mutual authentication complete!")
        print("   Both agents have verified each other's identity.")

    # -------------------------------------------------------------------------
    # Using handshake with HTTP headers
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("7. HTTP Header Integration")
    print("-" * 50)

    from agentfacts.middleware.headers import inject_agent_headers

    # Prepare headers for an HTTP request
    headers = {}
    nonce = challenge.nonce

    headers = inject_agent_headers(headers, client_agent, nonce=nonce)

    print("   HTTP Headers for authenticated request:")
    for key, value in headers.items():
        display_value = value[:40] + "..." if len(value) > 40 else value
        print(f"     {key}: {display_value}")

    print("\n" + "=" * 60)
    print("Verified handshake example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
