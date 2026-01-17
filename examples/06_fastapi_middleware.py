"""
Example 6: FastAPI Middleware Integration

This example demonstrates:
- Setting up FastAPI middleware for agent verification
- Protecting endpoints with agent identity
- Client-side header injection
- Complete request/response flow

To run this example:
    pip install agentfacts[middleware] uvicorn
    python 06_fastapi_middleware.py

Then test with the included client code or curl.
"""

import asyncio
import json
from typing import Optional

# Check for FastAPI
try:
    from fastapi import FastAPI, Depends, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install agentfacts[middleware] uvicorn")

from agentfacts import AgentFacts
from agentfacts.models import BaselineModel, ModelProvider


def create_server_app():
    """Create a FastAPI application with AgentFacts middleware."""
    from agentfacts.middleware import (
        FastAPIMiddleware,
        require_verified_agent,
        get_verified_agent,
    )
    from agentfacts.policy import Policy

    app = FastAPI(
        title="AgentFacts Protected API",
        description="An API that verifies agent identity",
    )

    # Add AgentFacts middleware
    # verify_peers=False means it will accept requests without agent headers
    # but will still extract and validate headers if present
    app.add_middleware(
        FastAPIMiddleware,
        verify_peers=False,  # Set to True for strict mode
        exclude_paths=["/", "/health", "/docs", "/openapi.json"],
    )

    @app.get("/")
    async def root():
        """Public endpoint - no verification required."""
        return {"message": "Welcome to the AgentFacts Protected API"}

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/public/data")
    async def public_data(agent=Depends(get_verified_agent)):
        """
        Public endpoint that optionally shows agent info.
        Works for both verified and anonymous requests.
        """
        if agent and agent.valid:
            return {
                "data": "Here is some public data",
                "viewer": agent.did,
                "verified": True,
            }
        return {
            "data": "Here is some public data",
            "viewer": "anonymous",
            "verified": False,
        }

    @app.get("/secure/data")
    async def secure_data(agent=Depends(require_verified_agent())):
        """
        Secure endpoint - requires verified agent identity.
        Returns 403 if agent is not verified.
        """
        return {
            "secret_data": "This is sensitive information",
            "accessed_by": agent.did,
            "access_time": "2024-01-15T10:30:00Z",
        }

    @app.post("/secure/action")
    async def secure_action(
        action: dict,
        agent=Depends(require_verified_agent()),
    ):
        """
        Secure action endpoint - requires verified agent.
        """
        return {
            "status": "success",
            "action_performed": action,
            "performed_by": agent.did,
        }

    return app


def demonstrate_client_requests():
    """Show how to make authenticated requests as an agent."""
    print("\n" + "=" * 60)
    print("Client-Side: Making Authenticated Requests")
    print("=" * 60)

    import httpx
    from agentfacts.middleware.headers import inject_agent_headers, create_httpx_auth

    # Create a client agent
    client_agent = AgentFacts(
        name="API Client Agent",
        description="An agent that consumes the protected API",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )
    client_agent.sign()

    print(f"\nClient Agent: {client_agent.name}")
    print(f"Client DID:   {client_agent.did}")

    # Method 1: Manual header injection
    print("\n--- Method 1: Manual Header Injection ---")
    headers = {}
    headers = inject_agent_headers(headers, client_agent)
    print("Headers to send:")
    for key, value in headers.items():
        display = value[:40] + "..." if len(value) > 40 else value
        print(f"  {key}: {display}")

    # Method 2: Using httpx Auth class
    print("\n--- Method 2: Using httpx Auth ---")
    auth = create_httpx_auth(client_agent)
    print(f"Created auth object: {type(auth).__name__}")
    print("Usage: httpx.get(url, auth=auth)")

    # Simulate requests (without actual server)
    print("\n--- Simulated Request Flow ---")

    print("\n1. Anonymous request to /public/data")
    print("   Response: {'data': '...', 'viewer': 'anonymous', 'verified': False}")

    print("\n2. Authenticated request to /public/data")
    print(f"   Headers: X-Agent-DID: {client_agent.did}")
    print("   Response: {'data': '...', 'viewer': '<did>', 'verified': True}")

    print("\n3. Authenticated request to /secure/data")
    print(f"   Headers: X-Agent-DID: {client_agent.did}")
    print("   Response: {'secret_data': '...', 'accessed_by': '<did>'}")

    print("\n4. Anonymous request to /secure/data")
    print("   Response: 403 {'error': 'Agent Identity Unverified'}")


def demonstrate_strict_mode():
    """Show strict mode where all requests require verification."""
    print("\n" + "=" * 60)
    print("Strict Mode: All Requests Must Be Verified")
    print("=" * 60)

    print("""
In strict mode (verify_peers=True), the middleware will:

1. Reject any request without X-Agent-DID header
2. Reject any request without X-Agent-Public-Key header
3. Verify that DID matches the public key
4. Optionally verify nonce signatures for replay protection

Configuration:

    app.add_middleware(
        FastAPIMiddleware,
        verify_peers=True,  # STRICT MODE
        exclude_paths=["/health"],  # Only health check is public
    )

Rejected requests receive:

    {
        "error": "Agent Identity Unverified",
        "code": 403,
        "message": "Missing agent identity headers..."
    }
""")


def demonstrate_with_nonce():
    """Show request with nonce for replay protection."""
    print("\n" + "=" * 60)
    print("Nonce-Based Replay Protection")
    print("=" * 60)

    from agentfacts.middleware.headers import inject_agent_headers
    from agentfacts.crypto.keys import KeyPair

    # Create agent
    agent = AgentFacts(name="Secure Client")
    agent.sign()

    # Generate a nonce for this request
    nonce = KeyPair.generate_nonce()

    # Inject headers with nonce
    headers = {}
    headers = inject_agent_headers(headers, agent, nonce=nonce)

    print("Headers with nonce:")
    for key, value in headers.items():
        display = value[:50] + "..." if len(value) > 50 else value
        print(f"  {key}: {display}")

    print("""
The server verifies:
1. X-Agent-Nonce is present
2. X-Agent-Nonce-Signature is valid signature of nonce
3. Nonce hasn't been used before (requires server-side tracking)

This prevents replay attacks where an attacker captures
and re-sends a valid request.
""")


async def run_live_demo():
    """Run a live demo with actual HTTP requests."""
    if not FASTAPI_AVAILABLE:
        print("\nSkipping live demo (FastAPI not installed)")
        return

    print("\n" + "=" * 60)
    print("Live Demo: Running Server and Making Requests")
    print("=" * 60)

    import httpx
    from agentfacts.middleware.headers import inject_agent_headers

    # Create the app
    app = create_server_app()

    # Start server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)

    # Run server in background task
    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(1)

    base_url = "http://127.0.0.1:8765"

    try:
        async with httpx.AsyncClient() as client:
            # Create agent for authenticated requests
            agent = AgentFacts(name="Demo Client")
            agent.sign()
            auth_headers = inject_agent_headers({}, agent)

            print("\n1. GET / (public)")
            resp = await client.get(f"{base_url}/")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")

            print("\n2. GET /health (public)")
            resp = await client.get(f"{base_url}/health")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")

            print("\n3. GET /public/data (anonymous)")
            resp = await client.get(f"{base_url}/public/data")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")

            print("\n4. GET /public/data (authenticated)")
            resp = await client.get(f"{base_url}/public/data", headers=auth_headers)
            print(f"   Status: {resp.status_code}")
            data = resp.json()
            print(f"   Verified: {data.get('verified')}")
            print(f"   Viewer: {data.get('viewer', 'N/A')[:50]}...")

            print("\n5. GET /secure/data (anonymous) - should fail")
            resp = await client.get(f"{base_url}/secure/data")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")

            print("\n6. GET /secure/data (authenticated)")
            resp = await client.get(f"{base_url}/secure/data", headers=auth_headers)
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")

    finally:
        # Shutdown server
        server.should_exit = True
        await server_task


def main():
    print("=" * 60)
    print("AgentFacts SDK - FastAPI Middleware Example")
    print("=" * 60)

    if not FASTAPI_AVAILABLE:
        print("\nFastAPI is not installed.")
        print("Install with: pip install agentfacts[middleware] uvicorn")
        print("\nShowing conceptual examples instead...")

    demonstrate_client_requests()
    demonstrate_strict_mode()
    demonstrate_with_nonce()

    # Run live demo if FastAPI is available
    if FASTAPI_AVAILABLE:
        print("\nRunning live demo...")
        asyncio.run(run_live_demo())

    print("\n" + "=" * 60)
    print("FastAPI middleware example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
