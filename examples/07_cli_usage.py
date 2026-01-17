"""
Example 7: CLI Usage Guide

This file documents the AgentFacts CLI commands.
Run these commands in your terminal after installing agentfacts[cli].

Installation:
    pip install agentfacts[cli]
"""

CLI_GUIDE = """
================================================================================
AgentFacts CLI - Command Reference
================================================================================

INSTALLATION
------------
pip install agentfacts[cli]

COMMANDS OVERVIEW
-----------------
agentfacts --help              Show all commands
agentfacts --version           Show version

================================================================================
KEY MANAGEMENT
================================================================================

Generate a new Ed25519 key pair:
--------------------------------
# Print to console
agentfacts keys generate

# Save to file
agentfacts keys generate -o my_agent.pem

# Overwrite existing file
agentfacts keys generate -o my_agent.pem --force


Show key information:
---------------------
agentfacts keys show my_agent.pem

Output:
┌──────────────────────────────────────────────────────────────┐
│ Key Information                                               │
├─────────────────┬────────────────────────────────────────────┤
│ Property        │ Value                                      │
├─────────────────┼────────────────────────────────────────────┤
│ DID             │ did:key:z6Mk...                            │
│ Public Key      │ abc123...                                  │
│ Fingerprint     │ 1234abcd...                                │
└─────────────────┴────────────────────────────────────────────┘


================================================================================
AGENT METADATA
================================================================================

Create new agent metadata:
--------------------------
# Basic agent
agentfacts agent create \\
    --name "My Agent" \\
    --description "A helpful assistant"

# With model info
agentfacts agent create \\
    --name "GPT-4 Agent" \\
    --model "gpt-4-turbo" \\
    --provider "openai"

# With existing key (signed by default)
agentfacts agent create \\
    --name "Production Agent" \\
    --model "claude-3-opus" \\
    --provider "anthropic" \\
    --key my_agent.pem \\
    --output agent_metadata.json


Sign existing metadata:
-----------------------
agentfacts agent sign agent_metadata.json --key my_agent.pem

# Save to different file
agentfacts agent sign agent_metadata.json --key my_agent.pem --output signed.json


Verify agent metadata:
----------------------
agentfacts agent verify agent_metadata.json

Output:
✓ Signature is valid
  DID: did:key:z6Mk...


================================================================================
VERIFICATION
================================================================================

Verify against a policy:
------------------------
# Basic policy (just checks signature)
agentfacts verify agent_metadata.json --policy basic

# Strict enterprise policy
agentfacts verify agent_metadata.json --policy strict

Output:
┌──────────────────────────────────────────────────────────────┐
│ Verification Results                                          │
├─────────────────┬─────────────┬──────────────────────────────┤
│ Check           │ Status      │ Details                      │
├─────────────────┼─────────────┼──────────────────────────────┤
│ Signature       │ ✓ Valid     │ DID: did:key:z6Mk...         │
│ Policy (basic)  │ ✓ Passed    │ All rules satisfied          │
└─────────────────┴─────────────┴──────────────────────────────┘

Exit codes:
  0 = Verification passed
  1 = Verification failed


================================================================================
INSPECTION
================================================================================

Inspect agent metadata details:
-------------------------------
agentfacts inspect agent_metadata.json

Output:
╭──────────────────────────────────────────────────────────────╮
│ My Agent                                                      │
│ did:key:z6Mk...                                               │
╰──────────────────────────────────────────────────────────────╯

┌──────────────────────────────────────────────────────────────┐
│ Agent Information                                             │
├─────────────────┬────────────────────────────────────────────┤
│ Name            │ My Agent                                   │
│ Description     │ A helpful assistant                        │
│ Version         │ 1.0.0                                      │
│ DID             │ did:key:z6Mk...                            │
│ Signed          │ ✓ Yes                                      │
└─────────────────┴────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Baseline Model                                                │
├─────────────────┬────────────────────────────────────────────┤
│ Name            │ gpt-4-turbo                                │
│ Provider        │ openai                                     │
│ Temperature     │ 0.7                                        │
│ Max Tokens      │ 4096                                       │
└─────────────────┴────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Capabilities (3)                                              │
├─────────────────┬─────────────┬──────────────────────────────┤
│ Name            │ Risk        │ Description                  │
├─────────────────┼─────────────┼──────────────────────────────┤
│ web_search      │ low         │ Search the web               │
│ calculator      │ low         │ Do math                      │
│ file_reader     │ medium      │ Read files                   │
└─────────────────┴─────────────┴──────────────────────────────┘


================================================================================
BADGE GENERATION
================================================================================

Generate trust badges for documentation:
----------------------------------------
# Markdown badge (for GitHub README)
agentfacts badge agent_metadata.json --format markdown

Output:
![AgentFacts verified](https://img.shields.io/badge/AgentFacts-verified-brightgreen)

**Agent:** My Agent
**DID:** `did:key:z6Mk...`
**Model:** gpt-4-turbo
**Provider:** openai
**Capabilities:** 3 tools


# HTML badge
agentfacts badge agent_metadata.json --format html

# JSON data
agentfacts badge agent_metadata.json --format json

# With verification URL
agentfacts badge agent_metadata.json \\
    --format markdown \\
    --verify-url "https://verify.agentfacts.dev/agent/z6Mk..."


================================================================================
COMMON WORKFLOWS
================================================================================

1. Setting up a new agent:
--------------------------
# Generate key
agentfacts keys generate -o my_agent.pem

# Create metadata (signed by default)
agentfacts agent create \\
    --name "Production Agent" \\
    --model "gpt-4" \\
    --provider "openai" \\
    --key my_agent.pem \\
    --output agent.json

# Verify it worked
agentfacts verify agent.json --policy basic


2. CI/CD verification:
----------------------
#!/bin/bash
# In your CI pipeline
if agentfacts verify agent.json --policy strict; then
    echo "Agent verification passed"
    deploy_agent
else
    echo "Agent verification failed"
    exit 1
fi


3. Generate README badge:
-------------------------
# Add to your README.md
agentfacts badge agent.json --format markdown >> README.md


================================================================================
"""


def main():
    print(CLI_GUIDE)

    # Also demonstrate programmatic CLI usage
    print("=" * 60)
    print("Programmatic Example: Calling CLI from Python")
    print("=" * 60)

    import subprocess
    import sys

    print("\nRunning: agentfacts --version")
    result = subprocess.run(
        [sys.executable, "-m", "agentfacts.cli.main", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Output: {result.stdout.strip()}")
    else:
        print("(CLI requires agentfacts[cli] to be installed)")


if __name__ == "__main__":
    main()
