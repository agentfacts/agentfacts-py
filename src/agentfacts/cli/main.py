"""
Main CLI entry point for AgentFacts SDK.

Provides commands for:
- Generating and managing keys
- Creating and signing agent metadata
- Generating trust badges
- Verifying agent credentials
"""

# mypy: disable-error-code=untyped-decorator

import json
import sys
from pathlib import Path
from typing import Any

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

cli: Any


if CLI_AVAILABLE:
    console = Console()

    # Import version dynamically
    from agentfacts import __version__

    @click.group()
    @click.version_option(version=__version__, prog_name="agentfacts")
    def cli() -> None:
        """AgentFacts CLI - The SSL of the Agentic Web.

        Generate verifiable identity for AI agents.
        """
        pass

    # -------------------------------------------------------------------------
    # Key Management
    # -------------------------------------------------------------------------

    @cli.group()
    def keys() -> None:
        """Manage Ed25519 key pairs."""
        pass

    @keys.command("generate")
    @click.option(
        "--output", "-o", type=click.Path(), help="Output file for private key (PEM)"
    )
    @click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
    def keys_generate(output: str | None, force: bool) -> None:
        """Generate a new Ed25519 key pair."""
        from agentfacts.crypto.did import DID
        from agentfacts.crypto.keys import KeyPair

        key_pair = KeyPair.generate()
        did = DID.from_key_pair(key_pair)

        if output:
            path = Path(output)
            if path.exists() and not force:
                console.print(
                    f"[red]Error:[/red] File {path} already exists. Use --force to overwrite."
                )
                sys.exit(1)
            key_pair.save(path)
            console.print(f"[green]✓[/green] Private key saved to: {path}")
        else:
            console.print(
                Panel(
                    key_pair.to_pem().decode(),
                    title="Private Key (PEM)",
                    subtitle="Save this securely!",
                )
            )

        console.print()
        console.print(f"[bold]DID:[/bold] {did.uri}")
        console.print(f"[bold]Public Key:[/bold] {key_pair.public_key_base64}")

    @keys.command("show")
    @click.argument("keyfile", type=click.Path(exists=True))
    def keys_show(keyfile: str) -> None:
        """Show public key and DID from a private key file."""
        from agentfacts.crypto.did import DID
        from agentfacts.crypto.keys import KeyPair

        key_pair = KeyPair.from_file(keyfile)
        did = DID.from_key_pair(key_pair)

        table = Table(title="Key Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("DID", did.uri)
        table.add_row("Public Key (base64)", key_pair.public_key_base64)
        table.add_row("Fingerprint", did.fingerprint())

        console.print(table)

    # -------------------------------------------------------------------------
    # Agent Metadata
    # -------------------------------------------------------------------------

    @cli.group()
    def agent() -> None:
        """Manage agent metadata."""
        pass

    @agent.command("create")
    @click.option("--name", "-n", required=True, help="Agent name")
    @click.option("--description", "-d", default="", help="Agent description")
    @click.option("--model", "-m", default="unknown", help="Model name")
    @click.option("--provider", "-p", default="unknown", help="Model provider")
    @click.option("--key", "-k", type=click.Path(exists=True), help="Private key file")
    @click.option(
        "--output", "-o", type=click.Path(), help="Output file for metadata (JSON)"
    )
    def agent_create(
        name: str,
        description: str,
        model: str,
        provider: str,
        key: str | None,
        output: str | None,
    ) -> None:
        """Create new agent metadata."""
        from agentfacts.core import AgentFacts
        from agentfacts.crypto.keys import KeyPair
        from agentfacts.models import BaselineModel, ModelProvider

        # Load or generate key
        key_pair = KeyPair.from_file(key) if key else KeyPair.generate()

        # Map provider string to enum
        try:
            provider_enum = ModelProvider(provider.lower())
        except ValueError:
            provider_enum = ModelProvider.UNKNOWN

        # Create agent facts
        facts = AgentFacts(
            name=name,
            description=description,
            baseline_model=BaselineModel(name=model, provider=provider_enum),
            key_pair=key_pair,
        )

        facts.sign()
        console.print("[green]✓[/green] Metadata signed")

        # Output
        json_output = facts.to_json()

        if output:
            Path(output).write_text(json_output)
            console.print(f"[green]✓[/green] Metadata saved to: {output}")
        else:
            syntax = Syntax(json_output, "json", theme="monokai")
            console.print(syntax)

        console.print()
        console.print(f"[bold]DID:[/bold] {facts.did}")

    @agent.command("sign")
    @click.argument("metadata", type=click.Path(exists=True))
    @click.option(
        "--key",
        "-k",
        required=True,
        type=click.Path(exists=True),
        help="Private key file",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file (default: overwrite input)",
    )
    def agent_sign(metadata: str, key: str, output: str | None) -> None:
        """Sign agent metadata."""
        from agentfacts.core import AgentFacts
        from agentfacts.crypto.keys import KeyPair

        key_pair = KeyPair.from_file(key)
        data = json.loads(Path(metadata).read_text())

        facts = AgentFacts.from_dict(data, key_pair=key_pair)
        facts.sign(key_pair)

        output_path = output or metadata
        Path(output_path).write_text(facts.to_json())

        console.print(f"[green]✓[/green] Metadata signed and saved to: {output_path}")

    @agent.command("verify")
    @click.argument("metadata", type=click.Path(exists=True))
    def agent_verify(metadata: str) -> None:
        """Verify agent metadata signature."""
        from agentfacts.core import AgentFacts

        data = json.loads(Path(metadata).read_text())
        facts = AgentFacts.from_dict(data)

        result = facts.verify()

        if result.valid:
            console.print("[green]✓[/green] Signature is valid")
            console.print(f"  DID: {result.did}")
        else:
            console.print("[red]✗[/red] Signature verification failed")
            for error in result.errors:
                console.print(f"  [red]•[/red] {error}")

        if result.warnings:
            for warning in result.warnings:
                console.print(f"  [yellow]•[/yellow] {warning}")

    # -------------------------------------------------------------------------
    # Badge Generation
    # -------------------------------------------------------------------------

    @cli.command("badge")
    @click.argument("metadata", type=click.Path(exists=True))
    @click.option(
        "--format",
        "-f",
        type=click.Choice(["markdown", "html", "json"]),
        default="markdown",
    )
    @click.option("--verify-url", "-u", default=None, help="URL to verification page")
    def badge(metadata: str, format: str, verify_url: str | None) -> None:
        """Generate a trust badge for your agent.

        Creates a dynamic badge snippet for GitHub READMEs
        or documentation sites.
        """
        from agentfacts.core import AgentFacts
        from agentfacts.crypto.did import DID

        data = json.loads(Path(metadata).read_text())
        facts = AgentFacts.from_dict(data)

        # Verify signature
        result = facts.verify()
        is_verified = result.valid

        # Generate badge
        status = "verified" if is_verified else "unverified"
        color = "brightgreen" if is_verified else "red"
        short_did = DID.parse(facts.did).short_id(12)

        if format == "markdown":
            badge_url = f"https://img.shields.io/badge/AgentFacts-{status}-{color}"
            if verify_url:
                output = f"[![AgentFacts {status}]({badge_url})]({verify_url})"
            else:
                output = f"![AgentFacts {status}]({badge_url})"

            output += f"\n\n**Agent:** {facts.name}\n"
            output += f"**DID:** `{facts.did}`\n"
            output += f"**Model:** {facts.metadata.agent.model.name}\n"
            output += f"**Provider:** {facts.metadata.agent.model.provider.value}\n"

            if facts.metadata.agent.capabilities:
                output += f"**Capabilities:** {len(facts.metadata.agent.capabilities)} tools\n"

        elif format == "html":
            badge_url = f"https://img.shields.io/badge/AgentFacts-{status}-{color}"
            output = f'<a href="{verify_url or "#"}">'
            output += f'<img src="{badge_url}" alt="AgentFacts {status}">'
            output += "</a>"

        else:  # json
            output = json.dumps(
                {
                    "status": status,
                    "verified": is_verified,
                    "did": facts.did,
                    "short_did": short_did,
                    "name": facts.name,
                    "model": facts.metadata.agent.model.name,
                    "provider": facts.metadata.agent.model.provider.value,
                    "capabilities_count": len(facts.metadata.agent.capabilities),
                    "badge_url": f"https://img.shields.io/badge/AgentFacts-{status}-{color}",
                },
                indent=2,
            )

        console.print(output)

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    @cli.command("verify")
    @click.argument("metadata", type=click.Path(exists=True))
    @click.option(
        "--policy", "-p", type=click.Choice(["basic", "strict"]), default="basic"
    )
    def verify(metadata: str, policy: str) -> None:
        """Verify agent metadata against a policy."""
        from agentfacts.core import AgentFacts
        from agentfacts.policy.rules import Policy

        data = json.loads(Path(metadata).read_text())
        facts = AgentFacts.from_dict(data)

        # Verify signature
        sig_result = facts.verify()

        # Evaluate policy
        policy_obj = (
            Policy.strict_enterprise() if policy == "strict" else Policy.basic_trust()
        )

        policy_result = policy_obj.evaluate(facts.metadata)

        # Display results
        table = Table(title="Verification Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")

        # Signature check
        if sig_result.valid:
            table.add_row(
                "Signature", "[green]✓ Valid[/green]", f"DID: {sig_result.did}"
            )
        else:
            table.add_row(
                "Signature", "[red]✗ Invalid[/red]", "; ".join(sig_result.errors)
            )

        # Policy check
        if policy_result.passed:
            table.add_row(
                f"Policy ({policy})", "[green]✓ Passed[/green]", "All rules satisfied"
            )
        else:
            violations = "; ".join(str(v) for v in policy_result.violations)
            table.add_row(f"Policy ({policy})", "[red]✗ Failed[/red]", violations)

        console.print(table)

        # Exit code
        if sig_result.valid and policy_result.passed:
            sys.exit(0)
        else:
            sys.exit(1)

    @cli.command("inspect")
    @click.argument("metadata", type=click.Path(exists=True))
    def inspect(metadata: str) -> None:
        """Inspect agent metadata details."""
        from agentfacts.core import AgentFacts

        data = json.loads(Path(metadata).read_text())
        facts = AgentFacts.from_dict(data)

        console.print(Panel(f"[bold]{facts.name}[/bold]", subtitle=facts.did))

        # Basic info
        table = Table(title="Agent Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Name", facts.metadata.agent.name)
        table.add_row("Description", facts.metadata.agent.description or "(none)")
        table.add_row("Version", facts.metadata.agent.version)
        table.add_row("DID", facts.did)
        table.add_row(
            "Signed", "[green]Yes[/green]" if facts.is_signed else "[red]No[/red]"
        )

        console.print(table)

        # Model info
        model = facts.metadata.agent.model
        model_table = Table(title="Baseline Model")
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value")

        model_table.add_row("Name", model.name)
        model_table.add_row("Provider", model.provider.value)
        model_table.add_row(
            "Temperature", str(model.temperature) if model.temperature else "(default)"
        )
        model_table.add_row(
            "Max Tokens", str(model.max_tokens) if model.max_tokens else "(default)"
        )

        console.print(model_table)

        # Capabilities
        if facts.metadata.agent.capabilities:
            cap_table = Table(
                title=f"Capabilities ({len(facts.metadata.agent.capabilities)})"
            )
            cap_table.add_column("Name", style="cyan")
            cap_table.add_column("Risk", style="bold")
            cap_table.add_column("Description")

            for cap in facts.metadata.agent.capabilities:
                risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
                    cap.risk_level or "", "white"
                )
                cap_table.add_row(
                    cap.name,
                    f"[{risk_color}]{cap.risk_level or 'unknown'}[/{risk_color}]",
                    (
                        cap.description[:50] + "..."
                        if len(cap.description) > 50
                        else cap.description
                    ),
                )

            console.print(cap_table)

        # Attestations
        if facts.metadata.attestations:
            att_table = Table(
                title=f"Attestations ({len(facts.metadata.attestations)})"
            )
            att_table.add_column("Type", style="cyan")
            att_table.add_column("Issuer")
            att_table.add_column("Date")

            for att in facts.metadata.attestations:
                att_table.add_row(
                    att.type,
                    att.issuer[:30] + "..." if len(att.issuer) > 30 else att.issuer,
                    att.issued_at.strftime("%Y-%m-%d"),
                )

            console.print(att_table)

else:

    def _cli_unavailable() -> None:
        """CLI not available - install with: pip install agentfacts[cli]"""
        print("CLI tools require additional dependencies.")
        print("Install with: pip install agentfacts[cli]")
        sys.exit(1)

    cli = _cli_unavailable


if __name__ == "__main__":
    cli()
