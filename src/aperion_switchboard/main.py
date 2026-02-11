"""
The Switchboard - Main Entry Point.

Usage:
    # Development
    python -m aperion_switchboard.main

    # Production
    uvicorn aperion_switchboard.main:app --host 0.0.0.0 --port 8080

    # Using the CLI
    switchboard
"""

import os
import sys

import uvicorn

from .service.app import create_app

# Create app instance for uvicorn
app = create_app()


def main() -> None:
    """CLI entry point."""
    host = os.environ.get("SWITCHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("SWITCHBOARD_PORT", "8080"))
    reload = os.environ.get("SWITCHBOARD_RELOAD", "false").lower() == "true"

    print(f"🔌 Starting The Switchboard on {host}:{port}")
    print("   OpenAI-compatible endpoint: POST /v1/chat/completions")
    print("   Health check: GET /health")
    print("   API docs: GET /docs")
    print()

    try:
        uvicorn.run(
            "aperion_switchboard.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n🔌 The Switchboard stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
