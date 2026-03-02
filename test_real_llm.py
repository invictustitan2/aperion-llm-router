#!/usr/bin/env python3
"""
Real LLM Integration Test.

Tests The Switchboard against real LLM providers using .env credentials.
"""

import asyncio
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx  # noqa: E402

from aperion_switchboard.core import LLMRouter, TaskType  # noqa: E402
from aperion_switchboard.providers import EchoProvider, WorkersAIProvider  # noqa: E402
from aperion_switchboard.providers.base import BaseProvider  # noqa: E402


async def test_workers_ai_direct():
    """Test Workers AI provider directly."""
    print("\n" + "=" * 60)
    print("TEST 1: Workers AI Provider - Direct Call")
    print("=" * 60)

    provider = WorkersAIProvider()

    print(f"Provider: {provider.name}")
    print(f"Configured: {provider.is_configured}")
    print(f"Base URL: {provider._base_url[:50]}..." if provider._base_url else "No URL")
    print(f"Model: {provider._model}")

    if not provider.is_configured:
        print("❌ SKIP: Workers AI not configured")
        return False

    # Set up shared async client
    async with httpx.AsyncClient(timeout=30.0) as client:
        BaseProvider.set_shared_client(client)

        prompt = "What is 2 + 2? Answer in one word."
        print(f"\nPrompt: {prompt}")

        try:
            result = await provider.async_generate(prompt)
            print("\n✅ SUCCESS!")
            print(f"Response: {result}")
            return True
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False


async def test_router_with_real_provider():
    """Test router selects and uses real provider."""
    print("\n" + "=" * 60)
    print("TEST 2: LLM Router - Task-Based Routing")
    print("=" * 60)

    # Create router and register available providers
    from aperion_switchboard.providers import GeminiProvider, OpenAIProvider, WorkersAIProvider

    router = LLMRouter()

    # Register all providers
    workers = WorkersAIProvider()
    if workers.is_configured:
        router.register_provider("workers_ai", workers)

    gemini = GeminiProvider()
    if gemini.is_configured:
        router.register_provider("gemini", gemini)

    openai = OpenAIProvider()
    if openai.is_configured:
        router.register_provider("openai", openai)

    echo = EchoProvider()
    router.register_provider("echo", echo)

    print(f"Registered providers: {list(router._providers.keys())}")

    # Check which providers are configured
    for name, provider in router._providers.items():
        status = "✅" if provider.is_configured else "❌"
        print(f"  {status} {name}: configured={provider.is_configured}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        BaseProvider.set_shared_client(client)

        # Test bulk task (should route to Workers/Gemini)
        task_type = TaskType.DOC_GENERATION
        provider, decision = router.get_provider(task_type)
        print(f"\nTask: {task_type.value}")
        print(f"Selected provider: {provider.name}")
        print(f"Routing reason: {decision.reason}")

        if not provider.is_configured:
            print("❌ SKIP: No configured provider for this task")
            return False

        prompt = "Summarize what an LLM is in one sentence."
        print(f"Prompt: {prompt}")

        try:
            result = await provider.async_generate(prompt)
            print("\n✅ SUCCESS!")
            print(f"Response: {result}")
            return True
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False


async def test_streaming():
    """Test streaming with real provider."""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming Response")
    print("=" * 60)

    provider = WorkersAIProvider()

    if not provider.is_configured:
        print("❌ SKIP: Workers AI not configured")
        return False

    async with httpx.AsyncClient(timeout=30.0) as client:
        BaseProvider.set_shared_client(client)

        prompt = "Count from 1 to 5."
        print(f"Prompt: {prompt}")
        print("Streaming: ", end="", flush=True)

        try:
            chunks = []
            async for chunk in provider.stream_generate(prompt):
                chunks.append(chunk)
                # Print partial content
                if isinstance(chunk, dict) and "content" in chunk:
                    print(chunk["content"], end="", flush=True)

            print(f"\n\n✅ SUCCESS! Received {len(chunks)} chunks")
            return True
        except NotImplementedError:
            print("\n⚠️  SKIP: Streaming not implemented for this provider")
            return True  # Not a failure - expected for some providers
        except Exception as e:
            print(f"\n\n❌ FAILED: {e}")
            return False


async def test_api_endpoint():
    """Test the FastAPI endpoint with real provider."""
    print("\n" + "=" * 60)
    print("TEST 4: FastAPI Endpoint (OpenAI-Compatible)")
    print("=" * 60)

    from fastapi.testclient import TestClient

    from aperion_switchboard.service.app import create_app

    app = create_app()

    # Use TestClient which properly handles lifespan events
    with TestClient(app) as client:
        # Non-streaming request
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 3 words."}
                ],
                "max_tokens": 50
            },
            headers={"X-Aperion-Task-Type": "doc_generation"}
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data.get('model')}")
            print(f"Provider: {response.headers.get('X-Switchboard-Provider')}")
            if data.get("choices"):
                content = data["choices"][0]["message"]["content"]
                print(f"Response: {content}")
            print("\n✅ SUCCESS!")
            return True
        else:
            print(f"Error: {response.text}")
            print("\n❌ FAILED")
            return False


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# THE SWITCHBOARD - Real LLM Integration Tests")
    print("#" * 60)

    results = []

    results.append(("Workers AI Direct", await test_workers_ai_direct()))
    results.append(("Router with Provider", await test_router_with_real_provider()))
    results.append(("Streaming", await test_streaming()))
    results.append(("FastAPI Endpoint", await test_api_endpoint()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL/SKIP"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
