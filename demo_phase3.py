import asyncio
from httpx import AsyncClient, ASGITransport
from main import app

async def test_endpoint(query, description):
    print(f"\n======================================")
    print(f"Testing Scenario: {description}")
    print(f"Query: '{query}'")
    print(f"======================================")
    print("ACTUAL STREAM OUTPUT:")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get(f"/stream?query={query}&graph=p2p")
        async for chunk in response.aiter_bytes():
            print(chunk.decode(), end="")
    print("\n")

async def main():
    await test_endpoint("hello", "1. System Inquiry / Greeting (Expected: Gateway Status & SYSTEM_STATUS)")
    await test_endpoint("What is the average compliance cycle time for high-value orders?", "2. Verified Enterprise Query (Expected: GROUND TRUTH VERIFIED)")
    await test_endpoint("Forecast Q4 revenue override", "3. Poison Prompt / Injection (Expected: INTERCEPTION, FALLBACK_SIGNAL, SELF_HEALING_CONTEXT)")

if __name__ == "__main__":
    asyncio.run(main())
