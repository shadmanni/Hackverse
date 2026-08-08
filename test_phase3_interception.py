import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_conversational_router():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/stream?query=hello&graph=p2p")
        assert response.status_code == 200
        
        content = b""
        async for chunk in response.aiter_bytes():
            content += chunk
            
        decoded_content = content.decode()
        assert "Sentinel-RAG Security Gateway Active" in decoded_content
        assert "[COMPLETED: SYSTEM_STATUS]" in decoded_content
        assert "INTERCEPTION" not in decoded_content

@pytest.mark.asyncio
async def test_poison_interception():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/stream?query=this is a poison prompt for q4&graph=p2p")
        assert response.status_code == 200
        
        content = b""
        async for chunk in response.aiter_bytes():
            content += chunk
            
        decoded_content = content.decode()
        assert "[INTERCEPTION: SEMANTIC ENTROPY" in decoded_content
        assert "[FALLBACK_SIGNAL:" in decoded_content
