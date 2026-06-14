"""
NafidhAI — FastAPI Endpoint
Exposes POST /v1/agents/{agent_id}/run
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from models import AgentRunRequest, AgentRunResult, TriggerType, AgentRunStatus
from agent_runner import AgentRunner
from db_models import Base

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

def create_engine_from_env() -> Any:
    database_url = os.environ["DATABASE_URL"]

    # Railway (and most providers) give postgresql:// — convert to asyncpg driver
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

    if not database_url.startswith("postgresql+asyncpg://"):
        raise ValueError(
            "DATABASE_URL must use postgresql+asyncpg:// for async driver. "
            f"Got: {database_url[:30]}..."
        )

    return create_async_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

engine = create_engine_from_env()
session_factory = async_sessionmaker(engine, expire_on_commit=False)
agent_runner = AgentRunner(session_factory=session_factory)

app = FastAPI(
    title="NafidhAI Agent Runtime API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","),
)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

class RunAgentRequest(AgentRunRequest):
    pass


@app.post(
    "/v1/agents/{agent_id}/run",
    response_model=AgentRunResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger an agent run",
)
async def run_agent(
    agent_id: uuid.UUID,
    body: RunAgentRequest,
    request: Request,
) -> AgentRunResult:
    """
    Body mein tenant_id aur agent_definition_id dono required hain.
    """
    log = logger.bind(
        tenant_id=str(body.tenant_id),
        agent_id=str(agent_id),
        request_id=request.headers.get("X-Request-ID", "unknown"),
    )

    if body.agent_definition_id != agent_id:
        log.warning("agent_id_mismatch",
                    path_agent_id=str(agent_id),
                    body_agent_id=str(body.agent_definition_id))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="agent_id in path must match agent_definition_id in request body",
        )

    log.info("agent_run_request_received")
    result = await agent_runner.run(body)

    if result.status in (AgentRunStatus.FAILED, AgentRunStatus.TIMEOUT):
        log.warning("agent_run_returned_failure",
                    status=result.status.value,
                    error_code=result.error_code)

    return result


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "service": "nafidhAI-agent-runtime"}


@app.get("/ready")
async def readiness_check() -> dict:
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ready", "db": "connected"}
    except Exception as exc:
        logger.error("readiness_check_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed",
        )
