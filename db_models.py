"""
NafidhAI — SQLAlchemy ORM Models
Database table definitions used by AgentRunner.
"""
from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped


class Base(DeclarativeBase):
    pass


class TenantORM(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_ar: Mapped[str | None] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    region: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class AgentDefinitionORM(Base):
    __tablename__ = "agent_definitions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_ar: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    description_ar: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    graph_definition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    llm_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    allowed_connector_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=[])
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)


class AgentRunORM(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    agent_definition_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    input_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_node: Mapped[str | None] = mapped_column(Text)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False)
    triggered_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    triggered_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[object | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True))
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class AgentStepORM(Base):
    __tablename__ = "agent_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(Text, nullable=False)
    node_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    input_state: Mapped[dict | None] = mapped_column(JSONB)
    output_state: Mapped[dict | None] = mapped_column(JSONB)
    llm_model: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer)
    connector_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    connector_response_code: Mapped[int | None] = mapped_column(Integer)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True))


class AuditLogORM(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    resource_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    correlation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(Text)
    ip_address: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    log_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default={})
    data_classification: Mapped[str] = mapped_column(String(16), nullable=False, default="internal")
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)