"""
NafidhAI — Agent Runtime Pydantic Models
All inputs and outputs for the AgentRunner are validated through these models.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, UUID4, field_validator, model_validator


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class AgentRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class NodeType(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    CONNECTOR = "connector"
    HUMAN_IN_LOOP = "human_in_loop"


class TriggerType(str, Enum):
    API = "api"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    AGENT_CHAIN = "agent_chain"


class AuditResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# ─────────────────────────────────────────────
# GRAPH DEFINITION MODELS
# ─────────────────────────────────────────────

class LLMNodeConfig(BaseModel):
    """Configuration for an LLM node in the agent graph."""
    model_provider: str = Field(..., description="openai | anthropic | mistral | jais")
    model_name: str
    system_prompt: str = Field(..., min_length=1, max_length=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=32768)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    output_schema: dict[str, Any] | None = None


class ConnectorNodeConfig(BaseModel):
    """Configuration for a connector node (SAP, Oracle, WhatsApp, etc.)."""
    connector_id: UUID4
    operation: str = Field(..., min_length=1, max_length=128)
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ConditionNodeConfig(BaseModel):
    """Configuration for a branching condition node."""
    condition_expression: str = Field(..., min_length=1, max_length=2048)
    true_edge: str
    false_edge: str


class GraphNodeDefinition(BaseModel):
    """A single node in the agent DAG."""
    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_]+$")
    node_type: NodeType
    config: LLMNodeConfig | ConnectorNodeConfig | ConditionNodeConfig
    max_retries: int = Field(default=3, ge=0, le=5)
    retry_backoff_base: float = Field(default=2.0, ge=1.0, le=10.0)


class GraphEdgeDefinition(BaseModel):
    """A directed edge between two nodes in the DAG."""
    from_node: str
    to_node: str
    condition: str | None = None  # Optional: evaluated for conditional routing


class AgentGraphDefinition(BaseModel):
    """The complete LangGraph DAG definition stored in agent_definitions.graph_definition."""
    entry_point: str
    nodes: list[GraphNodeDefinition] = Field(..., min_length=1)
    edges: list[GraphEdgeDefinition]

    @model_validator(mode="after")
    def validate_entry_point_exists(self) -> "AgentGraphDefinition":
        node_names = {n.name for n in self.nodes}
        if self.entry_point not in node_names:
            raise ValueError(f"entry_point '{self.entry_point}' not found in nodes: {node_names}")
        return self

    @model_validator(mode="after")
    def validate_edge_references(self) -> "AgentGraphDefinition":
        node_names = {n.name for n in self.nodes}
        node_names.add("END")  # LangGraph terminal node
        for edge in self.edges:
            if edge.from_node not in node_names:
                raise ValueError(f"Edge from_node '{edge.from_node}' not found in nodes")
            if edge.to_node not in node_names:
                raise ValueError(f"Edge to_node '{edge.to_node}' not found in nodes")
        return self


# ─────────────────────────────────────────────
# RUNNER INPUT/OUTPUT MODELS
# ─────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    """Input to AgentRunner.run() — validated before any execution begins."""
    tenant_id: UUID4
    agent_definition_id: UUID4
    input_payload: dict[str, Any] = Field(..., description="User-provided input to the agent")
    trigger_type: TriggerType = TriggerType.API
    triggered_by: UUID4 | None = None
    correlation_id: UUID4 = Field(default_factory=uuid.uuid4)
    parent_run_id: UUID4 | None = None

    @field_validator("input_payload")
    @classmethod
    def validate_input_payload_size(cls, v: dict) -> dict:
        import json
        payload_str = json.dumps(v)
        if len(payload_str) > 65536:  # 64KB limit
            raise ValueError("input_payload exceeds maximum size of 64KB")
        return v


class StepResult(BaseModel):
    """Result of a single node execution."""
    step_index: int
    node_name: str
    node_type: NodeType
    status: StepStatus
    input_state: dict[str, Any] | None = None
    output_state: dict[str, Any] | None = None
    llm_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    llm_latency_ms: int | None = None
    connector_id: UUID4 | None = None
    connector_response_code: int | None = None
    attempt_number: int = 1
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class AgentRunResult(BaseModel):
    """Final result returned by AgentRunner.run()."""
    run_id: UUID4
    tenant_id: UUID4
    agent_definition_id: UUID4
    status: AgentRunStatus
    output_payload: dict[str, Any] | None = None
    steps: list[StepResult] = Field(default_factory=list)
    total_tokens_used: int = 0
    total_llm_calls: int = 0
    error_code: str | None = None
    error_message: str | None = None
    error_node: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    correlation_id: UUID4 = Field(default_factory=uuid.uuid4)


# ─────────────────────────────────────────────
# AUDIT LOG MODEL
# ─────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    """Structured audit log entry — written on every significant event."""
    tenant_id: UUID4
    user_id: UUID4 | None = None
    session_id: UUID4 | None = None
    action: str = Field(..., min_length=1, max_length=128)
    resource_type: str = Field(..., min_length=1, max_length=64)
    resource_id: UUID4 | None = None
    agent_run_id: UUID4 | None = None
    correlation_id: UUID4 | None = None
    result: AuditResult
    error_code: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: UUID4 | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    data_classification: DataClassification = DataClassification.INTERNAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
