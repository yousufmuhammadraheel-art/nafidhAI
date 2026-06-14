"""
NafidhAI — AgentRunner
Core agent execution engine.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models import (
    AgentGraphDefinition,
    AgentRunRequest,
    AgentRunResult,
    AgentRunStatus,
    AuditLogEntry,
    AuditResult,
    ConnectorNodeConfig,
    DataClassification,
    GraphNodeDefinition,
    LLMNodeConfig,
    NodeType,
    StepResult,
    StepStatus,
)
from db_models import AgentDefinitionORM, AgentRunORM, AgentStepORM, AuditLogORM

logger = structlog.get_logger(__name__)


class AgentDefinitionNotFoundError(Exception):
    pass

class AgentDefinitionInactiveError(Exception):
    pass

class NodeExecutionError(Exception):
    def __init__(self, node_name: str, original_error: Exception) -> None:
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed after all retries: {original_error}")

class WorkflowTimeoutError(Exception):
    pass

class TenantIsolationError(Exception):
    pass


class LLMFactory:
    @staticmethod
    def create(config: LLMNodeConfig) -> Any:
        if config.model_provider == "openai":
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                request_timeout=config.timeout_seconds,
            )
        elif config.model_provider == "anthropic":
            return ChatAnthropic(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
            )
        elif config.model_provider == "groq":
            return ChatGroq(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported model_provider '{config.model_provider}'. "
                f"Supported: openai, anthropic, groq"
            )


class AgentRunner:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        run_id = uuid.uuid4()
        started_at = datetime.now(timezone.utc)
        agent_def = None

        log = logger.bind(
            run_id=str(run_id),
            tenant_id=str(request.tenant_id),
            agent_definition_id=str(request.agent_definition_id),
            correlation_id=str(request.correlation_id),
        )

        log.info("agent_run_started", trigger_type=request.trigger_type.value)

        await self._write_audit_log(AuditLogEntry(
            tenant_id=request.tenant_id,
            user_id=request.triggered_by,
            action="AGENT_RUN_STARTED",
            resource_type="agent_run",
            resource_id=run_id,
            agent_run_id=run_id,
            correlation_id=request.correlation_id,
            result=AuditResult.PARTIAL,
            metadata={
                "trigger_type": request.trigger_type.value,
                "agent_definition_id": str(request.agent_definition_id),
            },
            data_classification=DataClassification.INTERNAL,
        ))

        try:
            agent_def, graph_def = await self._load_agent_definition(
                agent_definition_id=request.agent_definition_id,
                tenant_id=request.tenant_id,
            )

            await self._create_agent_run(request, run_id, started_at)

            result = await asyncio.wait_for(
                self._execute_workflow(
                    run_id=run_id,
                    request=request,
                    graph_def=graph_def,
                    log=log,
                ),
                timeout=agent_def.timeout_seconds,
            )

            completed_at = datetime.now(timezone.utc)
            await self._finalize_run(
                run_id=run_id,
                tenant_id=request.tenant_id,
                status=AgentRunStatus.COMPLETED,
                output_payload=result.output_payload,
                total_tokens=result.total_tokens_used,
                total_llm_calls=result.total_llm_calls,
                completed_at=completed_at,
            )

            await self._write_audit_log(AuditLogEntry(
                tenant_id=request.tenant_id,
                user_id=request.triggered_by,
                action="AGENT_RUN_COMPLETED",
                resource_type="agent_run",
                resource_id=run_id,
                agent_run_id=run_id,
                correlation_id=request.correlation_id,
                result=AuditResult.SUCCESS,
                metadata={
                    "total_steps": len(result.steps),
                    "total_tokens": result.total_tokens_used,
                    "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
                },
            ))

            log.info("agent_run_completed",
                     total_steps=len(result.steps),
                     total_tokens=result.total_tokens_used)
            return result

        except asyncio.TimeoutError:
            timeout_val = agent_def.timeout_seconds if agent_def else "unknown"
            log.warning("agent_run_timeout", timeout_seconds=timeout_val)
            return await self._handle_run_failure(
                run_id=run_id, request=request,
                error_code="WORKFLOW_TIMEOUT",
                error_message="Workflow exceeded timeout",
                error_node=None, status=AgentRunStatus.TIMEOUT,
                started_at=started_at, steps=[],
            )

        except AgentDefinitionNotFoundError as exc:
            log.error("agent_definition_not_found", error=str(exc))
            return await self._handle_run_failure(
                run_id=run_id, request=request,
                error_code="AGENT_DEFINITION_NOT_FOUND",
                error_message=str(exc),
                error_node=None, status=AgentRunStatus.FAILED,
                started_at=started_at, steps=[],
            )

        except NodeExecutionError as exc:
            log.error("node_execution_failed",
                      node_name=exc.node_name,
                      original_error=str(exc.original_error))
            return await self._handle_run_failure(
                run_id=run_id, request=request,
                error_code="NODE_EXECUTION_FAILED",
                error_message=str(exc.original_error),
                error_node=exc.node_name, status=AgentRunStatus.FAILED,
                started_at=started_at, steps=[],
            )

        except TenantIsolationError:
            log.critical("tenant_isolation_violation_detected",
                         requested_agent=str(request.agent_definition_id))
            await self._write_audit_log(AuditLogEntry(
                tenant_id=request.tenant_id,
                user_id=request.triggered_by,
                action="TENANT_ISOLATION_VIOLATION",
                resource_type="agent_definition",
                resource_id=request.agent_definition_id,
                agent_run_id=run_id,
                correlation_id=request.correlation_id,
                result=AuditResult.FAILURE,
                error_code="TENANT_ISOLATION_VIOLATION",
                metadata={"severity": "critical"},
                data_classification=DataClassification.RESTRICTED,
            ))
            return AgentRunResult(
                run_id=run_id,
                tenant_id=request.tenant_id,
                agent_definition_id=request.agent_definition_id,
                status=AgentRunStatus.FAILED,
                error_code="FORBIDDEN",
                error_message="Access denied",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                correlation_id=request.correlation_id,
            )

        except Exception as exc:
            log.exception("agent_run_unexpected_error", error=str(exc))
            return await self._handle_run_failure(
                run_id=run_id, request=request,
                error_code="INTERNAL_ERROR",
                error_message="An unexpected error occurred",
                error_node=None, status=AgentRunStatus.FAILED,
                started_at=started_at, steps=[],
            )

    async def _load_agent_definition(
        self,
        agent_definition_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> tuple[AgentDefinitionORM, AgentGraphDefinition]:
        async with self._session_factory() as session:
            stmt = select(AgentDefinitionORM).where(
                AgentDefinitionORM.id == agent_definition_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                raise AgentDefinitionNotFoundError(
                    f"AgentDefinition {agent_definition_id} not found"
                )
            if str(row.tenant_id) != str(tenant_id):
                raise TenantIsolationError(
                    f"AgentDefinition {agent_definition_id} does not belong to tenant {tenant_id}"
                )
            if not row.is_active:
                raise AgentDefinitionInactiveError(
                    f"AgentDefinition {agent_definition_id} is not active"
                )

            graph_def = AgentGraphDefinition.model_validate(row.graph_definition)
            return row, graph_def

    async def _execute_workflow(
        self,
        run_id: uuid.UUID,
        request: AgentRunRequest,
        graph_def: AgentGraphDefinition,
        log: Any,
    ) -> AgentRunResult:
        steps: list[StepResult] = []
        total_tokens = 0
        total_llm_calls = 0

        node_executors = self._build_node_executors(
            nodes=graph_def.nodes,
            run_id=run_id,
            request=request,
            steps=steps,
        )

        compiled_graph = self._compile_graph(graph_def, node_executors)

        initial_state = {
            "input": request.input_payload,
            "run_id": str(run_id),
            "tenant_id": str(request.tenant_id),
            "steps": [],
            "output": None,
        }

        log.info("langgraph_execution_starting",
                 entry_point=graph_def.entry_point,
                 node_count=len(graph_def.nodes))

        final_state = await compiled_graph.ainvoke(
            initial_state,
            config=RunnableConfig(recursion_limit=50),
        )

        for step in steps:
            if step.prompt_tokens:
                total_tokens += step.prompt_tokens
            if step.completion_tokens:
                total_tokens += step.completion_tokens
            if step.node_type == NodeType.LLM:
                total_llm_calls += 1

        return AgentRunResult(
            run_id=run_id,
            tenant_id=request.tenant_id,
            agent_definition_id=request.agent_definition_id,
            status=AgentRunStatus.COMPLETED,
            output_payload=final_state.get("output"),
            steps=steps,
            total_tokens_used=total_tokens,
            total_llm_calls=total_llm_calls,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            correlation_id=request.correlation_id,
        )

    def _compile_graph(
        self,
        graph_def: AgentGraphDefinition,
        node_executors: dict[str, Any],
    ) -> Any:
        from typing import TypedDict

        class AgentState(TypedDict):
            input: dict
            run_id: str
            tenant_id: str
            steps: list
            output: dict | None

        workflow = StateGraph(AgentState)

        for node in graph_def.nodes:
            workflow.add_node(node.name, node_executors[node.name])

        for edge in graph_def.edges:
            if edge.to_node == "END":
                workflow.add_edge(edge.from_node, END)
            elif edge.condition:
                workflow.add_conditional_edges(
                    edge.from_node,
                    lambda state, cond=edge.condition: self._evaluate_condition(state, cond),
                    {edge.to_node: edge.to_node, "END": END},
                )
            else:
                workflow.add_edge(edge.from_node, edge.to_node)

        workflow.set_entry_point(graph_def.entry_point)
        return workflow.compile()

    def _evaluate_condition(self, state: dict, condition_expression: str) -> str:
        safe_globals: dict[str, Any] = {"__builtins__": {}}
        safe_locals = {"state": state, "output": state.get("output", {})}
        try:
            result = eval(condition_expression, safe_globals, safe_locals)  # noqa: S307
            return "true" if result else "false"
        except Exception as exc:
            logger.error("condition_evaluation_failed",
                         expression=condition_expression, error=str(exc))
            return "false"

    def _build_node_executors(
        self,
        nodes: list[GraphNodeDefinition],
        run_id: uuid.UUID,
        request: AgentRunRequest,
        steps: list[StepResult],
    ) -> dict[str, Any]:
        executors: dict[str, Any] = {}
        step_index_counter = {"value": 0}
        for node_def in nodes:
            executors[node_def.name] = self._build_single_node_executor(
                node_def=node_def, run_id=run_id,
                request=request, steps=steps,
                step_index_counter=step_index_counter,
            )
        return executors

    def _build_single_node_executor(
        self,
        node_def: GraphNodeDefinition,
        run_id: uuid.UUID,
        request: AgentRunRequest,
        steps: list[StepResult],
        step_index_counter: dict,
    ):
        async def execute_node(state: dict) -> dict:
            step_index = step_index_counter["value"]
            step_index_counter["value"] += 1

            node_log = logger.bind(
                run_id=str(run_id),
                tenant_id=str(request.tenant_id),
                node_name=node_def.name,
                node_type=node_def.node_type.value,
                step_index=step_index,
                attempt=1,
            )

            step = StepResult(
                step_index=step_index,
                node_name=node_def.name,
                node_type=node_def.node_type,
                status=StepStatus.RUNNING,
                input_state=state,
                started_at=datetime.now(timezone.utc),
            )

            await self._write_agent_step(
                run_id=run_id, tenant_id=request.tenant_id,
                step=step, attempt=1, max_attempts=node_def.max_retries + 1,
            )

            await self._write_audit_log(AuditLogEntry(
                tenant_id=request.tenant_id,
                user_id=request.triggered_by,
                action="NODE_EXECUTION_STARTED",
                resource_type="agent_step",
                agent_run_id=run_id,
                correlation_id=request.correlation_id,
                result=AuditResult.PARTIAL,
                metadata={
                    "node_name": node_def.name,
                    "node_type": node_def.node_type.value,
                    "step_index": step_index,
                },
            ))

            last_error: Exception | None = None
            for attempt in range(1, node_def.max_retries + 2):
                node_log = node_log.bind(attempt=attempt)
                node_log.info("node_attempt_starting")

                try:
                    timeout = self._get_node_timeout(node_def)
                    output_state = await asyncio.wait_for(
                        self._execute_node_by_type(
                            node_def=node_def, state=state,
                            step=step, run_id=run_id, request=request,
                        ),
                        timeout=timeout,
                    )

                    step.status = StepStatus.COMPLETED
                    step.output_state = output_state
                    step.completed_at = datetime.now(timezone.utc)
                    step.attempt_number = attempt
                    steps.append(step)

                    await self._update_agent_step(step)
                    await self._write_audit_log(AuditLogEntry(
                        tenant_id=request.tenant_id,
                        user_id=request.triggered_by,
                        action="NODE_EXECUTION_COMPLETED",
                        resource_type="agent_step",
                        agent_run_id=run_id,
                        correlation_id=request.correlation_id,
                        result=AuditResult.SUCCESS,
                        metadata={
                            "node_name": node_def.name,
                            "step_index": step_index,
                            "attempt": attempt,
                            "duration_ms": int(
                                (step.completed_at - step.started_at).total_seconds() * 1000
                            ) if step.completed_at else 0,
                        },
                    ))

                    node_log.info("node_execution_completed", attempt=attempt)
                    return {**state, **output_state}

                except asyncio.TimeoutError:
                    last_error = asyncio.TimeoutError(
                        f"Node '{node_def.name}' timed out after {timeout}s"
                    )
                    node_log.warning("node_timeout", timeout_seconds=timeout, attempt=attempt)
                    if attempt <= node_def.max_retries:
                        await asyncio.sleep(node_def.retry_backoff_base ** attempt)

                except Exception as exc:
                    last_error = exc
                    node_log.warning("node_attempt_failed",
                                     error=str(exc), error_type=type(exc).__name__,
                                     attempt=attempt)
                    if attempt <= node_def.max_retries:
                        await asyncio.sleep(node_def.retry_backoff_base ** attempt)

            step.status = StepStatus.FAILED
            step.error_code = type(last_error).__name__
            step.error_message = str(last_error)
            step.completed_at = datetime.now(timezone.utc)
            steps.append(step)

            await self._update_agent_step(step)
            await self._write_audit_log(AuditLogEntry(
                tenant_id=request.tenant_id,
                user_id=request.triggered_by,
                action="NODE_EXECUTION_FAILED",
                resource_type="agent_step",
                agent_run_id=run_id,
                correlation_id=request.correlation_id,
                result=AuditResult.FAILURE,
                error_code=step.error_code,
                metadata={
                    "node_name": node_def.name,
                    "step_index": step_index,
                    "total_attempts": node_def.max_retries + 1,
                    "final_error": str(last_error),
                },
            ))

            raise NodeExecutionError(node_name=node_def.name, original_error=last_error)

        return execute_node

    async def _execute_node_by_type(
        self,
        node_def: GraphNodeDefinition,
        state: dict,
        step: StepResult,
        run_id: uuid.UUID,
        request: AgentRunRequest,
    ) -> dict:
        if node_def.node_type == NodeType.LLM:
            return await self._execute_llm_node(node_def, state, step)
        elif node_def.node_type == NodeType.CONNECTOR:
            return await self._execute_connector_node(node_def, state, step, request)
        elif node_def.node_type == NodeType.CONDITION:
            return await self._execute_condition_node(node_def, state)
        else:
            raise ValueError(f"Unsupported node_type '{node_def.node_type}'")

    async def _execute_llm_node(
        self,
        node_def: GraphNodeDefinition,
        state: dict,
        step: StepResult,
    ) -> dict:
        config: LLMNodeConfig = node_def.config  # type: ignore[assignment]
        llm = LLMFactory.create(config)

        messages = [
            SystemMessage(content=config.system_prompt),
            HumanMessage(content=json.dumps(state.get("input", {}), ensure_ascii=False)),
        ]

        t_start = time.monotonic()
        response = await llm.ainvoke(messages)
        latency_ms = int((time.monotonic() - t_start) * 1000)

        step.llm_model = config.model_name
        step.llm_latency_ms = latency_ms
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            step.prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            step.completion_tokens = response.usage_metadata.get("output_tokens", 0)

        return {"output": {"content": response.content, "node": node_def.name}}

    async def _execute_connector_node(
        self,
        node_def: GraphNodeDefinition,
        state: dict,
        step: StepResult,
        request: AgentRunRequest,
    ) -> dict:
        config: ConnectorNodeConfig = node_def.config  # type: ignore[assignment]
        step.connector_id = config.connector_id
        logger.info("connector_node_executing",
                    connector_id=str(config.connector_id),
                    operation=config.operation,
                    tenant_id=str(request.tenant_id))
        return {"output": {"connector_id": str(config.connector_id), "operation": config.operation}}

    async def _execute_condition_node(self, node_def: GraphNodeDefinition, state: dict) -> dict:
        return state

    def _get_node_timeout(self, node_def: GraphNodeDefinition) -> int:
        config = node_def.config
        if hasattr(config, "timeout_seconds"):
            return config.timeout_seconds  # type: ignore[union-attr]
        return 30

    async def _create_agent_run(
        self, request: AgentRunRequest, run_id: uuid.UUID, started_at: datetime,
    ) -> uuid.UUID:
        async with self._session_factory() as session:
            run = AgentRunORM(
                id=run_id,
                tenant_id=request.tenant_id,
                agent_definition_id=request.agent_definition_id,
                status=AgentRunStatus.RUNNING.value,
                input_payload=request.input_payload,
                trigger_type=request.trigger_type.value,
                triggered_by=request.triggered_by,
                triggered_at=started_at,
                started_at=started_at,
                correlation_id=request.correlation_id,
                parent_run_id=request.parent_run_id,
            )
            session.add(run)
            await session.commit()
        return run_id

    async def _finalize_run(
        self,
        run_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: AgentRunStatus,
        output_payload: dict | None,
        total_tokens: int,
        total_llm_calls: int,
        completed_at: datetime,
        error_code: str | None = None,
        error_message: str | None = None,
        error_node: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            stmt = select(AgentRunORM).where(
                AgentRunORM.id == run_id, AgentRunORM.tenant_id == tenant_id
            )
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if run:
                run.status = status.value
                run.output_payload = output_payload
                run.total_tokens_used = total_tokens
                run.total_llm_calls = total_llm_calls
                run.completed_at = completed_at
                run.error_code = error_code
                run.error_message = error_message
                run.error_node = error_node
                await session.commit()

    async def _write_agent_step(
        self,
        run_id: uuid.UUID,
        tenant_id: uuid.UUID,
        step: StepResult,
        attempt: int,
        max_attempts: int,
    ) -> None:
        async with self._session_factory() as session:
            db_step = AgentStepORM(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                agent_run_id=run_id,
                step_index=step.step_index,
                node_name=step.node_name,
                node_type=step.node_type.value,
                status=step.status.value,
                input_state=step.input_state,
                attempt_number=attempt,
                max_attempts=max_attempts,
                started_at=step.started_at,
            )
            session.add(db_step)
            await session.commit()

    async def _update_agent_step(self, step: StepResult) -> None:
        pass

    async def _write_audit_log(self, entry: AuditLogEntry) -> None:
        try:
            async with self._session_factory() as session:
                log_orm = AuditLogORM(
                    tenant_id=entry.tenant_id,
                    user_id=entry.user_id,
                    session_id=entry.session_id,
                    action=entry.action,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    agent_run_id=entry.agent_run_id,
                    correlation_id=entry.correlation_id,
                    result=entry.result.value,
                    error_code=entry.error_code,
                    ip_address=entry.ip_address,
                    user_agent=entry.user_agent,
                    request_id=entry.request_id,
                    metadata=entry.metadata,
                    data_classification=entry.data_classification.value,
                    created_at=entry.created_at,
                )
                session.add(log_orm)
                await session.commit()
        except Exception as exc:
            logger.critical("audit_log_write_failed",
                            action=entry.action,
                            tenant_id=str(entry.tenant_id),
                            error=str(exc))

    async def _handle_run_failure(
        self,
        run_id: uuid.UUID,
        request: AgentRunRequest,
        error_code: str,
        error_message: str,
        error_node: str | None,
        status: AgentRunStatus,
        started_at: datetime,
        steps: list[StepResult],
    ) -> AgentRunResult:
        completed_at = datetime.now(timezone.utc)

        await self._write_audit_log(AuditLogEntry(
            tenant_id=request.tenant_id,
            user_id=request.triggered_by,
            action="AGENT_RUN_FAILED",
            resource_type="agent_run",
            resource_id=run_id,
            agent_run_id=run_id,
            correlation_id=request.correlation_id,
            result=AuditResult.FAILURE,
            error_code=error_code,
            metadata={
                "error_message": error_message,
                "error_node": error_node,
                "status": status.value,
                "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
            },
        ))

        try:
            await self._finalize_run(
                run_id=run_id,
                tenant_id=request.tenant_id,
                status=status,
                output_payload=None,
                total_tokens=sum(
                    (s.prompt_tokens or 0) + (s.completion_tokens or 0) for s in steps
                ),
                total_llm_calls=sum(1 for s in steps if s.node_type == NodeType.LLM),
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                error_node=error_node,
            )
        except Exception as db_exc:
            logger.error("failed_to_finalize_run_record",
                         run_id=str(run_id), error=str(db_exc))

        return AgentRunResult(
            run_id=run_id,
            tenant_id=request.tenant_id,
            agent_definition_id=request.agent_definition_id,
            status=status,
            steps=steps,
            error_code=error_code,
            error_message=error_message,
            error_node=error_node,
            started_at=started_at,
            completed_at=completed_at,
            correlation_id=request.correlation_id,
        )
