"""
Agent module -- Stage 1 Coordinator layer.

Public API:
  Coordinator       -- event-driven pipeline orchestrator
  LLMClient         -- Anthropic / OpenAI wrapper with 30s timeout
  MemoryStore       -- Redis HASH-backed memory
  StateMachine      -- pipeline state transitions
  AgentLoop         -- Redis Streams event loop
  publish_event     -- publish event to pipeline stream
  contracts.*       -- PipelineContract, StageSummary, GateDecision, ...
"""

from ml_engine.agent.coordinator import Coordinator
from ml_engine.agent.llm_client import LLMClient
from ml_engine.agent.loop import AgentLoop, publish_event
from ml_engine.agent.memory import MemoryStore
from ml_engine.agent.skills import Skill, SkillLoader
from ml_engine.agent.state_machine import StateMachine
from ml_engine.agent.workers import ExecutorWorker, EvaluatorWorker
from ml_engine.agent.contracts import (
    PipelineContract,
    TargetSpec,
    DataSpec,
    BudgetSpec,
    AcceptanceCriteria,
    LineageSpec,
    GateDecision,
    StageSummary,
)

__all__ = [
    "Coordinator",
    "ExecutorWorker",
    "EvaluatorWorker",
    "LLMClient",
    "AgentLoop",
    "publish_event",
    "MemoryStore",
    "Skill",
    "SkillLoader",
    "StateMachine",
    "PipelineContract",
    "TargetSpec",
    "DataSpec",
    "BudgetSpec",
    "AcceptanceCriteria",
    "LineageSpec",
    "GateDecision",
    "StageSummary",
]
