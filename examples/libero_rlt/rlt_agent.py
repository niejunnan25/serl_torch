"""Compatibility shim for the shared RLT agent implementation."""

from serl_launcher.agents.rlt.agent import RLTAgent, RLTTrainState, create_rlt_agent_from_cfg

__all__ = ["RLTAgent", "RLTTrainState", "create_rlt_agent_from_cfg"]
