import pytest
from pydantic import ValidationError

from src.agent.state import AgentState, HopRecord
from src.config import AGENT_MAX_HOPS


class TestHopRecord:
    def test_round_trips_through_json(self):
        hop = HopRecord(
            file_path="starlette/authentication.py",
            symbol_name="AuthenticationMiddleware",
            start_line=10,
            end_line=20,
            reason="Followed reference to AuthenticationMiddleware",
        )

        restored = HopRecord.model_validate_json(hop.model_dump_json())

        assert restored == hop

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            HopRecord(file_path="a.py")


class TestAgentState:
    def test_defaults_are_empty_with_full_hop_budget(self):
        state = AgentState(question="how is auth validated?")

        assert state.context == []
        assert state.hop_path == []
        assert state.hops_remaining == AGENT_MAX_HOPS
        assert state.pending_symbol is None
        assert state.answer is None

    def test_requires_a_question(self):
        with pytest.raises(ValidationError):
            AgentState()
