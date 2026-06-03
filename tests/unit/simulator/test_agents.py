import pytest
from simulator.domain.agents import NormalUserAgent, BotAgent
from shared.domain.models import ActionType

def test_normal_user_generates_action_with_human_label():
    agent = NormalUserAgent()
    action = agent.generate_action(content="Hello world")
    
    assert action.user_id is not None
    assert action.true_label == 0
    assert action.action_type in [ActionType.POST, ActionType.REPLY, ActionType.FOLLOW]

def test_bot_generates_action_with_bot_label():
    agent = BotAgent()
    action = agent.generate_action(content="Buy crypto now!", target_id="user_123")
    
    assert action.user_id is not None
    assert action.true_label == 1

def test_pydantic_model_excludes_true_label_on_json_dump():
    agent = BotAgent()
    action = agent.generate_action()
    json_dump = action.model_dump_json()
    
    assert "true_label" not in json_dump
    assert "user_id" in json_dump