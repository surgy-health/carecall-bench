import json

import pytest
from tau2.data_model.message import ToolCall

from domains.hospital_in.environment import get_environment


@pytest.fixture
def env():
    """A fresh hospital environment on the seed DB."""
    return get_environment()


@pytest.fixture
def db(env):
    return env.tools.db


@pytest.fixture
def call(env):
    """Call a tool the way tau2 does. Returns (error, parsed_content)."""

    def _call(tool_name: str, /, **arguments):
        msg = env.get_response(ToolCall(id="t", name=tool_name, arguments=arguments))
        try:
            content = json.loads(msg.content)
        except (TypeError, json.JSONDecodeError):
            content = msg.content
        return msg.error, content

    return _call
