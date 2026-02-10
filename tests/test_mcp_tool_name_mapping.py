from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.agents.spoon_react_skill import SpoonReactSkill


class _ToolManagerStub:
    def __init__(self, names):
        self.tool_map = {name: object() for name in names}


def _mcp_agent_with_tools(tool_names):
    return SpoonReactMCP(available_tools=_ToolManagerStub(tool_names))


def _skill_agent_with_tools(tool_names):
    # Avoid full agent initialization in pure mapping unit tests
    agent = SpoonReactSkill.__new__(SpoonReactSkill)
    agent.available_tools = _ToolManagerStub(tool_names)
    return agent


def test_mcp_map_tool_name_prefers_exact_match():
    agent = _mcp_agent_with_tools(["proxy_search", "search"])
    assert agent._map_mcp_tool_name("proxy_search") == "proxy_search"


def test_mcp_map_tool_name_generic_prefix_fallback_single_segment():
    agent = _mcp_agent_with_tools(["search"])
    assert agent._map_mcp_tool_name("proxy_search") == "search"


def test_mcp_map_tool_name_generic_prefix_fallback_multi_segment():
    agent = _mcp_agent_with_tools(["search"])
    assert agent._map_mcp_tool_name("vendor_proxy_search") == "search"


def test_mcp_map_tool_name_unmatched_keeps_original_for_remote_routing():
    agent = _mcp_agent_with_tools(["other_tool"])
    assert agent._map_mcp_tool_name("gateway_unknown_tool") == "gateway_unknown_tool"


def test_skill_map_tool_name_prefers_exact_match():
    agent = _skill_agent_with_tools(["proxy_search", "search"])
    assert agent._map_mcp_tool_name("proxy_search") == "proxy_search"


def test_skill_map_tool_name_generic_prefix_fallback_single_segment():
    agent = _skill_agent_with_tools(["search"])
    assert agent._map_mcp_tool_name("proxy_search") == "search"


def test_skill_map_tool_name_generic_prefix_fallback_multi_segment():
    agent = _skill_agent_with_tools(["search"])
    assert agent._map_mcp_tool_name("vendor_proxy_search") == "search"


def test_skill_map_tool_name_unmatched_keeps_original_for_remote_routing():
    agent = _skill_agent_with_tools(["other_tool"])
    assert agent._map_mcp_tool_name("gateway_unknown_tool") == "gateway_unknown_tool"

