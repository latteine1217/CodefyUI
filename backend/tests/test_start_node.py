from app.core.node_base import DataType
from app.nodes.control.start_node import StartNode


def test_start_node_metadata():
    assert StartNode.NODE_NAME == "Start"
    assert StartNode.CATEGORY == "Control"
    assert StartNode.DESCRIPTION  # non-empty


def test_start_node_has_no_inputs():
    assert StartNode.define_inputs() == []


def test_start_node_has_one_trigger_output():
    outputs = StartNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "trigger"
    assert outputs[0].data_type == DataType.TRIGGER


def test_start_node_has_no_params():
    assert StartNode.define_params() == []


def test_start_node_execute_is_noop():
    node = StartNode()
    result = node.execute(inputs={}, params={})
    assert result == {}


def test_start_node_is_auto_discovered():
    """The node registry should pick up StartNode after discovery."""
    from app.config import settings
    from app.core.node_registry import NodeRegistry

    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    assert reg.get("Start") is StartNode
