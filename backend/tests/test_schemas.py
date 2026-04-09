from app.schemas.models import NodeData, EdgeData


def test_node_data_default_is_entry_point_false():
    n = NodeData(id="n1", type="Dataset")
    assert n.isEntryPoint is False


def test_node_data_can_set_is_entry_point():
    n = NodeData(id="n1", type="Dataset", isEntryPoint=True)
    assert n.isEntryPoint is True


def test_edge_data_default_type_is_data():
    e = EdgeData(id="e1", source="a", target="b")
    assert e.type == "data"


def test_edge_data_can_set_type_trigger():
    e = EdgeData(id="e1", source="a", target="b", type="trigger")
    assert e.type == "trigger"


def test_edge_data_rejects_unknown_type():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EdgeData(id="e1", source="a", target="b", type="bogus")
