from app.schemas.models import NodeData, EdgeData


def test_node_data_default_data_has_no_is_entry_point():
    """isEntryPoint lives inside `data` dict and defaults to absent (falsy)."""
    n = NodeData(id="n1", type="Dataset")
    assert n.data.get("isEntryPoint") is None
    assert not n.data.get("isEntryPoint", False)


def test_node_data_can_set_is_entry_point_in_data():
    """isEntryPoint can be set inside the data dict and round-trips."""
    n = NodeData(id="n1", type="Dataset", data={"isEntryPoint": True})
    assert n.data.get("isEntryPoint") is True
    # Round-trip through model_dump
    dumped = n.model_dump()
    assert dumped["data"]["isEntryPoint"] is True


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
