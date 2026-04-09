from fastapi import APIRouter

from ..core.device_utils import get_available_devices
from ..core.node_base import BaseNode
from ..core.node_registry import registry
from ..schemas import NodeDefinition, ParamDefinitionSchema, PortDefinitionSchema

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


def _filter_device_options(param_name: str, options: list[str]) -> list[str]:
    """For params named 'device', remove backends that aren't available in this environment."""
    if param_name != "device" or not options:
        return options
    available = set(get_available_devices())
    filtered = [o for o in options if o in available]
    return filtered if filtered else ["cpu"]


def _node_to_definition(cls: type[BaseNode]) -> NodeDefinition:
    return NodeDefinition(
        node_name=cls.NODE_NAME,
        category=cls.CATEGORY,
        description=cls.DESCRIPTION,
        inputs=[
            PortDefinitionSchema(
                name=p.name,
                data_type=p.data_type.value,
                description=p.description,
                optional=p.optional,
            )
            for p in cls.define_inputs()
        ],
        outputs=[
            PortDefinitionSchema(
                name=p.name,
                data_type=p.data_type.value,
                description=p.description,
                optional=p.optional,
            )
            for p in cls.define_outputs()
        ],
        params=[
            ParamDefinitionSchema(
                name=p.name,
                param_type=p.param_type.value,
                default=p.default if p.name != "device" or p.default in get_available_devices() else "cpu",
                description=p.description,
                options=_filter_device_options(p.name, p.options),
                min_value=p.min_value,
                max_value=p.max_value,
            )
            for p in cls.define_params()
        ],
    )


@router.get("", response_model=list[NodeDefinition])
async def list_nodes():
    return [_node_to_definition(cls) for cls in registry.nodes.values()]


@router.get("/{node_name}", response_model=NodeDefinition)
async def get_node(node_name: str):
    cls = registry.get(node_name)
    if not cls:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Node '{node_name}' not found")
    return _node_to_definition(cls)
