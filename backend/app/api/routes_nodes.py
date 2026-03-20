from fastapi import APIRouter

from ..core.node_base import BaseNode
from ..core.node_registry import registry
from ..schemas import NodeDefinition, ParamDefinitionSchema, PortDefinitionSchema

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


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
                default=p.default,
                description=p.description,
                options=p.options,
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
