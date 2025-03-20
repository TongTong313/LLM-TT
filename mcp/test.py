from mcp.server.fastmcp.utilities.func_metadata import func_metadata


def add(number1: float, number2: float) -> float:
    """Add two numbers together
    """
    return number1 + number2


meta = func_metadata(add)

parameters = meta.arg_model.model_json_schema()

# print(parameters)

globalns = getattr(add, "__globals__", {})
print(globalns['parameters'])

a = {
    'properties': {
        'a': {
            'title': 'A',
            'type': 'number'
        },
        'b': {
            'title': 'B',
            'type': 'number'
        }
    },
    'required': ['a', 'b'],
    'title': 'addArguments',
    'type': 'object'
}

b = {
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词"
        },
        "num_results": {
            "type": "integer",
            "description": "搜索结果数量",
            "default": 3
        }
    },
    "required": ["query"],
    "title": "baidu_searchArguments",
    "type": "object"
}
