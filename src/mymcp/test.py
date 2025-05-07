# from mcp.server.fastmcp.utilities.func_metadata import func_metadata


def add(number1: float, number2: float) -> float:
    """Add two numbers together
    """
    return number1 + number2


# meta = func_metadata(add)

# parameters = meta.arg_model.model_json_schema()

# print(parameters)

# 可以把参数属性带出来，如果增加注释的解析，效果会更好
add.__signature__ = 'hahaha'
add._parameter_schema = {'a': {'description': 'a的描述', 'type': 'number'}}
print(add.__signature__)
print(add.__doc__)
print(add.__name__)
print(add._parameter_schema)

globalns = getattr(add, "__globals__", {})
print(globalns)
# print(globalns['parameters'])

# a = {
#     'properties': {
#         'a': {
#             'title': 'A',
#             'type': 'number'
#         },
#         'b': {
#             'title': 'B',
#             'type': 'number'
#         }
#     },
#     'required': ['a', 'b'],
#     'title': 'addArguments',
#     'type': 'object'
# }

# b = {
#     "properties": {
#         "query": {
#             "type": "string",
#             "description": "搜索关键词"
#         },
#         "num_results": {
#             "type": "integer",
#             "description": "搜索结果数量",
#             "default": 3
#         }
#     },
#     "required": ["query"],
#     "title": "baidu_searchArguments",
#     "type": "object"
# }
