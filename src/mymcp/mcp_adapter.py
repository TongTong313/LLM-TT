from typing import Any, Optional, List
import copy
import mcp
from abc import ABC, abstractmethod


class BaseMCPAdapter(ABC):
    """MCP适配器基类，把MCP的工具schema转换为openai的工具schema"""

    def __init__(self):
        self.template_schema = {}

    @abstractmethod
    def convert_to_tool_schema(
            self, tools: List[mcp.server.fastmcp.tools.base.Tool]) -> dict:
        pass


class MCPOpenAIAdapter(BaseMCPAdapter):

    def __init__(self):
        self.template_schema = {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def _get_param_description(self, func_des: str, param_name: str) -> str:
        """从MCP工具description中提取参数的描述

        Args:
            func_des (str): MCP工具的description
            param_name (str): 参数名称
        Returns:
            str: 参数描述
        """
        if not func_des:
            return ""

        # 从函数文档中提取参数描述
        doc_lines = func_des.split('\n')

        # 先找到Args出现的地方，然后下面每行冒号后面都是参数的描述，冒号有可能是中文的冒号也可能是英文的冒号
        arg_start_line_index = -1
        for i, line in enumerate(doc_lines):
            if 'Args' in line:
                arg_start_line_index = i
                break

        if arg_start_line_index == -1:
            return ""

        for i in range(arg_start_line_index + 1, len(doc_lines)):
            line = doc_lines[i].strip()
            # 空行跳过
            if line == '':
                continue

            # 如果遇到下一个主要部分（如Returns:），则停止循环，因为参数信息都有了
            # 同时考虑中文和英文冒号
            if line and not line.startswith(' ') and (line.endswith(':')
                                                      or line.endswith('：')):
                break

            # 如果遇到参数名称，则提取参数名称后面的内容，同时考虑中文和英文冒号
            if line.startswith(param_name):
                # 如果存在中文冒号，则提取中文冒号后面的内容
                if '：' in line:
                    description = line.split('：')[-1].strip()
                else:
                    description = line.split(':')[-1].strip()
                return description

        return ""

    def _recursive_find_field(self, value: Any,
                              field_name: str) -> Optional[Any]:
        """递归查找字典指定字段
        
        Args:
            value (Any): 要查找的值，可能是字典、列表或其他类型
            field_name (str): 要查找的字段名
            
        Returns:
            Optional[Any]: 如果找到字段则返回其值，否则返回 None
        """
        # 如果是字典
        if isinstance(value, dict):
            # 如果当前层就有目标字段，直接返回
            if field_name in value:
                return value[field_name]

            # 递归查找所有嵌套的字典
            for v in value.values():
                field_value = self._recursive_find_field(v, field_name)
                if field_value is not None:
                    return field_value

        # 如果是列表
        elif isinstance(value, list):
            # 遍历列表中的每个元素
            for item in value:
                field_value = self._recursive_find_field(item, field_name)
                if field_value is not None:
                    return field_value

        return None

    def convert_to_tool_schema(
            self, tools: List[mcp.server.fastmcp.tools.base.Tool]) -> dict:
        """将MCP tool中的字段转换为openai接口需要的tool_schema
        
        举例：
        - 对于MCP它的工具形式
        [Tool(name='get_current_time', description='查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“\n\n    Returns:\n        str: 当前时间\n    ', inputSchema={'properties': {}, 'title': 'get_current_timeArguments', 'type': 'object'}), Tool(name='add', description='加法工具\n    \n    Args:\n        a (float): 加数\n        b (float): 加数\n\n    Returns:\n        float: 和\n    ', inputSchema={'properties': {'a': {'title': 'A', 'type': 'number'}, 'b': {'title': 'B', 'type': 'number'}}, 'required': ['a', 'b'], 'title': 'addArguments', 'type': 'object'}), Tool(name='baidu_search', description='百度搜索工具\n\n    Args:\n        query (str): 搜索关键词\n        num_results (int, optional): 搜索结果数量，默认10条.\n\n    Returns:\n        str: 格式化的搜索结果\n    ', inputSchema={'properties': {'query': {'title': 'Query', 'type': 'string'}, 'num_results': {'default': 10, 'title': 'Num Results', 'type': 'integer'}}, 'required': ['query'], 'title': 'baidu_searchArguments', 'type': 'object'}), Tool(name='get_current_weather', description='查询当前天气的工具。返回结果示例：“北京今天是雨天。”\n\n    Args:\n        location (str): 城市名称\n        units (Optional[Literal["celsius", "fahrenheit"]], optional): 温度单位. \n        默认是摄氏度，可选摄氏度或华氏度\n\n    Returns:\n        str: 当前天气\n    ', inputSchema={'properties': {'location': {'title': 'Location', 'type': 'string'}, 'units': {'anyOf': [{'enum': ['celsius', 'fahrenheit'], 'type': 'string'}, {'type': 'null'}], 'default': 'celsius', 'title': 'Units'}}, 'required': ['location'], 'title': 'get_current_weatherArguments', 'type': 'object'})]
        
        - 对于openai它的工具形式
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieves current weather for the given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        },
                        "units": {
                            "type": ["string", "null"],
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Units the temperature will be returned in."
                        }
                        },
                        "required": ["location", "units"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        
        目标就是把MCP tool的工具转换成openai接受的tool_schema
        
        Args:
            tools (List[mcp.server.fastmcp.tools.base.Tool]): MCP tool列表

        Returns:
            List[dict]: openai接受的tool_schema
        """
        # 汇总所有工具的schema
        tool_schemas = []

        for tool in tools:
            # 使用 deepcopy 进行深度拷贝
            tool_schema = copy.deepcopy(self.template_schema)
            # name和description就直接带入
            tool_schema["function"]["name"] = tool.name
            # description需要注意，如果函数注释是标准注释格式，则MCP都会全部写到description中，可以从这个里面提取各入参的描述
            tool_schema["function"]["description"] = tool.description.split(
                "\n")[0]

            # 一个参数一个参数来
            for key, value in tool.inputSchema["properties"].items():
                # 这里还是需要做一些定制化修改
                para_dict = {}  # 至少包含type和description，可选包含enum

                # 递归查找 type 字段
                type_value = self._recursive_find_field(value, "type")
                if type_value is not None:
                    if "default" in value:
                        para_dict["type"] = [type_value, "null"]
                    else:
                        para_dict["type"] = type_value

                # 递归查找 enum 字段
                enum_values = self._recursive_find_field(value, "enum")
                if enum_values is not None:
                    para_dict["enum"] = enum_values

                # 但是参数的description无法从MCP工具中获取，只能从函数的注释中提取，如果是按照标准注释格式，可以从描述里面提取信息
                para_dict["description"] = self._get_param_description(
                    tool.description, key)

                # 把每个变量的para_dict放进去
                tool_schema["function"]["parameters"]["properties"][
                    key] = para_dict
            # required在strict=True时，所有变量都是required的
            tool_schema["function"]["parameters"]["required"] = list(
                tool.inputSchema["properties"].keys())

            tool_schemas.append(tool_schema)

        return tool_schemas
