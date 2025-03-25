from pydantic import BaseModel, Field
from typing import Callable, get_type_hints, Dict, Any, Type, Optional
import random
import inspect
import warnings


class Tool:
    """工具描述类别，所有的工具都属于这个类
    """

    def __init__(self, func: Callable, tool_schema: Dict[str, Any]):
        self.func = func
        self.tool_schema = tool_schema


def get_tool_schema(func: Callable) -> dict:
    """tool都是以代码函数的形式存在，但大模型并不能直接认识“代码”，得把代码转成大模型能认识的格式（通常都是json格式字符串），也即tool（function） schema。
    
    Args:
        func (Callable): 工具函数

    Returns:
        dict: 工具schema

    openai接口的工具schema样式：
    {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves current weather for the given location.",
        "strict": true,
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
    """
    # 构建一个基本的工具schema模板，后面缺啥补啥
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description":
            func.__doc__.split('\n')[0] if func.__doc__ else "",  # 只取第一行工具描述
            "parameters": {
                "type": "object",
                "properties": {},  # 后面获取工具入参
                "required": [],  # 后面获取工具入参
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 获取函数签名，可以提取入参名称和类型信息和出参的类型信息
    # 例如：(location: str, units: Optional[str] = 'celsius') -> str
    sig = inspect.signature(func)

    # 获取函数参数类型提示，一个字典，分别描述函数的每个入参的类型和返回值的类型
    # 例如：{'location': <class 'str'>, 'return': <class 'str'>}
    type_hints = get_type_hints(func)

    # 获取函数参数，一个一个参数来
    for param_name, param in sig.parameters.items():
        # param_name: 参数名称
        # param：参数名称：类型

        # 根据参数名称获取参数类型
        param_type = type_hints.get(param_name)
        print(param.default)

        # 初始的，并没有考虑可选类型
        type_ori = python_type_to_schema_type(param_type)

        # 检查是否有默认值，有默认值就是可选的
        if param.default != inspect.Parameter.empty:
            type_final = [type_ori, "null"]
        else:
            type_final = type_ori

        param_schema = {
            "type": type_final,
            "description": get_param_description(func, param_name),
        }

        # required在strict为true时，必须传入所有入参，申明一个数组
        schema['function']['parameters']['required'].append(param_name)

        print(param_schema)

    print(schema)


def python_type_to_schema_type(py_type: Type) -> str:
    """将python类型转换为openai工具schema类型，主要用于函数入参的类型描述

    Args:
        py_type (Type): python类型
        

    Returns:
        str: openai工具schema类型
    """
    # 如果是Optional类型，
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    return type_mapping.get(py_type, "string")


def get_param_description(func: Callable, param_name: str) -> str:
    """从函数文档中提取函数的描述

    Args:
        func (Callable): 工具函数
        param_name (str): 参数名称

    Returns:
        str: 参数描述
    """
    if not func.__doc__:
        return ""

    # 从函数文档中提取参数描述
    doc = func.__doc__
    doc_lines = doc.split('\n')

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


# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
async def get_current_weather(location: str,
                              units: Optional[str] = "celsius") -> str:
    """获取当前天气

    Args:
        location (str): 城市名称
        units (Optional[str]): 温度单位，可选值为"celsius"或"fahrenheit"，默认值为"celsius"

    Returns:
        str: 天气信息
    """
    # 定义备选的天气条件列表
    weather_conditions = ["晴天", "多云", "雨天"]
    # 随机选择一个天气条件
    random_weather = random.choice(weather_conditions)
    # 返回格式化的天气信息
    return f"{location}今天是{random_weather}。"


if __name__ == "__main__":

    func = get_current_weather
    get_tool_schema(func)
