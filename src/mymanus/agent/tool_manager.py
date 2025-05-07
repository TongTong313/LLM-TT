from pydantic import BaseModel, Field
from typing import Callable, get_type_hints, Dict, Any, Type, Optional, List, Literal, get_args, get_origin
import random
import inspect
import warnings
from mymanus.tool.math import add


class Tool:
    """工具描述类别，所有的工具都属于这个类
    
    Args:
        func (Callable): 工具函数
        tool_name (str, optional): 工具名称，默认是函数名
    """

    def __init__(self, func: Callable, tool_name: Optional[str] = None):
        self.func = func
        if tool_name is None:
            self.tool_name = func.__name__
        else:
            self.tool_name = tool_name
        self.tool_schema = self._get_tool_schema(func)

    def execute(self, **kwargs):
        """执行工具"""
        return self.func(**kwargs)

    def _get_tool_schema(self, func: Callable) -> Dict:
        """tool都是以函数代码的形式存在，但大模型并不能直接认识"代码"，得把代码转成大模型能认识的格式（通常都是json格式字符串），也即tool（function） schema。
        
        Args:
            func (Callable): 工具函数

        Returns:
            Dict: 工具schema

        openai接口的工具schema样式：
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
        """
        # 构建一个基本的工具schema模板，后面缺啥补啥
        schema = {
            "type": "function",
            "function": {
                "name": self.tool_name if self.tool_name else func.__name__,
                "description": func.__doc__.split('\n')[0]
                if func.__doc__ else "",  # 只取第一行工具描述
                "parameters": {
                    "type": "object",
                    "properties": {},  # 后面获取工具入参
                    "required": [],  # 一旦strict=True，所有变量都是required的
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        # 获取函数签名：可以提取入参名称和类型信息和出参的类型信息
        # 例如：(location: str, units: Optional[str] = 'celsius') -> str
        sig = inspect.signature(func)

        # 获取函数参数类型提示：一个字典，分别描述函数的每个入参的类型和返回值的类型
        # 例如：{'location': <class 'str'>, 'return': <class 'str'>}
        type_hints = get_type_hints(func)

        # 获取函数参数，一个一个【入参】来
        for param_name, param in sig.parameters.items():
            # param_name: 参数名称
            # param：参数名称：类型 = 默认值（如果有）  <class 'inspect.Parameter'>
            # 根据参数名称获取参数类型
            param_type = type_hints.get(param_name)

            # 检查是否是 Literal 类型
            if get_origin(param_type) is Literal:
                # 获取 Literal 的所有可能值
                enum_values = get_args(param_type)
                # 初始的类型，并没有考虑可选情况
                type_ori = self._python_type_to_schema_type(
                    type(enum_values[0]))
                # 检查是否有默认值，有默认值就是可选的
                if param.default != inspect.Parameter.empty:
                    type_final = [type_ori, "null"]
                else:
                    type_final = type_ori

                # required在strict为true时，必须传入所有入参，申明一个数组
                schema['function']['parameters']['required'].append(param_name)

                # 将参数信息添加到properties中，包含enum字段
                schema['function']['parameters']['properties'][param_name] = {
                    "type": type_final,
                    "enum": list(enum_values),
                    "description":
                    self._get_param_description(func, param_name),
                }
            else:
                # 其他的类型
                type_ori = self._python_type_to_schema_type(param_type)

                # 考虑List类型
                if get_origin(param_type) is list:
                    type_ori = "array"

                # 检查是否有默认值，有默认值就是可选的
                if param.default != inspect.Parameter.empty:
                    type_final = [type_ori, "null"]
                else:
                    type_final = type_ori

                # required在strict为true时，必须传入所有入参，申明一个数组
                schema['function']['parameters']['required'].append(param_name)

                # 将参数信息添加到properties中
                schema['function']['parameters']['properties'][param_name] = {
                    "type": type_final,
                    "description":
                    self._get_param_description(func, param_name),
                }

        return schema

    def _python_type_to_schema_type(self, py_type: Type) -> str:
        """将python类型转换为openai工具schema类型，主要用于函数入参的类型描述

        Args:
            py_type (Type): python类型
            

        Returns:
            str: openai工具schema类型
        """

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        return type_mapping.get(py_type, "string")

    def _get_param_description(self, func: Callable, param_name: str) -> str:
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


class ToolManager:
    """工具管理类，管理所有的工具，期望具备的功能：
    1. 工具注册：让工具管理器感知到，包括生成对应的schema保存起来
    2. 工具执行：执行工具，并返回结果
    3. 工具删除：删除工具
    4. 工具列表：获取所有工具列表
    """

    # 初始化类
    def __init__(self):
        self.tools: Dict[str, Tool] = {}  # 每一个工具都是Tool实例

    # 工具注册：让工具管理器感知到
    def register_tool(self, func: Callable, tool_name: Optional[str] = None):
        """注册工具

        Args:
            func (Callable): 工具函数
            tool_name (Optional[str]): 工具名称，默认是函数名
        """
        # 后面可能会增加工具是类的可能性，现在默认就是一个函数
        # 生成工具的名称，没有名称给一个默认的名称
        if tool_name is None:
            tool_name = func.__name__
        elif tool_name in self.tools:
            warnings.warn(f"工具名称{tool_name}已存在，将覆盖原有工具")

        # 生成工具的实例
        tool = Tool(func=func, tool_name=tool_name)
        self.tools[tool_name] = tool

    # 工具执行：执行工具，并返回结果
    def execute_tool(self, tool_name: str, **kwargs):
        """执行工具

        Args:
            name (str): 工具名称
            **kwargs: 工具入参

        Returns:
            Any: 工具返回结果
        """
        if tool_name not in self.tools:
            raise ValueError(f"工具名称{tool_name}不存在")

        return self.tools[tool_name].execute(**kwargs)

    # 工具删除：删除工具
    def delete_tool(self, tool_name: str) -> bool:
        """删除工具

        Args:
            tool_name (str): 工具名称
        
        Returns:
            bool: 是否删除成功
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True

        return False

    # 工具列表：获取所有工具列表
    def get_tool_list(self) -> List[Tool]:
        """获取所有工具，并返回列表

        Returns:
            List[Tool]: 工具列表
        """
        return list(self.tools.values())

    # 获取所有的schema
    def get_tool_schema_list(self) -> List[Dict]:
        """获取所有工具的schema

        Returns:
            List[Dict]: 工具schema列表
        """
        return [tool.tool_schema for tool in self.tools.values()]


# 模拟天气查询工具。返回结果示例："北京今天是雨天。"
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
    # 随机选择一个温度
    random_temperature = random.randint(10, 30)
    # 摄氏度转华氏度
    # 根据units返回
    if units == "celsius":
        return f"{location}今天是{random_weather}，温度是{random_temperature}度。"
    else:
        random_temperature = random_temperature * 1.8 + 32
        return f"{location}今天是{random_weather}，温度是{random_temperature}华氏度。"


if __name__ == "__main__":

    # func = get_current_weather
    tool_manager = ToolManager()
    # tool_manager.register_tool(func)
    # print(tool_manager.tools['get_current_weather'].__dict__)

    func = add
    tool_manager.register_tool(func)
    print(tool_manager.tools['add'].tool_schema)
