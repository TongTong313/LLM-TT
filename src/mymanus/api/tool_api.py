from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Callable
import importlib
import sys
from ..agent.llm import LLM

router = APIRouter()

# 全局 LLM 实例
llm_instance: Optional[LLM] = None


class ToolDefinition(BaseModel):
    """工具定义模型"""
    name: str
    description: str
    parameters: Dict
    module_path: str  # 工具函数所在的模块路径
    function_name: str  # 工具函数名称


@router.post("/register_tool")
async def register_tool(tool_def: ToolDefinition):
    """注册新工具
    
    Args:
        tool_def: 工具定义
        
    Returns:
        注册结果
    """
    try:
        # 动态导入模块
        module = importlib.import_module(tool_def.module_path)
        # 获取函数
        function = getattr(module, tool_def.function_name)

        # 注册工具
        llm_instance.register_tool(name=tool_def.name,
                                   description=tool_def.description,
                                   parameters=tool_def.parameters,
                                   function=function)

        return {"status": "success", "message": f"工具 {tool_def.name} 注册成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register_tools")
async def register_tools(tool_defs: List[ToolDefinition]):
    """批量注册工具
    
    Args:
        tool_defs: 工具定义列表
        
    Returns:
        注册结果
    """
    results = []
    for tool_def in tool_defs:
        try:
            # 动态导入模块
            module = importlib.import_module(tool_def.module_path)
            # 获取函数
            function = getattr(module, tool_def.function_name)

            # 注册工具
            llm_instance.register_tool(name=tool_def.name,
                                       description=tool_def.description,
                                       parameters=tool_def.parameters,
                                       function=function)

            results.append({
                "name": tool_def.name,
                "status": "success",
                "message": f"工具 {tool_def.name} 注册成功"
            })
        except Exception as e:
            results.append({
                "name": tool_def.name,
                "status": "error",
                "message": str(e)
            })

    return {"results": results}


@router.get("/list_tools")
async def list_tools():
    """获取所有已注册的工具列表"""
    if not llm_instance:
        raise HTTPException(status_code=500, detail="LLM 实例未初始化")
    return {"tools": llm_instance.tool_manager.get_all_tools()}


@router.delete("/remove_tool/{tool_name}")
async def remove_tool(tool_name: str):
    """移除指定工具
    
    Args:
        tool_name: 工具名称
    """
    if not llm_instance:
        raise HTTPException(status_code=500, detail="LLM 实例未初始化")
    try:
        llm_instance.tool_manager.remove_tool(tool_name)
        return {"status": "success", "message": f"工具 {tool_name} 已移除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def init_llm(llm: LLM):
    """初始化 LLM 实例
    
    Args:
        llm: LLM 实例
    """
    global llm_instance
    llm_instance = llm
