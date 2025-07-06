from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import json
import asyncio
from typing import Dict, Any
import os
from pathlib import Path

# 添加项目根目录到Python路径
import sys
import io
import contextlib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入现有的智能体代码
try:
    from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
    from typing import TypedDict, Literal, List, Callable
    from graphagent.node.planning import PlanningNode, PlanningNodeState, PlanningNodeConfig, PlanningNodeRunnableConfig
    from graphagent.node.tool import ToolNode, ToolNodeState, ToolNodeConfig, FunctionTool
    from graphagent.prompt.system_prompt import DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE, DEFAULT_SYSTEM_PROMPT_FOR_ROUTER_NODE, DEFAULT_SYSTEM_PROMPT_FOR_SUMMARY_NODE
    from graphagent.message.openai import OpenAIMessage
    from langchain_core.runnables.config import RunnableConfig
    from graphagent.tool import add, baidu_search, get_current_time
    from graphagent.node.tool import BaseTool
    from graphagent.node.router import RouterNodeState, RouterNodeConfig, RouterNodeRunnableConfig
    from graphagent.node.router import RouterNode, router_function
    from graphagent.node.summary import SummaryNodeState, SummaryNodeConfig, SummaryNodeRunnableConfig
    from graphagent.node.summary import SummaryNode
    from graphagent.stream_handler import create_stream_handler
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入智能体模块: {e}")
    print("将使用模拟模式运行")
    AGENT_AVAILABLE = False

app = FastAPI(title="LLM-TT Chat Bot", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1. 预制节点配套预制状态(TypedDict)，用户可以直接使用，也支持自定义
class State(PlanningNodeState, ToolNodeState, RouterNodeState,
            SummaryNodeState):
    pass


# 2. 用户可以拿到静态预制配置(Pydantic模型)，也可以自己设定，甚至修改参数
# TypedDict不支持默认值语法
class ConfigSchema(PlanningNodeConfig, ToolNodeConfig, RouterNodeConfig,
                   SummaryNodeConfig):
    stream: bool = True
    tools: List[BaseTool | Callable] = [
        FunctionTool(tool=add),
        FunctionTool(tool=baidu_search),
        FunctionTool(tool=get_current_time)
    ]


# 3. 用户可以拿到预制运行时配置(TypedDict模型)，也可以自己设定，甚至修改参数
class RunnableConfigSchema(PlanningNodeRunnableConfig,
                           RouterNodeRunnableConfig,
                           SummaryNodeRunnableConfig):
    pass


# 4. 用户拿到预制节点形成智能体，或直接使用智能体模板
class MyAgent:

    def __init__(self, config: ConfigSchema, runnable_config: RunnableConfig):
        self.config = config
        self.runnable_config = runnable_config
        # 配置静态参数
        self.planning_node = PlanningNode(
            api_key=config.api_key,
            base_url=config.base_url,
            tools=config.tools,
            stream=config.stream,
            enable_thinking=config.enable_thinking,
            step_start_token=config.step_start_token,
            step_end_token=config.step_end_token)
        self.tool_node = ToolNode(tools=config.tools)
        self.router_node = RouterNode(api_key=config.api_key,
                                      base_url=config.base_url,
                                      tools=config.tools,
                                      stream=config.stream,
                                      enable_thinking=config.enable_thinking)
        self.summary_node = SummaryNode(api_key=config.api_key,
                                        base_url=config.base_url,
                                        stream=config.stream,
                                        enable_thinking=config.enable_thinking)

    def create_graph(self) -> CompiledStateGraph:
        self.graph = StateGraph(State, config_schema=self.runnable_config)
        # 添加节点
        self.graph.add_node("planning", self.planning_node)
        self.graph.add_node("tool", self.tool_node)
        self.graph.add_node("router", self.router_node)
        self.graph.add_node("summary", self.summary_node)
        # 添加边
        self.graph.add_edge(START, "planning")
        self.graph.add_edge("planning", "router")
        # 使用条件边，让router根据状态决定下一步
        self.graph.add_conditional_edges("router", router_function, {
            "tool": "tool",
            "router": "router",
            "summary": "summary"
        })
        # tool节点执行完后回到router节点
        self.graph.add_edge("tool", "router")
        self.graph.add_edge("summary", END)

        return self.graph.compile()


# 全局智能体实例
agent = None
graph = None
# 全局对话历史字典，以WebSocket ID为键
conversation_histories = {}


def initialize_agent():
    """初始化智能体"""
    global agent, graph
    if not AGENT_AVAILABLE:
        return None, None

    if agent is None:
        try:
            config = ConfigSchema()
            runnable_config = RunnableConfigSchema()
            agent = MyAgent(config, runnable_config)
            graph = agent.create_graph()
        except Exception as e:
            print(f"智能体初始化失败: {e}")
            return None, None
    return agent, graph


async def handle_simulation_mode(websocket: WebSocket):
    """处理模拟模式"""
    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            if not user_message.strip():
                continue

            # 发送开始响应信号
            await websocket.send_text(
                json.dumps({
                    "type": "start",
                    "message": "开始生成回复..."
                }))

            # 模拟不同类型的响应
            responses = [
                ("planning", "🤔 正在分析你的需求..."), ("tool", "🔧 正在执行相关工具..."),
                ("router", "🔄 正在路由到下一个步骤..."),
                ("log", "🔧 调用工具: baidu_search | 参数: 搜索关键词"),
                ("plan_step", "📝 模拟计划步骤：分析用户需求"),
                ("tool_result",
                 '{"search_results": ["结果1", "结果2", "结果3"], "status": "success"}'
                 ), ("chunk", f"收到你的消息：{user_message}\n\n这是一个模拟回复，用于测试前端功能。")
            ]

            for response_type, content in responses:
                if response_type == "tool_result":
                    await websocket.send_text(
                        json.dumps({
                            "type": response_type,
                            "content": content,
                            "tool_name": "模拟工具"
                        }))
                else:
                    await websocket.send_text(
                        json.dumps({
                            "type": response_type,
                            "content": content,
                            "full_content": content
                        }))
                await asyncio.sleep(0.5)  # 500ms延迟

            # 发送完成信号
            await websocket.send_text(
                json.dumps({
                    "type": "end",
                    "message": "回复完成"
                }))

    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        print(f"模拟模式错误: {e}")
        try:
            await websocket.send_text(
                json.dumps({
                    "type": "error",
                    "message": f"模拟模式错误: {str(e)}"
                }))
        except:
            pass


# 静态文件服务
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def read_root():
    """返回主页"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>聊天机器人</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            .success { color: green; }
            .info { color: blue; }
        </style>
    </head>
    <body>
        <h1>聊天机器人</h1>
        <div class="success">✓ 服务器运行正常</div>
        <div class="info">这是一个简单的测试页面</div>
        <div class="info">
            <a href="/static/index.html">访问完整聊天界面</a>
        </div>
        <div class="info">
            <a href="/static/test.html">访问测试页面</a>
        </div>
    </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "服务器运行正常"}


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，处理流式聊天"""
    await websocket.accept()

    try:
        # 初始化智能体
        agent, graph = initialize_agent()
        if agent is None or graph is None:
            print("智能体初始化失败，使用模拟模式")
            await handle_simulation_mode(websocket)
            return

        runnable_config = {
            'configurable': {
                'planning_max_tokens': 8000,
                'planning_model': 'qwen-plus',
                'planning_system_prompt':
                DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE,
                'planning_temperature': 0.7,
                'router_max_tokens': 8000,
                'router_model': 'qwen-plus',
                'router_system_prompt': DEFAULT_SYSTEM_PROMPT_FOR_ROUTER_NODE,
                'router_temperature': 0.7,
                'summary_max_tokens': 8000,
                'summary_model': 'qwen-plus',
                'summary_system_prompt':
                DEFAULT_SYSTEM_PROMPT_FOR_SUMMARY_NODE,
                'summary_temperature': 0.7
            }
        }

        # 获取或初始化对话历史
        websocket_id = id(websocket)
        if websocket_id not in conversation_histories:
            conversation_histories[websocket_id] = []
        conversation_history = conversation_histories[websocket_id]

        while True:
            # 接收用户消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message_type = message_data.get("type", "message")

            if message_type == "clear":
                # 清除对话历史
                conversation_histories[websocket_id] = []
                await websocket.send_text(
                    json.dumps({
                        "type": "clear_success",
                        "message": "对话历史已清除"
                    }))
                continue

            user_message = message_data.get("message", "")

            if not user_message.strip():
                continue

            # 发送开始响应信号
            await websocket.send_text(
                json.dumps({
                    "type": "start",
                    "message": "开始生成回复..."
                }))

            # 流式处理响应
            try:
                # 使用改进的流式输出处理器
                async def send_to_websocket(message):
                    await websocket.send_text(message)

                # 创建流式输出处理器
                from graphagent.stream_handler import create_stream_handler
                stream_handler = create_stream_handler(send_to_websocket,
                                                       use_async=True)

                with stream_handler:
                    # 构建包含历史对话的消息列表
                    messages = conversation_history.copy()
                    messages.append(
                        OpenAIMessage.user_message(content=user_message))

                    print(f"发送给智能体的消息数量: {len(messages)}")
                    for i, msg in enumerate(messages):
                        # 安全地获取消息内容
                        msg_content = ""
                        if hasattr(msg, 'content'):
                            msg_content = msg.content
                        elif isinstance(msg, dict) and 'content' in msg:
                            msg_content = msg['content']

                        print(
                            f"消息 {i}: {type(msg).__name__} - {msg_content[:50]}..."
                        )

                    # 在流式处理器上下文中执行智能体
                    events = graph.astream({"messages": messages},
                                           stream_mode="updates",
                                           config=runnable_config)

                    current_response = ""
                    current_log_content = ""
                    current_step = ""
                    final_messages = None

                    async for event in events:
                        # 保存最终的messages状态
                        if "messages" in event:
                            final_messages = event["messages"]

                        # 检查当前节点类型并发送状态信息
                        current_node = None
                        if "planning" in str(event):
                            current_node = "planning"
                            await websocket.send_text(
                                json.dumps({
                                    "type": "status",
                                    "content": "🤔 智能体正在思考...",
                                    "node": "planning"
                                }))
                        elif "tool" in str(event):
                            current_node = "tool"
                            await websocket.send_text(
                                json.dumps({
                                    "type": "status",
                                    "content": "🔧 智能体正在执行...",
                                    "node": "tool"
                                }))
                        elif "router" in str(event):
                            current_node = "router"
                            await websocket.send_text(
                                json.dumps({
                                    "type": "status",
                                    "content": "🔄 智能体正在路由...",
                                    "node": "router"
                                }))
                        elif "summary" in str(event):
                            current_node = "summary"
                            await websocket.send_text(
                                json.dumps({
                                    "type": "status",
                                    "content": "📝 智能体正在总结...",
                                    "node": "summary"
                                }))

                        # 检查messages中的内容
                        if "messages" in event and event["messages"]:
                            latest_message = event["messages"][-1]

                            # 安全地获取消息内容
                            content = None
                            if hasattr(latest_message, 'content'):
                                content = latest_message.content
                            elif isinstance(
                                    latest_message,
                                    dict) and 'content' in latest_message:
                                content = latest_message['content']

                            if content:
                                # 判断消息类型
                                message_type = "chunk"
                                if current_node == "planning":
                                    message_type = "planning"
                                elif current_node == "tool":
                                    message_type = "tool"
                                elif current_node == "router":
                                    message_type = "router"
                                elif current_node == "summary":
                                    message_type = "summary"

                                # 对于summary类型，发送到新的对话框
                                if message_type == "summary":
                                    await websocket.send_text(
                                        json.dumps({
                                            "type": "summary",
                                            "content": content,
                                            "full_content": content
                                        }))
                                else:
                                    # 对于其他类型，更新当前响应
                                    if content != current_response:
                                        await websocket.send_text(
                                            json.dumps({
                                                "type": message_type,
                                                "content": content,
                                                "full_content": content
                                            }))
                                        current_response = content

                        # 检查plan中的内容
                        if "plan" in event and event["plan"]:
                            for step in event["plan"]:
                                # 安全地获取步骤描述
                                step_description = None
                                if hasattr(step, 'description'):
                                    step_description = step.description
                                elif isinstance(
                                        step, dict) and 'description' in step:
                                    step_description = step['description']

                                if step_description:
                                    # 安全地获取步骤状态
                                    step_status = 'pending'
                                    if hasattr(step, 'status'):
                                        step_status = step.status
                                    elif isinstance(step,
                                                    dict) and 'status' in step:
                                        step_status = step['status']

                                    # 只有当步骤状态发生变化时才发送
                                    if step_description != current_step or step_status in [
                                            'running', 'completed'
                                    ]:
                                        await websocket.send_text(
                                            json.dumps({
                                                "type": "plan_step",
                                                "content": step_description,
                                                "status": step_status
                                            }))
                                        current_step = step_description

                        # 检查工具执行结果
                        if "tool_results" in event and event["tool_results"]:
                            for tool_result in event["tool_results"]:
                                # 安全地获取工具结果内容
                                result_content = None
                                if hasattr(tool_result, 'content'):
                                    result_content = tool_result.content
                                elif isinstance(
                                        tool_result,
                                        dict) and 'content' in tool_result:
                                    result_content = tool_result['content']

                                if result_content:
                                    # 安全地获取工具名称
                                    tool_name = 'unknown'
                                    if hasattr(tool_result, 'tool_name'):
                                        tool_name = tool_result.tool_name
                                    elif isinstance(
                                            tool_result, dict
                                    ) and 'tool_name' in tool_result:
                                        tool_name = tool_result['tool_name']

                                    await websocket.send_text(
                                        json.dumps({
                                            "type": "tool_result",
                                            "content": result_content,
                                            "tool_name": tool_name
                                        }))

                # 更新对话历史 - 确保包含用户消息和智能体回复
                if final_messages:
                    # 确保对话历史包含完整的对话
                    conversation_histories[websocket_id] = final_messages
                    print(f"对话历史已更新，当前长度: {len(final_messages)}")
                    for i, msg in enumerate(final_messages):
                        # 安全地获取消息内容
                        msg_content = ""
                        if hasattr(msg, 'content'):
                            msg_content = msg.content
                        elif isinstance(msg, dict) and 'content' in msg:
                            msg_content = msg['content']

                        print(
                            f"消息 {i}: {type(msg).__name__} - {msg_content[:50]}..."
                        )
                else:
                    # 如果没有final_messages，至少添加用户消息和空的智能体回复
                    conversation_histories[websocket_id].append(
                        OpenAIMessage.user_message(content=user_message))
                    # 添加一个空的智能体回复以保持对话结构
                    conversation_histories[websocket_id].append(
                        OpenAIMessage.assistant_message(content=""))

                # 发送完成信号
                await websocket.send_text(
                    json.dumps({
                        "type": "end",
                        "message": "回复完成"
                    }))

            except Exception as e:
                # 发送错误信息
                import traceback
                error_details = traceback.format_exc()
                print(f"智能体执行错误: {e}")
                print(f"错误详情: {error_details}")

                await websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "message": f"处理过程中出现错误: {str(e)}"
                    }))

    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        print(f"WebSocket错误: {e}")
        try:
            await websocket.send_text(
                json.dumps({
                    "type": "error",
                    "message": f"连接错误: {str(e)}"
                }))
        except:
            pass


@app.post("/api/chat")
async def chat_api(request: Dict[str, Any]):
    """HTTP API端点，用于非流式聊天"""
    try:
        user_message = request.get("message", "")
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        if not AGENT_AVAILABLE:
            # 模拟模式
            response = f"收到你的消息：{user_message}\n\n这是一个模拟回复，智能体模块不可用。"
            return {"response": response}

        # 初始化智能体
        agent, graph = initialize_agent()

        runnable_config = {
            'configurable': {
                'planning_max_tokens': 8000,
                'planning_model': 'qwen-plus',
                'planning_system_prompt':
                DEFAULT_SYSTEM_PROMPT_FOR_PLANNING_NODE,
                'planning_temperature': 0.7,
                'router_max_tokens': 8000,
                'router_model': 'qwen-plus',
                'router_system_prompt': DEFAULT_SYSTEM_PROMPT_FOR_ROUTER_NODE,
                'router_temperature': 0.7,
                'summary_max_tokens': 8000,
                'summary_model': 'qwen-plus',
                'summary_system_prompt':
                DEFAULT_SYSTEM_PROMPT_FOR_SUMMARY_NODE,
                'summary_temperature': 0.7
            }
        }

        # 执行智能体
        result = await graph.ainvoke(
            {"messages": [OpenAIMessage.user_message(content=user_message)]},
            config=runnable_config)

        # 提取回复内容
        if "messages" in result and result["messages"]:
            response = result["messages"][-1].content
        else:
            response = "抱歉，我无法生成回复。"

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
