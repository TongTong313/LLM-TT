from fastapi import FastAPI
from .tool_api import router as tool_router, init_llm
from ..agent.llm import LLM
import uvicorn

app = FastAPI()

# 注册路由
app.include_router(tool_router, prefix="/api/tools", tags=["tools"])

# 初始化 LLM 实例
llm = LLM(api_key="your_api_key", base_url="your_base_url")
init_llm(llm)


@app.get("/")
async def root():
    return {"message": "Welcome to MyManus API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
