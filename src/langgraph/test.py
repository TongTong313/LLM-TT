from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
# from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Dict, Any, List, Union


def add_messages(left: List[Any], right: Union[Any, List[Any]]) -> List[Any]:
    if isinstance(right, list):
        return left + right
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


class MyAgent:

    def __init__(self):
        self.graph = StateGraph(State)

    def create_graph(self) -> "CompiledStateGraph":
        self.graph.add_node("greet", greet)
        self.graph.add_node("end", end)
        self.graph.add_edge(START, "greet")
        self.graph.add_edge("greet", "end")
        self.graph.set_finish_point("end")
        return self.graph.compile()
