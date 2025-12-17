"""
LangGraph可視化スクリプト
work_1.py ~ work_4.py の各グラフをPNG形式で出力します。
"""
import os
from pathlib import Path


def visualize_work1():
    """work_1.py: 貯金箱シミュレーション"""
    from typing import Annotated, TypedDict
    import operator
    from langgraph.graph import StateGraph, END
    import functools

    class PiggyBankState(TypedDict):
        total: Annotated[int, operator.add]
        count: Annotated[int, operator.add]
        last_deposit: int

    def deposit(state: PiggyBankState) -> dict:
        return {"total": 100, "count": 1, "last_deposit": 100}

    def finalize(state: PiggyBankState) -> dict:
        return {"total": 0}

    def check_goal(state: PiggyBankState, goal: int) -> str:
        if state["total"] >= goal:
            return "full"
        else:
            return "continue"

    workflow = StateGraph(PiggyBankState)
    workflow.add_node("Deposit", deposit)
    workflow.add_node("Full", finalize)
    workflow.add_conditional_edges(
        "Deposit",
        functools.partial(check_goal, goal=1000),
        {"continue": "Deposit", "full": "Full"},
    )
    workflow.add_edge("Full", END)
    workflow.set_entry_point("Deposit")
    graph = workflow.compile()

    return graph


def visualize_work2():
    """work_2.py: シェルコマンド実行エージェント"""
    import operator
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def agent_node(state: AgentState):
        return {"messages": []}

    def tool_node(state: AgentState):
        return {"messages": []}

    def should_continue(state: AgentState):
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("Agent", agent_node)
    workflow.add_node("Tool", tool_node)
    workflow.add_conditional_edges(
        "Agent",
        should_continue,
        {
            "tool": "Tool",
            "end": END,
        },
    )
    workflow.add_edge("Tool", "Agent")
    workflow.set_entry_point("Agent")
    graph = workflow.compile()

    return graph


def visualize_work3():
    """work_3.py: 販売員と主夫の会話"""
    from typing import Annotated, TypedDict, Sequence
    import operator
    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def salesman_node(state):
        return {"messages": []}

    def shed_node(state):
        return {"messages": []}

    def route(state):
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("Salesman", salesman_node)
    workflow.add_node("SHED", shed_node)
    workflow.add_conditional_edges(
        "Salesman",
        route,
        {
            "continue": "SHED",
            "finish": END,
        },
    )
    workflow.add_edge("SHED", "Salesman")
    workflow.set_entry_point("Salesman")
    graph = workflow.compile()

    return graph


def visualize_work4():
    """work_4.py: ソフトウェア開発チーム"""
    import operator
    from typing import Annotated, Sequence, TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next: str
        task: str

    def leader_node(state: AgentState) -> dict:
        return {"messages": [], "next": "Finish"}

    def programmer_node(state: AgentState) -> dict:
        return {"messages": []}

    def tester_node(state: AgentState) -> dict:
        return {"messages": []}

    def evaluator_node(state: AgentState) -> dict:
        return {"messages": []}

    def tool_node(state: AgentState) -> dict:
        return {"messages": []}

    def router(state: AgentState) -> str:
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("Leader", leader_node)
    workflow.add_node("Evaluator", evaluator_node)
    workflow.add_node("Tool", tool_node)
    workflow.add_node("Programmer", programmer_node)
    workflow.add_node("TestWriter", tester_node)
    workflow.add_conditional_edges(
        "Evaluator", router, {"continue": "Leader", "call_tool": "Tool"}
    )
    workflow.add_conditional_edges(
        "Leader",
        lambda x: x["next"],
        {
            "Programmer": "Programmer",
            "TestWriter": "TestWriter",
            "Evaluator": "Evaluator",
            "Finish": END,
        },
    )
    workflow.add_edge("Programmer", "Leader")
    workflow.add_edge("TestWriter", "Leader")
    workflow.add_edge("Tool", "Evaluator")
    workflow.set_entry_point("Leader")
    graph = workflow.compile()

    return graph


def main():
    """各グラフを可視化してPNGファイルとして保存"""
    # 出力ディレクトリを作成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    workflows = {
        "work_1_piggy_bank": (visualize_work1, "貯金箱シミュレーション"),
        "work_2_shell_agent": (visualize_work2, "シェルコマンド実行エージェント"),
        "work_3_conversation": (visualize_work3, "販売員と主夫の会話"),
        "work_4_dev_team": (visualize_work4, "ソフトウェア開発チーム"),
    }

    for filename, (viz_func, description) in workflows.items():
        try:
            print(f"生成中: {description} ({filename})...")
            graph = viz_func()
            png_data = graph.get_graph(xray=True).draw_mermaid_png()

            output_path = output_dir / f"{filename}.png"
            with open(output_path, "wb") as f:
                f.write(png_data)

            print(f"✓ 保存完了: {output_path}")
        except Exception as e:
            print(f"✗ エラー ({filename}): {e}")

    print(f"\n完了！すべての図は {output_dir}/ に保存されました。")


if __name__ == "__main__":
    main()
