import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import functools
    import json
    import subprocess
    import operator
    from typing import Annotated, Sequence, TypedDict, Union, Literal
    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    import os
    from dotenv import load_dotenv
    return (
        AIMessage,
        Annotated,
        AzureChatOpenAI,
        BaseMessage,
        BaseModel,
        ChatPromptTemplate,
        END,
        Field,
        HumanMessage,
        HumanMessagePromptTemplate,
        Literal,
        MessagesPlaceholder,
        Sequence,
        StateGraph,
        SystemMessagePromptTemplate,
        ToolNode,
        TypedDict,
        Union,
        functools,
        json,
        load_dotenv,
        operator,
        os,
        subprocess,
        tool,
    )


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(subprocess, tool):
    # ツール定義
    @tool
    def evaluate(code: str, test: str) -> tuple[str, str]:
        """
        コードをテストし、標準出力と標準エラーのペアを返す。
        code: テスト対象のコード
        test: pytestのテストコード
        """
        with open("product.py", "w") as f:
            f.write(code)
        with open("test_product.py", "w") as f:
            f.write(test)
        result = subprocess.run(["pytest", "test_product.py"], capture_output=True)
        return result.stdout.decode(), result.stderr.decode()
    return (evaluate,)


@app.cell
def _(Annotated, BaseMessage, Sequence, TypedDict, operator):
    # 状態管理
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next: str
        task: str
    return (AgentState,)


@app.cell
def _(BaseModel, Field, Literal, Union):
    class LeaderResponse(BaseModel):
        reasoning: str = Field(description="決定の背後にある理由。")
        next: Union[Literal["Finish"], str] = Field(
            description="次にアクションを行うチームメンバ。"
        )
        instructions: str = Field(description="次のチームメンバへの指示。")
    return (LeaderResponse,)


@app.cell
def _():
    MEMBERS = {
        "Leader": (
            "チームの進行を管理し、チームメンバの役割を割り当てます。"
            "チームの目標を達成した場合はFinishを選択します。\n",
        ),
        "Programmer": "仕様に基づいてコードを書きます。\n",
        "TestWriter": "仕様に基づいてテストを書きます。\n",
        "Evaluator": (
            "コードとテストを実行し、テスト結果を分析します。"
            "コードとテストを実行するには、evaluateツールを使ってください。\n",
        ),
    }
    return (MEMBERS,)


@app.cell
def _(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MEMBERS,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
):
    def create_agent(llm, name: str) -> ChatPromptTemplate:
        """特定の役割のためのプロンプトテンプレートを作成する。"""
        members = ", ".join(MEMBERS.keys())
        member_roles = "\n".join(f"{member}: {role}" for member, role in MEMBERS.items())
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "あなたはソフトウェア開発チームの一員の{name}です。"
                    "チームの最終成果物は、仕様を満たすことを確認したテスト済みのコードです。"
                    "コードはproduct.pyに保存されるものとします。"
                    "テストはtest_product.pyに保存されるものとします。"
                    "必ずステップバイステップで推論の過程を説明してから答えてください。\n\n"
                    "チームメンバ: {members}\n\n"
                    "{member_roles}"
                ),
                HumanMessagePromptTemplate.from_template("task: {task}\n"),
                MessagesPlaceholder("messages"),
            ]
        ).partial(name=name, members=members, member_roles=member_roles)
        return prompt | llm
    return (create_agent,)


@app.cell
def _(AzureChatOpenAI, LeaderResponse, create_agent, evaluate, os):
    llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_CODING_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    )

    leader_agent = create_agent(llm.with_structured_output(LeaderResponse), "Leader")
    programmer_agent = create_agent(llm, "Programmer")
    tester_agent = create_agent(llm, "TestWriter")
    evaluator_agent = create_agent(llm.bind_tools([evaluate]), "Evaluator")
    return evaluator_agent, leader_agent, programmer_agent, tester_agent


@app.cell
def _(AgentState, HumanMessage, leader_agent):
    def leader_node(state: AgentState) -> dict:
        """リーダエージェントのためのノード関数。"""
        response = leader_agent.invoke(state)
        return {
            "messages": [HumanMessage(content=response.instructions, name="Leader")],
            "next": response.next,
        }
    return (leader_node,)


@app.cell
def _(AIMessage, AgentState, evaluate):
    def member_node(state: AgentState, agent, name: str) -> dict:
        """汎用エージェントのためのノード関数。"""
        for _ in range(10):
            result = agent.invoke(state)
            has_errors = False
            if result.tool_calls:
                for call in result.tool_calls:
                    name = call["name"]
                    if name != "evaluate":
                        has_errors = True
                        print(f"無効なツール呼び出し: {name}")
                        break
                    args = call["args"]
                    try:
                        evaluate.args_schema.model_validate(args)
                    except Exception as e:
                        print(f"バリデーション失敗: {str(e)}")
                        has_errors = True
            if not has_errors:
                break
        else:
            raise ValueError("無効なツール呼び出しが続いたため、終了します。")
        return {
            "messages": [
                AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
            ]
        }
    return (member_node,)


@app.cell
def _(evaluator_agent, functools, member_node, programmer_agent, tester_agent):
    programmer_node = functools.partial(
        member_node, agent=programmer_agent, name="Programmer"
    )
    tester_node = functools.partial(member_node, agent=tester_agent, name="TestWriter")
    evaluator_node = functools.partial(member_node, agent=evaluator_agent, name="Evaluator")
    return evaluator_node, programmer_node, tester_node


@app.cell
def _(AgentState):
    def router(state: AgentState) -> str:
        """ワークフローの次のステップを決定するルータ関数。"""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "call_tool"
        return "continue"
    return (router,)


@app.cell
def _(
    AgentState,
    END,
    StateGraph,
    ToolNode,
    evaluate,
    evaluator_node,
    leader_node,
    programmer_node,
    router,
    tester_node,
):
    # ワークフロー定義
    workflow = StateGraph(AgentState)
    workflow.add_node("Leader", leader_node)
    workflow.add_node("Evaluator", evaluator_node)
    workflow.add_node("Tool", ToolNode([evaluate]))
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
    return (workflow,)


@app.cell
def _(json, workflow):
    # ワークフローの実行
    graph = workflow.compile()
    with open("src/HumanEval.jsonl") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines]
    entry = entries[0]
    return entry, graph


@app.cell
def _(entry, graph):
    result = graph.invoke({"messages": [], "task": entry["prompt"]})
    for message in result["messages"]:
        print(f"{message.name}: {message.content}")
    return


if __name__ == "__main__":
    app.run()
