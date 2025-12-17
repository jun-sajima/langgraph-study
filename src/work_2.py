import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import operator
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
    from langchain_openai import AzureChatOpenAI
    from langgraph.graph import END, StateGraph
    from langchain_core.tools import tool
    import subprocess

    import os
    from dotenv import load_dotenv
    return (
        Annotated,
        AzureChatOpenAI,
        BaseMessage,
        END,
        HumanMessage,
        Sequence,
        StateGraph,
        ToolMessage,
        TypedDict,
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
    @tool
    def exec_command(shell_command: str) -> str:
        """シェルコマンドを実行します。
        shell_command: Linuxシェルコマンド
        """
        result = subprocess.run(shell_command, shell=True, capture_output=True)
        return result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
    return (exec_command,)


@app.cell
def _(Annotated, BaseMessage, Sequence, TypedDict, operator):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
    return (AgentState,)


@app.cell
def _(AzureChatOpenAI, exec_command, os):
    llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        temperature=0,
    )
    llm_with_tool = llm.bind_tools([exec_command])
    return (llm_with_tool,)


@app.cell
def _(AgentState, llm_with_tool):
    def agent_node(state: AgentState):
        messages = state["messages"]
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}
    return (agent_node,)


@app.cell
def _(AgentState, ToolMessage, exec_command):
    def tool_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        messages = []
        for call in last_message.tool_calls:
            if call["name"] == "exec_command":
                value = exec_command.invoke(call["args"])
                tool_message = ToolMessage(
                    content=value,
                    name=call["name"],
                    tool_call_id=call["id"],
                )
                messages.append(tool_message)
        return {"messages": messages}
    return (tool_node,)


@app.cell
def _(AgentState):
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tool"
        else:
            return "end"
    return (should_continue,)


@app.cell
def _(AgentState, END, StateGraph, agent_node, should_continue, tool_node):
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
    return (graph,)


@app.cell
def _(HumanMessage, graph):
    query = input("query: ")
    state = graph.invoke({"messages": [HumanMessage(content=query)]})
    print(state["messages"][-1].content)
    return


if __name__ == "__main__":
    app.run()
