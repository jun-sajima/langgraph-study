import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from typing import Annotated, TypedDict, Sequence
    import operator
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
    )
    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import (
        ChatPromptTemplate,
    )
    from langgraph.graph import END, StateGraph
    import functools

    import os
    from dotenv import load_dotenv
    return (
        AIMessage,
        Annotated,
        AzureChatOpenAI,
        BaseMessage,
        ChatPromptTemplate,
        END,
        HumanMessage,
        Sequence,
        StateGraph,
        SystemMessage,
        TypedDict,
        functools,
        load_dotenv,
        operator,
        os,
    )


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(ChatPromptTemplate, HumanMessage, SystemMessage):
    salesman_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "あなたは熱意ある壺のベテラン訪問販売員、坪田壺夫です。"
                "営業が終了したら、「FINISH」と回答してください。"
                "壺が売れるか、売れる見込みがない場合に営業を終了してください。"
            ),
            HumanMessage(content="こんにちは。どちら様でしょうか？"),
            ("placeholder", "{messages}"),
        ]
    )

    shed_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("あなたは堅実な主夫の堅木実です。"),
            ("placeholder", "{messages}"),
        ]
    )
    return salesman_prompt, shed_prompt


@app.cell
def _(AzureChatOpenAI, os, salesman_prompt, shed_prompt):
    llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    )


    salesman_agent = salesman_prompt | llm
    shed_agent = shed_prompt | llm
    return salesman_agent, shed_agent


@app.cell
def _(Annotated, BaseMessage, Sequence, TypedDict, operator):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
    return (AgentState,)


@app.cell
def _(AIMessage):
    def agent_node(state, agent, name):
        result = agent.invoke(state)
        message = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
        print(f"{name}: {message.content}")
        return {"messages": [message]}
    return (agent_node,)


@app.function
def route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "FINISH" in last_message.content:
        return "finish"
    return "continue"


@app.cell
def _(agent_node, functools, salesman_agent, shed_agent):
    salesman_node = functools.partial(agent_node, agent=salesman_agent, name="Salesman")
    shed_node = functools.partial(agent_node, agent=shed_agent, name="SHED")
    return salesman_node, shed_node


@app.cell
def _(AgentState, END, StateGraph, salesman_node, shed_node):
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
    graph.invoke({"messages": []})
    return


if __name__ == "__main__":
    app.run()
