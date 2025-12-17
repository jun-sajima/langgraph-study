import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from typing import Annotated, TypedDict
    import operator
    from langgraph.graph import StateGraph, END
    import functools
    return END, StateGraph, functools


app._unparsable_cell(
    r"""
    class PiggyBankState(TypedDict):
        total: Annotated[int, o       perator.add]
        count: Annotated[int, operator.add]
        last_deposit: int
    """,
    name="_"
)


@app.cell
def _(PiggyBankState):
    def deposit(state: PiggyBankState) -> dict:
        amount = int(input("Enter the amount to deposit: "))
        return {"total": amount, "count": 1, "last_deposit": amount}
    return (deposit,)


@app.cell
def _(PiggyBankState):
    def finalize(state: PiggyBankState) -> dict:
        print(f"{state['count']}回の貯金で目標金額に到達しました。")
        print(f"{state['total']}円貯まっています。")
        print(f"最後の入金額は{state['last_deposit']}円でした。")
        return {"total": 0}
    return (finalize,)


@app.cell
def _(PiggyBankState):
    def check_goal(state: PiggyBankState, goal: int) -> dict:
        if state["total"] >= goal:
            return "full"
        else:
            return "continue"

    return (check_goal,)


@app.cell
def _(
    END,
    PiggyBankState,
    StateGraph,
    check_goal,
    deposit,
    finalize,
    functools,
):
    def piggy_bank(goal: int):
        workflow = StateGraph(PiggyBankState)
        workflow.add_node("Deposit", deposit)
        workflow.add_node("Full", finalize)
        workflow.add_conditional_edges(
            "Deposit",
            functools.partial(check_goal, goal=goal),
            {"continue": "Deposit", "full": "Full"},
        )
        workflow.add_edge("Full", END)
        workflow.set_entry_point("Deposit")
        graph = workflow.compile()
        final_state = graph.invoke({"total": 0, "count": 0, "last_deposit": 0})
        print(final_state)
    return (piggy_bank,)


@app.cell
def _(piggy_bank):
    if __name__ == "__main__":
        piggy_bank(1000)
    return


if __name__ == "__main__":
    app.run()
