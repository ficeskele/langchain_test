# ---------- imports ----------
from common_utils import (   
    msg_content,
    llm,
    search_tool,  
    BaseState, 
    split_by_newline,
)

from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# ---------- State ----------
class State(BaseState, total=False):
    search_queries: list[str]
    search_results: list[str]
    integrated_result: str
    approved: bool



# ------- Nodes -------
def gen_multi_queries(state: State):
    user_q = msg_content(state["messages"][-1])
    prompt = ("Generate EXACTLY three concise web-search queries from different perspectives:\n"
              f"{user_q}\n and ONLY bulletpoint don't use number and dont contain other opening.")
    raw = llm.invoke(prompt).content
    queries = [q.lstrip("-• ").strip() for q in raw.splitlines() if q.strip()]
    return {
        "search_queries": queries,
        "messages": [AIMessage(content="I could search for:\n" + "\n".join(f"• {q}" for q in queries))]
    }


def exec_multi_search(state: State):
    results = []
    sq = state["search_queries"]

    print(f"\nAssistant: Search queries:\n{sq}\n")
    for q in sq:
        res = search_tool.run(q)
        results.append(f"### {q}\n{res}")
    return {"search_results": results}


def integrate_results(state: State):
    """把多條結果串起來（可自行改寫成向量融合理證）"""
    integrated = "\n\n".join(state["search_results"])
    return {"integrated_result": integrated}

def generate_answer(state: State):
    user_q = msg_content(state["messages"][0])
    answer = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Use the search material to answer."),
        HumanMessage(content=user_q),
        AIMessage(content=f"Background material:\n{state['integrated_result']}"),
    ]).content
    return {"messages": [AIMessage(content=answer)]}

# ------- Graph -------
g = StateGraph(State)
g.add_node("gen_multi_queries",     gen_multi_queries)
g.add_node("execute_multi_search",  exec_multi_search)
g.add_node("integrate_results",     integrate_results)
g.add_node("generate_answer",       generate_answer)

g.add_edge(START, "gen_multi_queries")
g.add_edge("gen_multi_queries", "execute_multi_search")
g.add_edge("execute_multi_search", "integrate_results")
g.add_edge("integrate_results", "generate_answer")
g.add_edge("generate_answer", END)

chat_graph = g.compile()

png_bytes = chat_graph.get_graph().draw_mermaid_png()
with open("Level_3_AI_Agent_Workflow.png", "wb") as f:
    f.write(png_bytes)

print("✅ Level_3_AI_Agent_Workflow.png saved successfully.")


def chat_loop():
    while True:
        u = input("\nUser: ").strip()
        if u.lower() in {"quit","q","exit"}: print("Assistant: Bye!"); break
        for event in chat_graph.stream({"messages":[{"role":"user","content":u}]}):
            for p in event.values():
                if "messages" in p:
                    print("Assistant:", msg_content(p["messages"][-1]))

if __name__ == "__main__":
    chat_loop()