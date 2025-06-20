# ---------- imports ----------
from common_utils import (   
    msg_content,
    llm,
    search_tool,  
    BaseState, 
)

from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage



# ---------- State ----------
class State(BaseState, total=False):
    search_queries: list[str]
    search_results: list[str]
    integrated_result: str
    approved: bool

    # Level 4 新增
    current_q: str 
    past_q: str                 # 上一輪的 user 問題
    past_integrated: str        # 上一輪整合材料
    relevance: bool             # 新問題與上一輪是否相關
    past_answer: str           # 上一輪的回答（可重用）



# ------- Nodes -------
def gen_multi_queries(state: State):
    user_q = msg_content(state["messages"][-1])
    prompt = ("Generate EXACTLY three concise web-search queries from different perspectives:\n"
              f"{user_q}\n and ONLY bulletpoint don't use number and dont contain other opening.")
    raw = llm.invoke(prompt).content
    queries = [q.lstrip("-•* ").strip() for q in raw.splitlines() if q.strip()]
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

def same_answer(state: State):
    """若有舊答案就直接重播；否則什麼都不做（讓流程繼續跑）"""
    old_answer = state.get("past_answer")
    if old_answer:
        return {"messages": [AIMessage(content=old_answer)]}
    return {}   # 必回傳 dict

def check_relevance(state: State):
    user_q  = msg_content(state["messages"][-1])
    last_q  = state.get("past_q", "")
    if not last_q:
        return {"relevance": False, "current_q": user_q}

    judge = llm.invoke(
        f"Are these two questions about the same topic? '{last_q}' ### '{user_q}'. "
        "Answer yes or no."
    ).content.strip().lower().startswith("y")
    print(f"\nAssistant: Is the new question related to the last one? {judge}\n")
    return {"relevance": judge, "current_q": user_q}

def reuse_material(state: State):
    # 直接沿用過去整合結果
    return {"integrated_result": state["past_integrated"]}

def update_memory(state: State):
    return {
        "past_q":          state["current_q"],          # ← 這才是使用者的問句
        "past_integrated": state.get("integrated_result",
                                      state.get("past_integrated", "")),
        "past_answer": msg_content(state["messages"][-1]),
    }

# ------- Graph -------
g = StateGraph(State)
g.add_node("gen_multi_queries",     gen_multi_queries)
g.add_node("execute_multi_search",  exec_multi_search)
g.add_node("integrate_results",     integrate_results)
g.add_node("generate_answer",       generate_answer)
g.add_node("check_relevance",       check_relevance)
g.add_node("reuse_material",        reuse_material)
g.add_node("update_memory",         update_memory)
g.add_node("same_answer",          same_answer)

g.add_edge(START, "check_relevance")

g.add_conditional_edges(
    "check_relevance",
    lambda s: "reuse_material" if s["relevance"] else "gen_multi_queries"
)

g.add_edge("reuse_material", "same_answer")
g.add_edge("same_answer", "update_memory")

g.add_edge("gen_multi_queries", "execute_multi_search")
g.add_edge("execute_multi_search", "integrate_results")
g.add_edge("integrate_results", "generate_answer")

g.add_edge("generate_answer", "update_memory")
g.add_edge("update_memory", END)

chat_graph = g.compile()

png_bytes = chat_graph.get_graph().draw_mermaid_png()
with open("Level_4_AI_Agent_Workflow.png", "wb") as f:
    f.write(png_bytes)

print("✅ Level_4_AI_Agent_Workflow.png saved successfully.")

session_state = {}   # 只要一行即可跨輪保存

def chat_loop():
    global session_state
    while True:
        u = input("\nUser: ").strip()
        if u.lower() in {"quit", "q", "exit"}:
            print("Assistant: Bye!")
            break

        # 把上一輪記憶加到輸入
        input_state = {"messages": [{"role": "user", "content": u}], **session_state}

        final_state = None
        for event in chat_graph.stream(input_state):
            # 每個 event 可能只有 'data'，也可能是 {'node':..., 'type':..., 'value':...}
            node_state = event.get("data") or next(iter(event.values()), None)

            # 有助理訊息就印出
            if isinstance(node_state, dict) and "messages" in node_state:
                print("Assistant:", msg_content(node_state["messages"][-1]))

            # 不論有沒有 messages，都覆寫為最後一次有效 state
            if isinstance(node_state, dict):
                final_state = node_state

        # 把記憶寫回 session_state
        if final_state:
            session_state = {
                k: final_state[k]
                for k in ("past_q", "past_integrated", "past_answer")   # ← 加這個
                if k in final_state
            }

if __name__ == "__main__":
    chat_loop()