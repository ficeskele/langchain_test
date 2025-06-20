# -*- coding: utf-8 -*-
"""LangGraph Level-4 Search-and-Chat Agent
================================================

Handles follow-up questions by (1) detecting topical relevance, (2) deciding
whether the new question is exactly the *same* as the previous one, and
(3) when the topic is related but *not* identical, re-using the previously
integrated search material **and** enriching it with fresh search results.

If the new question is unrelated, the graph falls back to a full fresh
search cycle.
"""

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
    """Execution state propagated along the graph."""

    # Level-3 -------------------------------------------------------------
    search_queries: list[str]
    search_results: list[str]
    integrated_result: str
    approved: bool

    # Level-4 -------------------------------------------------------------
    current_q: str               # 本輪 user 問題
    past_q: str                  # 上一輪 user 問題
    past_integrated: str         # 上一輪整合後的素材
    past_search_results: list[str]
    past_answer: str             # 上一輪回答（可重用）

    relevance: bool              # 與上一輪是否同主題
    same_question: bool          # 與上一輪是否「完全同一句話」


# ---------- Graph nodes ----------

def gen_multi_queries(state: State):
    """使用 LLM 從不同視角為 *current_q* 產出三條搜尋關鍵字。"""
    user_q = state["current_q"]
    prompt = (
        "Generate EXACTLY three concise web-search queries from different perspectives:\n"
        f"{user_q}\n"
        "ONLY bulletpoint. Don't use numbers and don't add any other text."
    )
    raw = llm.invoke(prompt).content
    queries = [q.lstrip("-•* ").strip() for q in raw.splitlines() if q.strip()]

    # 向使用者顯示我們將進行的搜尋
    return {
        "search_queries": queries,
        "messages": [
            AIMessage(content="I could search for:\n" + "\n".join(f"• {q}" for q in queries))
        ],
    }


def exec_multi_search(state: State):
    """逐條執行搜尋並收集結果。"""
    results = []
    sq = state["search_queries"]

    print(f"\nAssistant: Search queries → {sq}\n")
    for q in sq:
        res = search_tool.run(q)
        results.append(f"### {q}\n{res}")
    return {"search_results": results}


def integrate_results(state: State):
    """把新 & 舊搜尋結果整合起來。"""
    # 新產生的素材
    new_integrated = "\n\n".join(state["search_results"])

    # 如果與上一輪相關，則將舊素材併入
    if state.get("relevance"):
        integrated = state.get("past_integrated", "") + "\n\n" + new_integrated
    else:
        integrated = new_integrated
    return {"integrated_result": integrated}


def generate_answer(state: State):
    """用整合後的素材回答 *current_q*。"""
    answer = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant. Use the search material to answer."),
            HumanMessage(content=state["current_q"]),
            AIMessage(content=f"Background material:\n{state['integrated_result']}"),
        ]
    ).content
    return {"messages": [AIMessage(content=answer)]}


def same_answer(state: State):
    """若問題完全相同且已有答案，直接重播；否則不處理。"""
    if state.get("same_question") and state.get("past_answer"):
        return {"messages": [AIMessage(content=state["past_answer"])]}
    # no -op so that the graph continues to next node
    return {}


def check_relevance(state: State):
    """判斷新問題與上一輪的相關性 & 是否為重覆問題。"""
    user_q = msg_content(state["messages"][-1])
    last_q = state.get("past_q", "")

    same_question = user_q.strip().lower() == last_q.strip().lower()

    if not last_q:
        # 第一次互動直接標記為不相關
        relevance = False
    else:
        relevance = llm.invoke(
            f"Are these two questions about the same topic? '{last_q}' ### '{user_q}'. "
            "Answer yes or no."
        ).content.strip().lower().startswith("y")

    print(f"\nAssistant: related={relevance}, same_question={same_question}\n")
    return {
        "relevance": relevance,
        "same_question": same_question,
        "current_q": user_q,
    }


def update_memory(state: State):
    """把這一輪的重要資訊寫回 *session_state*。"""
    return {
        "past_q": state["current_q"],
        "past_integrated": state.get("integrated_result", state.get("past_integrated", "")),
        "past_search_results": state.get(
            "search_results", state.get("past_search_results", [])
        ),
        "past_answer": msg_content(state["messages"][-1]),
    }


# ---------- Assemble the LangGraph ----------

g = StateGraph(State)

# Nodes
for name, fn in [
    ("check_relevance", check_relevance),
    ("gen_multi_queries", gen_multi_queries),
    ("execute_multi_search", exec_multi_search),
    ("integrate_results", integrate_results),
    ("generate_answer", generate_answer),
    ("same_answer", same_answer),
    ("update_memory", update_memory),
]:
    g.add_node(name, fn)

# Edges
#g.add_edge(START, "check_relevance")   ← added automatically below

g.add_edge(START, "check_relevance")

# 根據相關性 / 重覆性走不同分支

g.add_conditional_edges(
    "check_relevance",
    # 判斷下一步
    lambda s: "same_answer" if s["same_question"] else "gen_multi_queries",
)

# 如果重播答案，仍然要寫回 memory

g.add_edge("same_answer", "update_memory")

# Fresh / follow-up 搜尋流程

g.add_edge("gen_multi_queries", "execute_multi_search")

g.add_edge("execute_multi_search", "integrate_results")

g.add_edge("integrate_results", "generate_answer")

g.add_edge("generate_answer", "update_memory")

# 終點

g.add_edge("update_memory", END)

chat_graph = g.compile()

# ---------- CLI loop (simple demo) ----------

png_bytes = chat_graph.get_graph().draw_mermaid_png()
with open("Level_4_AI_Agent_Workflow.png", "wb") as f:
    f.write(png_bytes)
print("✅ Level_4_AI_Agent_Workflow.png saved successfully.")

session_state: dict = {}
def chat_loop():
    """簡易互動回圈。"""
    global session_state

    while True:
        try:
            u = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Bye!")
            break

        if u.lower() in {"quit", "q", "exit"}:
            print("Assistant: Bye!")
            break

        # 把上一輪記憶加入輸入 state
        input_state = {"messages": [{"role": "user", "content": u}], **session_state}

        final_state = None
        for event in chat_graph.stream(input_state):
            # event 可能只有 'data'，也可能是 {'node':..., 'type':..., 'value':...}
            node_state = event.get("data") or next(iter(event.values()), None)

            # 有 assistant 訊息就印
            if isinstance(node_state, dict) and "messages" in node_state:
                print("Assistant:", msg_content(node_state["messages"][-1]))

            # 保留最後一個有效 state
            if isinstance(node_state, dict):
                final_state = node_state

        # 更新記憶
        if final_state:
            session_state = {
                k: final_state[k]
                for k in (
                    "past_q",
                    "past_integrated",
                    "past_search_results",
                    "past_answer",
                )
                if k in final_state
            }


if __name__ == "__main__":
    chat_loop()
