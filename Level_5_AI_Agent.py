# -*- coding: utf-8 -*-
"""LangGraph Level-5 Search-and-Chat Agent with Debug Prints
============================================================
• 新增 `canon()` 正規化函式 → 改善 `same_question` 與查詢組合比對
• 在 `check_relevance()` 加入除錯輸出：顯示 `related` / `same_question`
"""

from __future__ import annotations

from typing import Dict, Tuple
import re, unicodedata

from common_utils import (
    msg_content,
    llm,
    search_tool,
    BaseState,
)
from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ---------- 文字正規化 ----------

def canon(q: str) -> str:
    """半形化 + 去多空白 + 小寫。"""
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q.lower()

# ---------- State ----------
class State(BaseState, total=False):
    search_queries: list[str]
    search_results: list[str]
    integrated_result: str
    approved: bool

    current_q: str
    past_q: str
    past_integrated: str
    past_search_results: list[str]
    past_answer: str
    relevance: bool
    same_question: bool

    qa_by_queries: Dict[Tuple[str, ...], Dict[str, str]]
    queries_seen: bool
    known_answer: str
    known_integrated: str

# ---------- Graph nodes ----------

def gen_multi_queries(state: State):
    user_q = state["current_q"]
    prompt = (
        "Generate EXACTLY three concise web-search queries from different perspectives:\n"
        f"{user_q}\n"
        "ONLY bulletpoint. Don't use numbers and don't add any other text."
    )
    raw = llm.invoke(prompt).content
    queries = [q.lstrip("-•* ").strip() for q in raw.splitlines() if q.strip()]
    return {
        "search_queries": queries,
        "messages": [AIMessage(content="I could search for:\n" + "\n".join(f"• {q}" for q in queries))],
    }


def check_memory(state: State):
    qs_key = tuple(canon(q) for q in state["search_queries"])
    qa_map = state.get("qa_by_queries", {})
    if qs_key in qa_map:
        entry = qa_map[qs_key]
        return {
            "queries_seen": True,
            "known_answer": entry["answer"],
            "known_integrated": entry["integrated"],
        }
    return {"queries_seen": False}


def show_known_answer(state: State):
    return {
        "messages": [AIMessage(content=state["known_answer"])],
        "integrated_result": state["known_integrated"],
    }


def exec_multi_search(state: State):
    results = [f"### {q}\n{search_tool.run(q)}" for q in state["search_queries"]]
    return {"search_results": results}


def integrate_results(state: State):
    new_int = "\n\n".join(state["search_results"])
    integrated = (
        state.get("past_integrated", "") + "\n\n" + new_int
        if state.get("relevance") else new_int
    )
    return {"integrated_result": integrated}


def generate_answer(state: State):
    answer = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Use the search material to answer."),
        HumanMessage(content=state["current_q"]),
        AIMessage(content=f"Background material:\n{state['integrated_result']}"),
    ]).content
    return {"messages": [AIMessage(content=answer)]}


def same_answer(state: State):
    if state.get("same_question") and state.get("past_answer"):
        return {"messages": [AIMessage(content=state["past_answer"])]}
    return {}


def check_relevance(state: State):
    user_q = msg_content(state["messages"][-1])
    last_q = state.get("past_q", "")

    same_question = canon(user_q) == canon(last_q)
    if not last_q:
        relevance = False
    else:
        relevance = llm.invoke(
            f"Are these two questions about the same topic? '{last_q}' ### '{user_q}'. "
            "Answer yes or no."
        ).content.strip().lower().startswith("y")

    # --- Debug print ---
    print(f"\nAssistant Debug → related={relevance}, same_question={same_question}\n")

    return {
        "relevance": relevance,
        "same_question": same_question,
        "current_q": user_q,
    }


def update_memory(state: State):
    qa_map = state.get("qa_by_queries", {})
    if "search_queries" in state and "integrated_result" in state and "messages" in state:
        qa_map[tuple(canon(q) for q in state["search_queries"])] = {
            "answer": msg_content(state["messages"][-1]),
            "integrated": state["integrated_result"],
        }
    return {
        "past_q": state["current_q"],
        "past_integrated": state.get("integrated_result", state.get("past_integrated", "")),
        "past_search_results": state.get("search_results", state.get("past_search_results", [])),
        "past_answer": msg_content(state["messages"][-1]),
        "qa_by_queries": qa_map,
    }

# ---------- Build graph ----------
g = StateGraph(State)
for n, fn in [
    ("check_relevance", check_relevance),
    ("gen_multi_queries", gen_multi_queries),
    ("check_memory", check_memory),
    ("show_known_answer", show_known_answer),
    ("execute_multi_search", exec_multi_search),
    ("integrate_results", integrate_results),
    ("generate_answer", generate_answer),
    ("same_answer", same_answer),
    ("update_memory", update_memory),
]:
    g.add_node(n, fn)

g.add_edge(START, "check_relevance")

g.add_conditional_edges("check_relevance", lambda s: "same_answer" if s["same_question"] else "gen_multi_queries")

g.add_edge("gen_multi_queries", "check_memory")

g.add_conditional_edges("check_memory", lambda s: "show_known_answer" if s["queries_seen"] else "execute_multi_search")

g.add_edge("execute_multi_search", "integrate_results")

g.add_edge("integrate_results", "generate_answer")

g.add_edge("generate_answer", "update_memory")

g.add_edge("show_known_answer", "update_memory")

g.add_edge("same_answer", "update_memory")

g.add_edge("update_memory", END)

chat_graph = g.compile()

# ---------- CLI ----------
if __name__ == "__main__":
    from pathlib import Path
    Path("Level_5_AI_Agent_Workflow.png").write_bytes(chat_graph.get_graph().draw_mermaid_png())
    print("✅ Level_5_AI_Agent_Workflow.png saved successfully.")
    session_state: dict = {}
    while True:
        try:
            u = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Bye!"); break
        if u.lower().strip() in {"q", "quit", "exit"}: break
        if not u.strip(): continue
        input_state = {"messages": [{"role": "user", "content": u}], **session_state}
        final_state = None
        for event in chat_graph.stream(input_state):
            node_state = event.get("data") or next(iter(event.values()), None)
            if isinstance(node_state, dict) and "messages" in node_state:
                print("Assistant:", msg_content(node_state["messages"][-1]))
            if isinstance(node_state, dict):
                final_state = node_state
        if final_state:
            session_state = {k: final_state[k] for k in ("past_q", "past_integrated", "past_search_results", "past_answer", "qa_by_queries") if k in final_state}
