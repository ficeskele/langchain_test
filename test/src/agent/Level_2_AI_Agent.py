# ---------- imports ----------
from .common_utils import (   
    msg_content,
    llm,
    search_tool,  
    BaseState,    
)

from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ---------- State ----------
class State(BaseState, total=False):  
    search_query: str
    approved: bool
    search_result: str


graph_builder = StateGraph(State)

# ------- Nodes -------
def generate_search_query(state: State):
    user_q = msg_content(state["messages"][-1])
    sq     = llm.invoke(f"Generate a concise web search query for: {user_q}, plz text only dont do bullet point.").content.strip()
    return {
        "search_query": sq,
        "messages": [AIMessage(content=f"I could search for **{sq}**")]
    }



def ask_user(state: State):
    sq = state["search_query"]
    
    ans = input(f"\nAssistant: Search the web for \"{sq}\"? (y/n) ").lower()
    ok  = ans in {"y", "yes"}
    return {"approved": ok,
            "messages": [{"role":"assistant",
                          "content": "Great, searching…" if ok else
                                     "Okay, I won’t search. Ask something else."}]}

def execute_search(state: State):
    res = search_tool.run(state["search_query"])       
    return {"search_result": res}

def generate_answer(state: State):
    user_q = msg_content(state["messages"][0])
    
    res    = state["search_result"]
    print(f"\nAssistant: Search result:\n{res}\n")
    ans    = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Use the search result to answer."),
        HumanMessage(content=user_q),
        AIMessage(content=f"Search result:\n{res}")
    ]).content
    return {"messages":[AIMessage(content=ans)]}

def interrupt_process(state: State):
    return {"messages":[{"role":"assistant",
                         "content":"Understood. What else would you like to ask?"}]}

# ------- Graph -------
g = StateGraph(State)
g.add_node("generate_search_query", generate_search_query)
g.add_node("ask_user",              ask_user)
g.add_node("execute_search",        execute_search)
g.add_node("generate_answer",       generate_answer)
g.add_node("interrupt_process",     interrupt_process)

g.add_edge(START, "generate_search_query")
g.add_edge("generate_search_query", "ask_user")

g.add_conditional_edges("ask_user",
                        lambda s: "execute_search" if s.get("approved") else "interrupt_process")

g.add_edge("execute_search", "generate_answer")
g.add_edge("generate_answer", END)
g.add_edge("interrupt_process", END)

chat_graph = g.compile()

png_bytes = chat_graph.get_graph().draw_mermaid_png()
with open("Level_2_AI_Agent_Workflow.png", "wb") as f:
    f.write(png_bytes)

print("✅ Level_2_AI_Agent_Workflow.png saved successfully.")


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