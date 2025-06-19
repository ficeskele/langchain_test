"""Common helpers & shared singletons for all LangGraph workflows.

Place this file at the root of your project (or package it as a module) and
`import` the utilities you need.  This keeps each workflow light-weight and
ensures you only spin up one LLM / search client per process.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.schema import BaseMessage

# ---------------------------------------------------------------------------
# 🏗️  Typed-state base class
# ---------------------------------------------------------------------------

class BaseState(TypedDict, total=False):
    """Minimal state definition: only the conversational message list.

    In每個 workflow 想再擴充別的欄位（例如 search_query、approved⋯），
    直接繼承 `BaseState` 再加欄位即可：

        class MyState(BaseState):
            search_query: str
            ...
    """

    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 🔧  Utility helpers
# ---------------------------------------------------------------------------

def msg_content(msg: BaseMessage | dict) -> str:
    """Return the text content from either a LangChain `BaseMessage` object
    or a raw `{"role": ..., "content": ...}` dict.
    """
    return msg["content"] if isinstance(msg, dict) else msg.content


# ---------------------------------------------------------------------------
# 🤖  Shared singletons (LLM & Search tool)
# ---------------------------------------------------------------------------

llm = init_chat_model("google_genai:gemini-2.0-flash")
search_tool = TavilySearch(max_results=10)
