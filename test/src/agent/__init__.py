# --- public helpers ---------------------------------------------------------
from .Level_1_AI_Agent import chat_graph as level1_graph
from .Level_2_AI_Agent import chat_graph as level2_graph
from .Level_3_AI_Agent import chat_graph as level3_graph
from .Level_4_AI_Agent import chat_graph as level4_graph
from .Level_5_AI_Agent import chat_graph as level5_graph

# Choose ONE to be the “default” that the template expects
graph = level3_graph            # <— change to whichever you want to demo first


def get_graph(name: str):
    """
    Return a graph object by short name:
        get_graph("level4") → level4_graph
    """
    lookup = {
        "level1": level1_graph,
        "level2": level2_graph,
        "level3": level3_graph,
        "level4": level4_graph,
        "level5": level5_graph,
    }
    return lookup[name]


__all__ = [
    "level1_graph",
    "level2_graph",
    "level3_graph",
    "level4_graph",
    "level5_graph",
    "graph",
    "get_graph",
]