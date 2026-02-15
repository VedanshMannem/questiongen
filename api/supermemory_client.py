import os
from typing import Any, Dict, List, Optional

try:
    from supermemory import Supermemory  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    Supermemory = None  # type: ignore

SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")

class SupermemoryError(Exception):
    pass

def _get_client() -> Optional[Any]:
    if Supermemory is None:
        return None
    try:
        return Supermemory()
    except Exception:
        return None

def store_user_memory(container_tag: str, content: str) -> bool:
    client = _get_client()
    if client is None:
        return False

    if not SUPERMEMORY_API_KEY:
        return False

    try:
        client.add(content=content, container_tag=container_tag)
        return True
    except Exception:
        return False

def build_user_context(container_tag: str, query: str, threshold: Optional[float] = None) -> str:
    client = _get_client()
    if client is None:
        return ""

    if not SUPERMEMORY_API_KEY:
        return ""

    try:
        kwargs: Dict[str, Any] = {"container_tag": container_tag, "q": query}
        if threshold is not None:
            kwargs["threshold"] = threshold
        profile = client.profile(**kwargs)

        static = []
        dynamic = []
        results: List[str] = []

        if hasattr(profile, "profile"):
            static = list(getattr(profile.profile, "static", []) or [])
            dynamic = list(getattr(profile.profile, "dynamic", []) or [])

        search_results = getattr(profile, "search_results", None)
        if search_results and hasattr(search_results, "results"):
            for item in search_results.results:
                memory = item.get("memory") if isinstance(item, dict) else None
                if memory:
                    results.append(memory)

        context = []
        if static:
            context.append("Static profile:\n" + "\n".join(static))
        if dynamic:
            context.append("Dynamic profile:\n" + "\n".join(dynamic))
        if results:
            context.append("Relevant memories:\n" + "\n".join(results))

        return "\n\n".join(context).strip()
    except Exception:
        return ""
