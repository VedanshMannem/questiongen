import os
from typing import Any, Dict, List, Optional

SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")

class SupermemoryError(Exception):
    pass

def _get_client() -> Optional[Any]:
    return None

def store_user_memory(container_tag: str, content: str) -> bool:
    return False

def build_user_context(container_tag: str, query: str, threshold: Optional[float] = None) -> str:
    return ""
