"""
Base class for language models.
"""

from langchain_core.language_models import BaseChatModel


class BaseLLM(BaseChatModel):
    """Base class for all LLM implementations in browser-use."""
    pass 