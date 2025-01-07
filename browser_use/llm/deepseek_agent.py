"""
Deepseek model specific agent implementation.
"""

import logging
import json
from typing import List, Any
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from browser_use.agent.logging_agent import AgentWithLogging
from browser_use.llm.deepseek import DeepseekChat

# 设置文件日志
file_handler = logging.FileHandler('agent_debug.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)


class DeepseekAgent(AgentWithLogging):
    """
    A specialized agent for Deepseek model that includes logging and GIF generation capabilities.
    This is needed because Deepseek model requires special handling for timeouts and error tracking.
    """
    def __init__(self, task: str, llm: BaseChatModel, **kwargs):
        if not isinstance(llm, DeepseekChat):
            raise ValueError("DeepseekAgent requires a DeepseekChat LLM instance")
        super().__init__(task=task, llm=llm, **kwargs)
        self.generate_gif = True  # Enable GIF generation by default for Deepseek
        
    async def get_next_action(self, input_messages: List[BaseMessage]) -> Any:
        """Override to add prompt logging."""
        # Log the messages being sent to the model
        logger.info("\n=== New Prompt Session ===")
        logger.info(f"Task: {self.task}")
        
        for i, msg in enumerate(input_messages):
            logger.info(f"\nMessage {i + 1} ({msg.__class__.__name__}):")
            # 完整保存消息内容到文件
            logger.info(f"Content: {str(msg.content)}")
            logger.info("-" * 100)
        
        # Call parent method with original messages
        return await super().get_next_action(input_messages) 