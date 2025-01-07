"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
import logging
import traceback
import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Any, Optional, cast, List, Union
from browser_use import Agent, ActionResult
from browser_use.llm import DeepseekChat, DeepseekAgent

# Set up logging with more detailed format
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(),
		logging.FileHandler('agent_debug.log')  # Also save logs to file
	]
)
logger = logging.getLogger(__name__)

llm = DeepseekChat(
	temperature=0.7,
	max_tokens=2048  # Limit completion tokens
)

try:
	agent = DeepseekAgent(
		task='Go to google.com and search for "best phone under 200 dollar"',
		llm=llm,
		use_vision=False,  # Disable vision functionality
		max_failures=3,
		retry_delay=5,
		max_input_tokens=32768,  # Limit input tokens to half of model's context
		validate_output=True,
		max_actions_per_step=5,
		include_attributes=[
			'type',         # Input type
			'role',         # ARIA role
			'name',         # Element name
			'value',        # Current value
			'placeholder',  # Placeholder text
			'title',        # Title attribute
			'aria-label',   # ARIA label
			'id',          # Element ID
			'class',       # CSS classes
			'tabindex',    # Tab order
			'alt',         # Alt text
		]  # Include all attributes needed for input recognition
	)
	logger.info("Agent initialized successfully")
except Exception as e:
	logger.error("Error initializing agent: %s\nTraceback:\n%s", 
		str(e), traceback.format_exc())
	raise

async def main():
	try:
		logger.info("Starting agent execution")
		# 增加步骤数以确保完成搜索
		history = await agent.run(max_steps=5)
		
		# 打印每个步骤的动作
		logger.info("=== Action History ===")
		for i, item in enumerate(history.history, 1):
			logger.info(f"Step {i}:")
			if item.model_output:
				logger.info(f"Goal: {item.model_output.current_state.next_goal}")
				for action in item.model_output.action:
					logger.info(f"Action: {action.model_dump_json()}")
			logger.info("-" * 50)
		
		if agent.generate_gif:
			try:
				agent.create_history_gif()
				logger.info("History GIF created successfully")
			except Exception as e:
				logger.error("Failed to create history GIF: %s", str(e))
		logger.info("Agent execution completed")
	except Exception as e:
		logger.error("Error during agent execution: %s\nTraceback:\n%s", 
			str(e), traceback.format_exc())
		raise

if __name__ == "__main__":
	asyncio.run(main())
