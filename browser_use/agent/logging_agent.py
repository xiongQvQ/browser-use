"""
Agent class with enhanced logging capabilities.
"""

import asyncio
import logging
import traceback
from typing import List

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult
from browser_use.agent.gif_generator import GIFGenerator

logger = logging.getLogger(__name__)


class AgentWithLogging(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_gif = True  # Enable GIF generation
        self.step_timeouts = {}  # Track step timeouts
        self.step_start_times = {}  # Track step start times

    async def step(self, step_info=None):
        try:
            step_start = asyncio.get_event_loop().time()
            self.step_start_times[self.n_steps] = step_start
            logger.info("Starting step %d", self.n_steps)

            # Create a task with timeout
            try:
                async with asyncio.timeout(60):  # 60 seconds timeout
                    await super().step(step_info)
                    step_end = asyncio.get_event_loop().time()
                    duration = step_end - step_start
                    logger.info("Completed step %d in %.2f seconds", self.n_steps, duration)
            except asyncio.TimeoutError:
                logger.error("Step %d timed out after 60 seconds", self.n_steps)
                self.step_timeouts[self.n_steps] = True
                raise TimeoutError(f"Step {self.n_steps} timed out")

        except Exception as e:
            step_end = asyncio.get_event_loop().time()
            duration = step_end - step_start
            logger.error("Error in step %d after %.2f seconds: %s\nTraceback:\n%s", 
                self.n_steps, duration, str(e), traceback.format_exc())
            raise

    def _handle_step_error(self, error: Exception) -> List[ActionResult]:
        if isinstance(error, TimeoutError):
            logger.error("Step %d timed out. Returning error result.", self.n_steps)
            return [ActionResult(error=f"Step {self.n_steps} timed out after 60 seconds")]
        
        logger.error("Handling step error: %s\nTraceback:\n%s", 
            str(error), traceback.format_exc())
        return super()._handle_step_error(error)

    def create_history_gif(self, output_path: str = 'agent_history.gif', **kwargs):
        """Create a GIF from the agent's history with overlaid task and goal text."""
        if not self.generate_gif:
            logger.info("GIF generation is disabled")
            return

        GIFGenerator.create_history_gif(
            history=self.history.history,
            task=self.task,
            output_path=output_path,
            **kwargs
        ) 