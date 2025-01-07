"""
Deepseek Chat model implementation for browser-use.
"""

import os
import json
import logging
import traceback
from typing import Any, List, Dict, Optional, cast, Iterable

from openai import OpenAI, APIStatusError
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field, PrivateAttr
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

from browser_use.llm.base import BaseLLM
from browser_use.agent.views import AgentOutput, AgentBrain

logger = logging.getLogger(__name__)


class DeepseekChat(BaseLLM):
    """
    A chat model implementation using the Deepseek API.
    """
    
    model: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    _client: OpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv('DEEPSEEKV3_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEKV3_API_KEY environment variable is not set")
        self._client = OpenAI(
            api_key=api_key,
            base_url='https://api.deepseek.com/v1'
        )
        logger.info("DeepseekChat initialized with model: %s", self.model)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("DeepseekChat only supports async generation")

    def _convert_message_to_dict(self, message: BaseMessage) -> ChatCompletionMessageParam:
        if isinstance(message, SystemMessage):
            return ChatCompletionSystemMessageParam(role="system", content=str(message.content))
        elif isinstance(message, HumanMessage):
            return ChatCompletionUserMessageParam(role="user", content=str(message.content))
        elif isinstance(message, AIMessage):
            return ChatCompletionAssistantMessageParam(role="assistant", content=str(message.content))
        elif isinstance(message, ChatMessage):
            if message.role == "system":
                return ChatCompletionSystemMessageParam(role="system", content=str(message.content))
            elif message.role == "assistant":
                return ChatCompletionAssistantMessageParam(role="assistant", content=str(message.content))
            else:
                return ChatCompletionUserMessageParam(role="user", content=str(message.content))
        else:
            logger.warning(f"Unknown message type {type(message)}, treating as user message")
            return ChatCompletionUserMessageParam(role="user", content=str(message.content))

    def _truncate_messages(self, messages: List[BaseMessage], max_tokens: int = 60000) -> List[BaseMessage]:
        total_tokens = 0
        truncated_messages = []
        
        if messages and isinstance(messages[0], SystemMessage):
            truncated_messages.append(messages[0])
            messages = messages[1:]
        
        for message in reversed(messages):
            content_length = len(str(message.content)) // 4
            if total_tokens + content_length > max_tokens:
                break
            total_tokens += content_length
            truncated_messages.insert(1 if len(truncated_messages) > 0 else 0, message)
        
        if len(truncated_messages) < len(messages):
            logger.warning(f"Truncated {len(messages) - len(truncated_messages)} messages to fit within token limit")
        
        return truncated_messages

    async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        try:
            truncated_messages = self._truncate_messages(messages)
            message_dicts: List[ChatCompletionMessageParam] = []
            
            for i, message in enumerate(truncated_messages):
                try:
                    msg_dict = self._convert_message_to_dict(message)
                    message_dicts.append(msg_dict)
                except Exception as e:
                    logger.error(f"Error processing message {i}: {str(e)}")
                    raise

            logger.debug(f"Sending {len(message_dicts)} messages to API")
            client = cast(OpenAI, self._client)
            response = client.chat.completions.create(
                model=self.model,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug("Received response from Deepseek API: %s", content)
            return AIMessage(content=content)
        except Exception as e:
            logger.error("Error calling Deepseek API: %s\nTraceback:\n%s", 
                str(e), traceback.format_exc())
            raise

    def with_structured_output(self, output_schema, **kwargs):
        def clean_json_response(content: str) -> str:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()

        def create_error_result(memory: str, next_goal: str = "Please try again") -> Dict:
            error_state = AgentBrain(
                evaluation_previous_goal="",
                memory=memory,
                next_goal=next_goal
            )
            error_result = output_schema(current_state=error_state, action=[])
            return {"parsed": error_result, "raw": memory}

        async def _process_messages(messages: List[BaseMessage]) -> dict:
            try:
                try:
                    schema_str = str(output_schema.model_json_schema())
                except AttributeError:
                    schema_str = str(output_schema.schema())
                
                from langchain_core.output_parsers import JsonOutputParser
                parser = JsonOutputParser(pydantic_object=output_schema)
                format_instructions = parser.get_format_instructions()
                
                message_dicts = [
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=f"You are a helpful assistant that generates JSON output according to this schema: {schema_str}\n\nFormat instructions: {format_instructions}\n\nIMPORTANT: Return ONLY the JSON without any markdown formatting or code blocks."
                    )
                ]
                
                for msg in messages:
                    message_dicts.append(self._convert_message_to_dict(msg))
                
                try:
                    client = cast(OpenAI, self._client)
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=message_dicts,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    if content is None:
                        return create_error_result("Error: Empty response from API")
                    
                    try:
                        cleaned_content = clean_json_response(content)
                        logger.debug(f"Cleaned content for parsing: {cleaned_content}")
                        
                        parsed_dict = json.loads(cleaned_content)
                        if not isinstance(parsed_dict, dict):
                            return create_error_result("Error: Response must be a JSON object")
                            
                        if 'current_state' not in parsed_dict:
                            parsed_dict['current_state'] = AgentBrain(
                                evaluation_previous_goal="",
                                memory="",
                                next_goal=""
                            ).model_dump()
                        if 'action' not in parsed_dict:
                            parsed_dict['action'] = []
                            
                        result = output_schema(**parsed_dict)
                        return {"parsed": result, "raw": content}
                        
                    except json.JSONDecodeError as e:
                        return create_error_result(f"Error: Failed to parse JSON response: {str(e)}")
                    except Exception as e:
                        return create_error_result(f"Error: {str(e)}")
                    
                except APIStatusError as e:
                    if e.status_code == 402:
                        return create_error_result(
                            "Error: Deepseek API account has insufficient balance",
                            "Please check your Deepseek API account balance"
                        )
                    raise
                    
            except Exception as e:
                logger.error(f"Error in structured output: {str(e)}\nTraceback:\n%s", 
                    traceback.format_exc())
                return create_error_result(f"Error: {str(e)}")

        return RunnableLambda(_process_messages) 