"""LLM 调用封装 - 统一接口"""
from langchain_openai import ChatOpenAI
from config.settings import settings


class LLMClient:
    """LLM 客户端封装"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """单次调用"""
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = self.llm.invoke(messages)
        return resp.content

    def invoke_messages(self, messages: list) -> str:
        """直接传入消息列表"""
        resp = self.llm.invoke(messages)
        return resp.content

    def get_llm(self):
        """返回底层 LangChain LLM 实例，用于绑定工具等场景"""
        return self.llm
