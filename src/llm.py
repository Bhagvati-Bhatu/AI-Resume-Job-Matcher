import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.language_models.llms import LLM
from typing import Optional, List


load_dotenv()


class HuggingFaceLLM(LLM):

    client: object

    def __init__(self, **kwargs):

        client = InferenceClient(
            provider="auto",
            api_key=os.getenv("HF_TOKEN")
        )

        super().__init__(
            client=client,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None
    ) -> str:

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000
        )

        return response.choices[0].message.content