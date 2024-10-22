from typing import List, Optional
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer
from config import Config


class LLMService(LLM):

    """ 
    模型服务 
    """
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response

    def load_model(self, model_name_or_path: str = "ClueAI/ChatYuan-large-v2"):
        """
        加载大模型LLM
        :return:
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.llm_model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = self.model.eval()

if __name__ == '__main__':
    ##加载模型
    chatLLM = LLMService()
    chatLLM.load_model()