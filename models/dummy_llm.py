from abc import ABC
from langchain.chains.base import Chain
from typing import (
    Any, Dict, List, Optional, Generator,
    )

from models.loader import LoaderCheckPoint
from langchain.callbacks.manager import CallbackManagerForChainRun
from models.base import (BaseAnswer,
                         AnswerResult, 
                         AnswerResultStream)


import logging

logger = logging.getLogger(__name__)

class DummyLLMChain(BaseAnswer, Chain, ABC):
    checkPoint: LoaderCheckPoint = None
    history_len: int = 10
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:

    def __init__(self,
                 checkPoint: LoaderCheckPoint = None,
                 #  api_base_url:str="http://localhost:8000/v1",
                 #  model_name:str="chatglm-6b",
                 #  api_key:str=""
                 ):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _chain_type(self) -> str:
        return "DummyLLMChain"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint
    
    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.prompt_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
            prompt = inputs[self.prompt_key]
            answer_result = AnswerResult()
            answer_result.history += [[prompt, "dummy"]]
            answer_result.llm_output = {"answer": "dummy"}
            generate_with_callback(answer_result)