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

import torch
import logging

import sys
sys.path.append('../text-generation-webui/')
from modules import shared as webui_shared


logger = logging.getLogger(__name__)

class QwenProxyLLMChain(BaseAnswer, Chain, ABC):
    checkPoint: LoaderCheckPoint = None
    max_token: int = 2048
    temperature: float = 0.7
    top_p = 0.8
    history_len: int = 10
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
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
        return "QwenProxyLLMChain"

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
        #print(inputs)

        if webui_shared.tokenizer == None or webui_shared.model == None:
            #print(webui_shared.model, webui_shared.tokenizer)
            prompt = inputs[self.prompt_key]
            answer_result = AnswerResult()
            answer = "Model not initialized."
            answer_result.history += [[prompt, answer]]
            answer_result.llm_output = {"answer": answer}
            generate_with_callback(answer_result)
            return

        history = inputs[self.history_key] or []
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]
        print(f"Qwen proxy __call:{prompt}")
        if len(history) > 0:
            history = history[-self.history_len:] if self.history_len > 0 else []
            prompt_w_history = str(history)
            prompt_w_history += '<|im_start|>user\n' + prompt + '<|im_end|><|im_start|>assistant\n'
        else:
            prompt_w_history = '<|im_start|>user\n' + prompt + '<|im_end|><|im_start|>assistant\n'

        inputs = webui_shared.tokenizer(prompt_w_history, return_tensors="pt")
        with torch.no_grad():
            outputs = webui_shared.model.generate(
                inputs.input_ids.cuda(),
                attention_mask=inputs.attention_mask.cuda(),
                max_length=self.max_token,
                do_sample=True,
                top_k=40,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=1.02,
                num_return_sequences=1,
                eos_token_id=151645, #<|im_end|>
                pad_token_id=webui_shared.tokenizer.pad_token_id)
            response = webui_shared.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                                        skip_special_tokens=True)
            #self.checkPoint.clear_torch_cache()
            #print(prompt)
            print("=====")
            print(response)
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}

            generate_with_callback(answer_result)
