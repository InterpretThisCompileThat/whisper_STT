'''Module for manging classes that interface with LLM'''
import logging
import os
from flask import stream_template_string
from openai import OpenAI
from .inference import InferenceClient

ollama = OpenAI(
    # base_url= f"http://{os.environ['GCP_SERVER_IP_ADDRESS']}:11434/v1",
    base_url= f"http://172.30.1.28:11434/v1",
    
    # base_url= os.environ['OLLAMA_SERVER_IP_ADDRESS'],
    api_key='ollama'
)

class OllamaInferenceClient(InferenceClient):
    '''ollama aya 4-bit'''

    def __init__(self, model: str = 'aya:35b'):
        self.model = model
        self.client = ollama

    def text_generation(self,
                        prompt: list[dict[str, str]],
                        temperature: float,
                        repetition_penalty: float,
                        max_new_tokens: int,
                        stop_sequences: list[str],
                        stream: bool,
                        **kwargs
                        ):
        '''Text generation on ollama'''
        response = self.client.chat.completions.create(
            messages=prompt,
            temperature=temperature,
            frequency_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            model=self.model,
            stream=stream,
        )
        logging.info(response.choices[0].message.content)
        return response.choices[0].message.content

    def _yield_response(self, response: stream_template_string):
        '''deal with generation on aya'''
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

__all__ = [
    'OllamaInferenceClient',
    'InferenceClient'
]
