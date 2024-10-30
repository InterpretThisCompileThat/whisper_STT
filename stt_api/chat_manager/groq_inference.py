'''Module for manging classes that interface with LLM'''

from groq import Groq
from groq.resources.chat.completions import Stream, ChatCompletionChunk
from huggingface_hub import InferenceClient as _InferenceClient
from .inference import InferenceClient

class GroqInferenceClient(InferenceClient):
    '''Groq inference'''

    def __init__(self, model: str = 'llama3-70b-8192'):
        self.model = model
        self.client = Groq()

    def text_generation(self,
                        prompt: list[dict[str, str]],
                        temperature: float,
                        repetition_penalty: float,
                        max_new_tokens: int,
                        stop_sequences: list[str],
                        stream: bool,
                        **kwargs
                        ):
        '''Text generation on groq'''
        response = self.client.chat.completions.create(
            messages=prompt,
            temperature=temperature,
            frequency_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            model=self.model,
            stream=stream,
        )
        return response.choices[0].message.content

    def _yield_response(self, response: Stream[ChatCompletionChunk]):
        '''deal with generation on groq'''
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


__all__ = [
    'GroqInferenceClient',
    'InferenceClient'
]
