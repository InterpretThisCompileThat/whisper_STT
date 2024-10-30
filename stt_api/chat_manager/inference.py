'''Module for manging classes that interface with LLM'''

class InferenceClient:
    '''Base class for interacting with LLM'''

    def text_generation(self,
                        prompt: str | list[dict[str, str]],
                        temperature: float,
                        repetition_penalty: float,
                        max_new_tokens: int,
                        stop_sequences: list[str],
                        stream: bool,
                        **kwargs
                        ):
        '''Method showing which args are required.'''
        raise NotImplementedError("Raise in subclass")
