'''Helps create the prompt for LLMs'''
# pylint: disable=W1202, C0209
import typing

class PromptBuilder:
    '''Class to help build prompts'''
    STOP_SEQUENCES = None
    SKIP_SEQUENCES = ['</s>']

    def format(self,
               query: str,
               history: dict[str, str] = None,
               system_prompt: typing.Optional[str] = None,
               limit: int = -1) -> str:
        '''Guide for classes to implement in subclass'''
        raise NotImplementedError("Implement in subclass")
