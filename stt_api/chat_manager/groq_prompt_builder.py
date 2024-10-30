'''Helps create the prompt for LLMs'''
# pylint: disable=W1202, C0209
import typing

from .prompt_builder import PromptBuilder

class GroqPromptBuilder(PromptBuilder):
    '''Class for building groq prompts'''

    @staticmethod
    def system_prompt(prompt: str) -> dict:
        '''Generate a system prompt'''
        return {"role": "system", "content": prompt}

    @staticmethod
    def user_prompt(prompt: str) -> dict:
        '''Generate a user prompt'''
        return {"role": "user", "content": prompt}

    @staticmethod
    def assistant_prompt(prompt: str) -> dict:
        '''Generate an assistant prompt'''
        return {"role": "assistant", "content": prompt}

    STOP_SEQUENCES = ['<|eot_id|>']
    SKIP_SEQUENCES = ['</s>', '<|eot_id|>']

    def _counter(self, text_list: list[dict[str, str]]):
        '''Count number of texts'''
        counter = 0
        for text_dict in text_list:
            counter += len(text_dict['content'])
        return counter

    def format(self,
               query: str,
               history: list[dict[str, str]] = None,
               system_prompt: typing.Optional[str] = None,
               limit: int = 1024 * 4) -> list[dict[str, str]]:
        '''Convert the groq prompt to a prompt list'''
        prompt_list: list[dict[str, str]] = []
        for ch in history or []:
            prompt_list.append(self.user_prompt(ch["Prompt"]))
            prompt_list.append(self.assistant_prompt(ch["Answer"]))
        prompt_list.append(self.user_prompt(query))
        if limit > 0:
            counter = [self.system_prompt(system_prompt)]
            while prompt_list and self._counter(counter + prompt_list) > limit:
                prompt_list.pop(0)
        if system_prompt:
            prompt_list.insert(0, self.system_prompt(system_prompt))
        return prompt_list
