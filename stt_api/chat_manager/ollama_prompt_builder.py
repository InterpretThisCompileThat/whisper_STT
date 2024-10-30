'''Helps create the prompt for LLMs'''
# pylint: disable=W1202, C0209
import typing

from .prompt_builder import PromptBuilder

class OllamaPromptBuilder(PromptBuilder):
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
        return {"role": "assistant", "content": """
        예시:
        [Q]: KB 국민은행의 'KB 스타적금' 상품은 어떤 플랫폼에서 가입할 수 있습니까?
        [A]: KB 스타뱅킹에서 가입할 수 있습니다.
        
        [Q]: KB 스타적금'의 가입 기간은 얼마나 됩니까?
        [A]: 가입 기간은 12개월입니다.
        
        [Q]: 국민은행이 전세자금대출 금리를 인상하는 이유는 무엇인가요?
        [A]: 국민은행은 가계부채 속도 조절 차원에서 전세자금대출 금리를 인상한다고 밝혔습니다.
                
        [Q]: 국민은행의 KB 스타 주택전세자금대출 금리는 얼마로 인상되었나요?
        [A]: KB 스타 주택전세자금대출 금리는 연 3.99%에서 0.15% 포인트 올라 연 4.14%로 조정되었습니다.
        
        [Q]: 국민은행은 최근 주택담보대출(주담대) 금리를 얼마나 인상했습니까?
        [A]: 국민은행은 주담대 금리를 0.13% 포인트 인상했습니다.
        
        [Q]: 국민은행이 주담대 금리 인상을 시작한 날짜는 언제입니까?
        [A]: 국민은행은 3일부터 주담대 금리를 인상했습니다.
                
        [Q]: KB 소상공인 신용대출을 약정한 이용자에게 제공되는 혜택은 무엇인가요?
        [A]: KB 소상공인 신용대출을 약정한 이용자에게는 6개월간 최대 50% 이자를 환급해줍니다.
        
        [Q]: 국민은행의 가계대출 잔액은 얼마입니까?
        [A]: 719조 9178억원입니다.
        
        [Q]: 이달 들어 국민은행의 가계대출 잔액은 얼마나 증가했습니까?
        [A]: 4조 1795억원 증가했습니다.
        
        [Q]: KB 국민은행이 1주택자의 수도권 주택 추가구입목적 주담대 취급을 언제부터 제한합니까?
        [A]: KB 국민은행은 9월 9일부터 1주택자의 수도권 주택 추가구입목적 주담대 취급을 제한합니다.
        
        결과:
        """}

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
               limit: int = 1024 * 8) -> list[dict[str, str]]:
        '''Convert the ollama prompt to a prompt list'''
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
