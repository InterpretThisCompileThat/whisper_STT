'''Prompt creator for LLM Inference'''
# pylint: disable=W1202, C0209
import logging

from .groq_prompt_builder import PromptBuilder
from .hyde import HyDEDocumentGenerator
from .groq_inference import InferenceClient
from .ollama_inference import InferenceClient

class ChatManager:
    """Class for managing chats"""

    def __init__(self,
                 user_id: str,
                 session_id: str,
                 chat_history: list,
                 prompt_builder: PromptBuilder
                 ):
        '''Initialise the ChatManager Object'''
        self._user_id = user_id
        self._session_id = session_id
        self._chat_history = chat_history
        self._prompt_builder = prompt_builder

    def format(self, query: str, system_prompt: str = ''):
        '''Modify the prompt'''
        logging.debug("Received query {} sys prompt {}".format(query, system_prompt))
        query = self._prompt_builder.format(query=query,
                                            history=self._chat_history,
                                            system_prompt=system_prompt)
        logging.info("Prepared query {}".format(query))
        return query

    def create_rag_query(self,
                         prompt: str,
                         inference_client: InferenceClient):
        '''Create the query for LLM with RAG'''
        if not self._chat_history:
            response = prompt
        else:
            response = inference_client.text_generation(
                prompt=self.format(
                    "Based on this conversation, add SHORT AND CONCISE "
                    "context in KOREAN to the question '{}'. Return ONLY "
                    "the question with additional context in the response.".format(prompt),
                    system_prompt="This system analyzes conversations to "
                                  "summarize intent of a query."),
                temperature=0.1,
                repetition_penalty=1.17,
                max_new_tokens=64,
                stream=False,
                stop_sequences=self._prompt_builder.STOP_SEQUENCES)
            for token in self._prompt_builder.SKIP_SEQUENCES:
                if token in response:
                    response = response[:response.index(token)]
        logging.info('Prepared RAG query {}'.format(response))
        return response

    def get_hyde_documents(self,
                           prompt: str,
                           inference_client: InferenceClient
                           ) -> list[str]:
        '''Return HyDE documents'''
        hyde_document_generator = HyDEDocumentGenerator(
            client=inference_client,
            stop_sequences=self._prompt_builder.STOP_SEQUENCES,
        )
        return [prompt] + hyde_document_generator(
            query=self.format(
                'Answer the below question in Korean:\n{}'.format(prompt),
                system_prompt='You are an AI assistant that generates answers to questions.'
            )
        )
