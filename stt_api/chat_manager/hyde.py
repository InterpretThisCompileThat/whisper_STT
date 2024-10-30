'''Prompt creator for LLM Inference'''
# pylint: disable=W1202, C0209, R0903
import logging
import threading

from .groq_inference import InferenceClient

class HyDEDocumentGenerator:
    '''HyDE'''
    def __init__(self,
                 client: InferenceClient,
                 k: int = 5,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 1.17,
                 temperature: float = 0.7,
                 **kwargs
                 ):
        self._client = client
        self._k = k
        self.kwargs = kwargs
        self.kwargs['max_new_tokens'] = max_new_tokens
        self.kwargs['repetition_penalty'] = repetition_penalty
        self.kwargs['temperature'] = temperature
        self.kwargs['stream'] = False

    @staticmethod
    def _worker(query: str, client: InferenceClient, results: list[str], **kwargs):
        '''Worker'''
        response = client.text_generation(
            prompt=query,
            **kwargs)
        results.append(response)

    def __call__(self, query: str) -> list[str]:
        '''Call'''
        threads = []
        results = []
        for _ in range(self._k):
            thread = threading.Thread(target=self._worker,
                                      args=(query, self._client, results),
                                      kwargs=self.kwargs)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        logging.info("Received hyde documents {} for query {}".format(results, query))
        return results
