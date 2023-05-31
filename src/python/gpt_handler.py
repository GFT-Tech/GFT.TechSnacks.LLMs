
# imports
import openai  # for calling the OpenAI API
import pandas   # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

class gpt_handler:
    def __init__(self, system_role:str, embeddings: pandas.DataFrame, gpt_model: str, embedding_model:str, gpt_model_to_count_tokens:str="gpt-3.5-turbo"):
        """
        Creates a Chat GPT Handler

        Args:
            pdf_file_path: PDF File to read
            gpt_model: GPT Model to use
            embedding_model: Embedding model
            gpt_model_to_count_tokens: GPT Model to use only for counting tokens(we need to use this because of azure)
        """
        self.gpt_model=gpt_model
        self.embedding_model=embedding_model
        self.gpt_model_to_count_tokens=gpt_model_to_count_tokens
        self.embeddings=embeddings
        self.system_role=system_role

    def ask(
        self,
        query: str,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about Sphinx."},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            engine=self.gpt_model,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message


    def query_message(
        self,
        query: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = self._private_strings_ranked_by_relatedness(query)
        introduction = self.system_role
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\nInformation section:\n```\n{string}\n```'
            if (
                self._private_count_tokens(message + next_article + question)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question

    def _private_count_tokens(self,text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.gpt_model_to_count_tokens)
        return len(encoding.encode(text))

    def _private_strings_ranked_by_relatedness(
        self,
        query: str,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            engine=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.embeddings.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]
