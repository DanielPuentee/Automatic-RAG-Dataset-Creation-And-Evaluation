from langchain_google_genai import ChatGoogleGenerativeAI

class Answering:

    def __init__(self):
        
        self.system_prompt = """You are a film and TV series critic, specializing in detailed analysis of plots, characters, and television productions.
        Using your extensive knowledge and experience, provide detailed and accurate answers to users' questions about movies and series.
        Respond in an impartial, professional, and analytical tone using fewer than 800 words.
        Respond in the language {language}.
        """

        self.user_prompt = """Context:
        {context}
        ===
        Task:
        Answer the question if the information is available in the above context and indicate which sources support it the most (for example, the specific document or referenced season).
        If you cannot answer the question based on the above context, respond with "Answer not found."
        ===
        Question:
        {query}
        ===
        Answer in less than 800 words and in {language}.
        Answer:
        """

    def answering(self, context: str, query: str, llm: ChatGoogleGenerativeAI):
        """
        Answer a question

        Args:
            context (str): context
            query (str): user query
            llm (ChatGoogleGenerativeAI): model to use

        Returns:
            str: answer
        """
        system = self.system_prompt.format(language="Spanish")
        user = self.user_prompt.format(context=context, query=query, language="Spanish")
        messages = [
            ("system", system),
            ("human", user)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    
    def answer_multiple_queries(self, queries: list, contexts: str, llm: ChatGoogleGenerativeAI):
        """
        Answer multiple questions

        Args:
            queries (list): list of queries
            context (str): context to use
            llm (ChatGoogleGenerativeAI): LLM to use

        Returns:
            list: list of answers
        """

        answers = []
        for query, context in zip(queries, contexts):
            answer = self.answering(context=context, query=query, llm=llm)
            answers.append(answer)
        return answers