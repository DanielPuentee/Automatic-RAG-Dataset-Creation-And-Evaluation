from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import os
import giskard
from giskard.rag import generate_testset, KnowledgeBase  
import re

os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

class RAGDataset:

    def __init__(self, 
                 llm_model: str = "gemini/gemini-1.5-flash", 
                 embedding_model: str = "gemini/embedding-001"):
        giskard.llm.set_llm_model(llm_model)  
        giskard.llm.set_embedding_model(embedding_model)  

    def get_chunk_ids(self, reference_context: str, df_chunks: pd.DataFrame) -> list:
        """
        Get the chunk ids from the reference context

        Args:
            reference_context (str): reference context

        Returns:
            list: list of chunk ids
        """
        pattern = r'Document (\d+):'
        numbers = re.findall(pattern, reference_context)
        numbers = list(map(int, numbers))

        # Get the chunk_id for each number
        chunk_ids = [df_chunks.iloc[number]['chunk_id'] for number in numbers]
        return chunk_ids
    
    def dataset_creation(self, df_chunks: pd.DataFrame, 
                         content_column: list = ['content'],
                         num_questions: int = 5,
                         language: str = "en",
                         agent_description: str = """As a specialist assistant in the creation of narrative datasets, your role is to generate questions that simulate real conversations from people passionate about TV series, cultural journalists, television critics, or audiovisual communication students. The questions should reflect a variety of interests: main and secondary plots, character development, narrative arcs, changes in relationships, ethical or strategic decisions, and the cultural or emotional impact of key events. Vary the level of complexity and style in the phrasing: from simple informative questions to those requiring analysis, comparison between series, or narrative deduction. Include the necessary context to evaluate the ability of the Retrieval-Augmented Generation (RAG) system to find accurate information in the source documents and generate coherent, contextualized, and useful responses for the user."""
                         ) -> pd.DataFrame:
        """
        Create a dataset for RAG

        Args:
            df (pd.DataFrame): Dataframe containing the text to be chunked
            content_column (list): List of columns to be used as content
            num_questions (int): Number of questions to be generated
            language (str): Language of the questions
            agent_description (str): Description of the agent

        Returns:
            pd.DataFrame: Dataframe containing the chunks
        """
        if df_chunks.empty:
            raise ValueError("Dataframe is empty")
        
        if not isinstance(df_chunks, pd.DataFrame):
            try:
                df_chunks = pd.DataFrame(df_chunks)
            except Exception as e:
                raise ValueError(f"Could not transform into dataframe: {e}")

        knowledge_base = KnowledgeBase.from_pandas(df_chunks, columns=content_column)

        testset = generate_testset(  
            knowledge_base,
            num_questions=num_questions, 
            language=language,
            agent_description=agent_description
        )  
        testset_df = testset.to_pandas()

        testset_df.reset_index(inplace=True)
        testset_df.drop(columns=['conversation_history'], inplace=True)
        testset_df.rename(columns={'metadata': 'reference_metadata'}, inplace=True)
        
        testset_df['reference_context_id'], testset_df['reference_context_metadata'] = '', ''
        for k, v in testset_df.iterrows():
            # 1. Reference context id
            reference_context_id = self.get_chunk_ids(v['reference_context'], df_chunks)
            testset_df.at[k, 'reference_context_id'] = reference_context_id

            # 2. Reference context
            reference_context = re.split(r'Document \d+: ', v['reference_context'])
            reference_context = [chunk.strip() for chunk in reference_context if chunk.strip()]
            testset_df.at[k, 'reference_context'] = reference_context

            # 3. Context metadata
            reference_context_metadata = df_chunks[df_chunks['chunk_id'].isin(reference_context_id)]
            reference_context_metadata = reference_context_metadata.to_dict(orient='records')
            testset_df.at[k, 'reference_context_metadata'] = reference_context_metadata
        
        testset_df = testset_df[['id', 'question', 'reference_answer', 'reference_context', 'reference_context_id', 'reference_metadata', 'reference_context_metadata']]
        
        return testset_df
    