from ragas.metrics import faithfulness, answer_relevancy, answer_similarity, context_recall, context_precision  
from ragas import evaluate, RunConfig  
from ragas.metrics import (  
    faithfulness,  
    answer_relevancy,  
    context_recall,  
    context_precision,  
)  
from datasets import Dataset
import ast
import pandas as pd

class Evaluation:

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

        self.run_config = RunConfig(  
                timeout = 10000,
                max_retries = 5, 
                max_wait = 60, 
                max_workers = 8, 
                exception_types = (Exception),
                log_tenacity = True,
                seed = 40
            )  
        self.metrics = {
            'context_precision': context_precision,
            'context_recall': context_recall,  
            'answer_relevancy': answer_relevancy,  
            'answer_similarity': answer_similarity,  
            'faithfulness': faithfulness
        } 

    def evaluate(self, testset_df: pd.DataFrame, retrieval_metrics: list = [], answer_metrics: list = []):
        """
        Evaluate the testset with the given metrics

        Args:
            testset_df (pd.DataFrame): Test Dataframe
            retrieval_metrics (list): List of retrieval metrics to be used. Possible values are:
                - context_precision
                - context_recall
            answer_metrics (list): List of answer metrics to be used
                - answer_relevancy
                - answer_similarity
                - faithfulness
        """
        reference_context = testset_df['reference_context']
        if isinstance(reference_context[0], str):
            reference_context = testset_df['reference_context'].\
                apply(lambda x: ast.literal_eval(x))
            
        data = {  
            "user_input": testset_df['question'].tolist(),
            "reference_contexts": reference_context.tolist(),  
            "reference": testset_df['reference_answer'].tolist(),
        }  

        if retrieval_metrics or answer_metrics:

            retrieved_contexts = testset_df['generated_context']
            if isinstance(retrieved_contexts[0], str):
                retrieved_contexts = testset_df['generated_context'].\
                    apply(lambda x: ast.literal_eval(x))
            data['retrieved_contexts'] =retrieved_contexts.tolist()

        if answer_metrics:
            data['response'] = testset_df['generated_answer'].tolist()
  
        dataset = Dataset.from_dict(data)  

        all_metrics = retrieval_metrics + answer_metrics
        evaluate_metrics = [v for k, v in self.metrics.items() if k in all_metrics]

        results = evaluate(  
            dataset=dataset,
            metrics=evaluate_metrics,
            llm=self.llm, 
            embeddings=self.embeddings,
            run_config=self.run_config 
        ).to_pandas() 

        results = results.rename(columns={
            'user_input': 'question',
            'retrieved_contexts': 'generated_context',
            'reference': 'reference_answer',
            'response': 'generated_answer'
        })

        return results