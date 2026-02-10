import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RagasEvaluator:
    def __init__(self):
        # We explicitly pass the LLM and Embeddings to Ragas 
        # to ensure Answer Relevancy can generate synthetic questions.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # Core metrics
        self.metrics = [faithfulness, answer_relevancy]

    def run_evaluation(self, question: str, answer: str, contexts: list):
        """
        Runs RAGAS evaluation on a single Q&A pair.
        contexts: List of strings (page_content from retrieved docs)
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts], 
        }
        
        dataset = Dataset.from_dict(data)
        
        # Execute evaluation with explicit model passing
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return result.to_pandas()