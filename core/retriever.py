from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
class AdvancedRetriever:
    @staticmethod
    def _deduplicate_docs(docs):
        """Standard industry practice: filter out chunks with identical text."""
        unique_texts = set()
        deduplicated = []
        for doc in docs:
            # We use a hash of the content to quickly identify duplicates
            content_hash = hash(doc.page_content.strip())
            if content_hash not in unique_texts:
                unique_texts.add(content_hash)
                deduplicated.append(doc)
        return deduplicated

    @staticmethod
    def get_ensemble_retriever(documents, embeddings, vectorstore):
        # 1. Deduplicate input documents first to ensure clean indexing
        unique_docs = AdvancedRetriever._deduplicate_docs(documents)
        
        bm25_retriever = BM25Retriever.from_documents(unique_docs)
        bm25_retriever.k = 5
        
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.35, 0.65]
        )
        
        # 2. Flashrank handles semantic deduplication (choosing the best version)
        compressor = FlashrankRerank(top_n=5) 
        
        return ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble
        )