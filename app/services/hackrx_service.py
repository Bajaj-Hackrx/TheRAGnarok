import logging
import hashlib
from typing import List, Dict
import asyncio

from .chromadb_service import chroma_service
from .ingestion_service import ingestion_service
from .llm_service import llm_service
from ..models.schemas import HackRXRequest, HackRXResponse

logger = logging.getLogger(__name__)

# In-memory cache for storing (doc_id, question) -> answer
qa_cache: Dict[str, str] = {}

class HackRxService:
    def _create_document_id(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def _get_answer_for_question(self, doc_id: str, question: str) -> str:
        """
        Performs a dedicated search for a single question and generates an answer.
        This function is now designed to be run concurrently for multiple questions.
        """
        # 1. Check the Q&A cache first
        cache_key = f"{doc_id}::{question}"
        if cache_key in qa_cache:
            logger.info(f"Q&A Cache HIT for question: '{question[:50]}...'")
            return qa_cache[cache_key]

        logger.info(f"Q&A Cache MISS. Searching context for: '{question[:50]}...'")
        
        # 2. Perform a focused search for the individual question
        context_results = await chroma_service.search_similar(
            query=question,
            n_results=5, # Retrieve 5 relevant chunks for good context
            where_filter={"document_id": doc_id}
        )
        
        context_chunks = [result.content for result in context_results]
        
        if not context_chunks:
            logger.warning("No relevant context found in ChromaDB for the question.")
            return "Could not find relevant information in the provided document to answer the question."
            
        # 3. Use the simple answer generation for a direct response
        answer = await llm_service.generate_simple_answer(question, context_chunks)
        
        # 4. Store the new answer in the cache before returning
        qa_cache[cache_key] = answer
        return answer

    async def process_request(self, request: HackRXRequest) -> HackRXResponse:
        doc_url = request.documents[0]
        questions = request.questions
        doc_id = self._create_document_id(doc_url)
        logger.info(f"Processing request for document URL: {doc_url} (ID: {doc_id[:10]}...)")

        existing_chunks = await chroma_service.search_similar(query=".", n_results=1, where_filter={"document_id": doc_id})

        if not existing_chunks:
            logger.info(f"Cache MISS for document {doc_id[:10]}.... Processing from URL.")
            try:
                chunks, metadatas, chunk_ids = await asyncio.to_thread(
                    ingestion_service.process_document_from_url, doc_url
                )
                
                for meta in metadatas:
                    meta['document_id'] = doc_id
                
                await chroma_service.add_documents(chunks, metadatas, chunk_ids)
            except Exception as e:
                logger.error(f"Failed to process and ingest document from URL {doc_url}: {e}")
                answers = [f"Error: Could not process the document from the provided URL. Details: {e}" for _ in questions]
                return HackRXResponse(answers=answers)
        else:
            logger.info(f"Cache HIT for document {doc_id[:10]}.... Skipping ingestion.")

        # --- NEW CONCURRENT LOGIC ---
        # Create a list of tasks, one for each question.
        tasks = [self._get_answer_for_question(doc_id, q) for q in questions]
        
        # Run all tasks concurrently and wait for all of them to complete.
        final_answers = await asyncio.gather(*tasks)

        return HackRXResponse(answers=final_answers)

hackrx_service = HackRxService()
