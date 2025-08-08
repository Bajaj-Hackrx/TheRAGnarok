import logging
import os
from typing import Dict, Any, List
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from ..models.config import settings
from ..models.schemas import DecisionType

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenRouter LLM client"""
        try:
            self.client = ChatOpenAI(
                model=settings.default_model,
                openai_api_key=settings.openrouter_api_key,
                openai_api_base=settings.openrouter_base_url,
                temperature=settings.temperature,
                timeout=120,
                max_retries=2
            )
            logger.info("LLM client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None
            
    async def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extracts entities from a user query using the LLM."""
        if not self.client:
            raise ConnectionError("LLM client is not initialized.")
        
        system_prompt = """You are an expert entity extractor for insurance documents.
Extract relevant entities from the user's query. Focus on:
- Age
- Medical procedures or treatments
- Time durations (e.g., '36 months')
- Medical conditions (e.g., 'pre-existing diseases')
Return your response as a single JSON object. Use null for missing information."""
        
        human_prompt = f"Query: {query}\n\nJSON Output:"
        
        response = await asyncio.to_thread(
            self.client.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction response as JSON.")
            return {}

    async def generate_final_decision(self, query: str, entities: Dict[str, Any], context_chunks: List[str], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a detailed, structured decision for the agentic workflow."""
        if not self.client:
            raise ConnectionError("LLM client is not initialized.")

        context = "\n\n".join(context_chunks)
        system_prompt = """You are an expert AI assistant for insurance document analysis.
Analyze the user's query against the provided context and generate a structured decision in a valid JSON format.
Decision can be 'approved', 'rejected', 'requires_review', or 'insufficient_info'.
Confidence must be a float between 0.0 and 1.0.
Base your decision strictly on the provided context."""
        
        human_prompt = f"Query: {query}\n\nEntities: {entities}\n\nContext:\n{context}\n\nDecision (JSON):"
        
        response = await asyncio.to_thread(
            self.client.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse final decision response as JSON.")
            return {"decision": "insufficient_info", "confidence": 0.0, "reasoning": "Failed to parse LLM response."}

    async def generate_simple_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generates a direct, concise answer for the hackathon workflow."""
        if not self.client:
            error_msg = "LLM client is not initialized. Please check your .env configuration and server logs."
            logger.error(error_msg)
            return error_msg
        try:
            context = "\n\n".join(f"--- Context Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(context_chunks))
            
            system_prompt = """You are a meticulous and precise AI assistant specializing in analyzing policy documents. Your task is to answer the user's question with extreme accuracy, based *only* on the provided document context.

Follow these rules strictly:
1.  **Prioritize Direct Information:** If the answer is stated directly in the context, use the exact wording from the document as much as possible.
2.  **Synthesize if Necessary:** If the answer is spread across multiple context chunks, synthesize them into a single, coherent answer.
3.  **Be Direct:** Provide only the direct answer to the question. Do not add conversational fillers like "Based on the provided context..." or "The document states that...".
4.  **Handle Missing Information:** Before concluding that information is missing, carefully re-read all provided context chunks. If the answer is truly not present, and only then, state: "The information required to answer this question is not available in the provided document excerpts." Do not guess or infer.
"""
            
            human_prompt = f"""DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nPRECISE ANSWER:"""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate simple answer: {str(e)}")
            return f"An error occurred while generating the answer: {e}"

    async def generate_batch_answers(self, questions: List[str], context_chunks: List[str]) -> List[str]:
        """Generates answers for a list of questions in a single, optimized LLM call."""
        if not self.client:
            error_msg = "LLM client is not initialized."
            logger.error(error_msg)
            return [error_msg] * len(questions)
        try:
            context = "\n\n".join(context_chunks)
            
            system_prompt = """You are an expert AI assistant. Your task is to answer a list of questions based *only* on the provided document context.
Provide a concise, direct answer for each question.
If the context does not contain the answer for a specific question, state that the information is not available in the document for that question.
Format your response as a numbered list, where each number corresponds to the question number. For example:
1. [Answer to question 1]
2. [Answer to question 2]"""
            
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            human_prompt = f"""CONTEXT:\n{context}\n\nQUESTIONS:\n{questions_text}\n\nANSWERS:"""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            
            response_text = response.content.strip()
            lines = response_text.split('\n')
            answers = []
            for line in lines:
                line = line.strip()
                if line and line.split('.', 1)[0].isdigit():
                    answer_text = line.split('.', 1)[1].strip()
                    answers.append(answer_text)
            
            if len(answers) != len(questions):
                logger.warning(f"LLM returned {len(answers)} answers for {len(questions)} questions. Padding with error message.")
                while len(answers) < len(questions):
                    answers.append("Failed to parse answer from LLM response.")
            
            return answers[:len(questions)]
        except Exception as e:
            logger.error(f"Failed to generate batch answers: {str(e)}")
            return [f"An error occurred while generating the answer: {e}"] * len(questions)

llm_service = LLMService()
