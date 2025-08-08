import logging
from typing import Dict, Any, List
from langchain_core.tools import tool

from ..services.chromadb_service import chroma_service

logger = logging.getLogger(__name__)

@tool
async def vector_search_tool(query: str, entities: Dict[str, Any], max_results: int = 5) -> Dict[str, Any]:
    """Performs a vector search in ChromaDB."""
    try:
        search_results = await chroma_service.search_similar(query=query, n_results=max_results)
        
        results_data = [result.dict() for result in search_results]
        
        return {"status": "success", "results": results_data}
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return {"status": "error", "error": str(e), "results": []}

class VectorSearchAgent:
    """Agent responsible for vector search and retrieval"""
    
    def __init__(self):
        self.tool = vector_search_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector search"""
        query = state.get("query", "")
        entities = state.get("entities", {})
        max_results = state.get("max_results", 5)
        
        result = await self.tool.ainvoke({
            "query": query,
            "entities": entities,
            "max_results": max_results
        })
        
        updated_state = state.copy()
        updated_state["search_results"] = result.get("results", [])
        updated_state["search_status"] = result.get("status", "error")
        
        if result.get("status") != "success":
            errors = updated_state.get("errors", [])
            errors.append(f"Vector search failed: {result.get('error', 'Unknown error')}")
            updated_state["errors"] = errors
        
        return updated_state
