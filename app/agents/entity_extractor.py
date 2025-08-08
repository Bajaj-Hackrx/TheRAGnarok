import logging
from typing import Dict, Any
from langchain_core.tools import tool

from ..services.llm_service import llm_service

logger = logging.getLogger(__name__)

@tool
async def extract_entities_tool(query: str) -> Dict[str, Any]:
    """Extracts entities from a user query."""
    try:
        entities = await llm_service.extract_entities(query)
        return {"status": "success", "entities": entities}
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        return {"status": "error", "error": str(e), "entities": {}}

class EntityExtractorAgent:
    """Agent responsible for extracting entities from user queries"""
    
    def __init__(self):
        self.tool = extract_entities_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity extraction"""
        query = state.get("query", "")
        result = await self.tool.ainvoke({"query": query})
        
        updated_state = state.copy()
        updated_state["entities"] = result.get("entities", {})
        updated_state["entity_extraction_status"] = result.get("status", "error")
        
        if result.get("status") != "success":
            errors = updated_state.get("errors", [])
            errors.append(f"Entity extraction failed: {result.get('error', 'Unknown error')}")
            updated_state["errors"] = errors
        
        return updated_state
