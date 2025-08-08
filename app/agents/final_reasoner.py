import logging
from typing import Dict, Any, List
from langchain_core.tools import tool

from ..services.llm_service import llm_service
from ..models.schemas import DecisionType

logger = logging.getLogger(__name__)

@tool
async def final_reasoning_tool(query: str, entities: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates a final structured decision based on context."""
    try:
        if not search_results:
            return {
                "status": "success",
                "decision": {
                    "decision": DecisionType.INSUFFICIENT_INFO.value,
                    "confidence": 0.1,
                    "reasoning": "No relevant documents found to make a decision.",
                }
            }
        
        context_chunks = [result["content"] for result in search_results]
        
        decision = await llm_service.generate_final_decision(
            query=query,
            entities=entities,
            context_chunks=context_chunks,
            search_results=search_results
        )
        
        return {"status": "success", "decision": decision}
    except Exception as e:
        logger.error(f"Final reasoning failed: {str(e)}")
        return {"status": "error", "error": str(e)}

class FinalReasonerAgent:
    """Agent responsible for generating final structured decisions"""
    
    def __init__(self):
        self.tool = final_reasoning_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final reasoning"""
        query = state.get("query", "")
        entities = state.get("entities", {})
        search_results = state.get("search_results", [])
        
        result = await self.tool.ainvoke({
            "query": query,
            "entities": entities,
            "search_results": search_results
        })
        
        updated_state = state.copy()
        updated_state["final_decision"] = result.get("decision", {})
        updated_state["reasoning_status"] = result.get("status", "error")
        
        if result.get("status") != "success":
            errors = updated_state.get("errors", [])
            errors.append(f"Final reasoning failed: {result.get('error', 'Unknown error')}")
            updated_state["errors"] = errors
        
        return updated_state
