import logging
import time
from typing import Dict, Any, List, TypedDict, Literal
from langgraph.graph import StateGraph, END

from ..models.schemas import QueryResponse, EntityExtraction, SearchResult, DecisionType
from .entity_extractor import EntityExtractorAgent
from .vector_searcher import VectorSearchAgent
from .final_reasoner import FinalReasonerAgent

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State shared between agents"""
    query: str
    max_results: int
    entities: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    final_decision: Dict[str, Any]
    errors: List[str]
    start_time: float
    current_step: str
    entity_extraction_status: str
    search_status: str
    reasoning_status: str
    search_metadata: Dict[str, Any]
    reasoning_metadata: Dict[str, Any]

class AgentCoordinator:
    """Coordinates the multi-agent workflow using LangGraph"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractorAgent()
        self.vector_searcher = VectorSearchAgent()
        self.final_reasoner = FinalReasonerAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        async def extract_entities_step(state: AgentState) -> AgentState:
            logger.info("Starting entity extraction step")
            state["current_step"] = "entity_extraction"
            return await self.entity_extractor.run(state)
        
        async def vector_search_step(state: AgentState) -> AgentState:
            logger.info("Starting vector search step")
            state["current_step"] = "vector_search"
            return await self.vector_searcher.run(state)
        
        async def final_reasoning_step(state: AgentState) -> AgentState:
            logger.info("Starting final reasoning step")
            state["current_step"] = "final_reasoning"
            return await self.final_reasoner.run(state)
        
        def should_continue_after_entities(state: AgentState) -> Literal["vector_search", "end"]:
            return "vector_search" if state.get("entity_extraction_status") == "success" else "end"
        
        def should_continue_after_search(state: AgentState) -> Literal["final_reasoning", "end"]:
            return "final_reasoning"

        workflow = StateGraph(AgentState)
        
        workflow.add_node("extract_entities", extract_entities_step)
        workflow.add_node("vector_search", vector_search_step)
        workflow.add_node("final_reasoning", final_reasoning_step)
        
        workflow.set_entry_point("extract_entities")
        workflow.add_conditional_edges("extract_entities", should_continue_after_entities, {"vector_search": "vector_search", "end": END})
        workflow.add_conditional_edges("vector_search", should_continue_after_search, {"final_reasoning": "final_reasoning", "end": END})
        workflow.add_edge("final_reasoning", END)
        
        return workflow.compile()
    
    async def process_query(self, query: str, max_results: int = 5) -> QueryResponse:
        """Process a query through the complete agent workflow"""
        start_time = time.time()
        try:
            initial_state: AgentState = {
                "query": query, "max_results": max_results, "entities": {},
                "search_results": [], "final_decision": {}, "errors": [],
                "start_time": start_time, "current_step": "initialization",
                "entity_extraction_status": "pending", "search_status": "pending", 
                "reasoning_status": "pending", "search_metadata": {}, "reasoning_metadata": {}
            }
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            processing_time = time.time() - start_time
            
            entities = final_state.get("entities", {})
            search_results = final_state.get("search_results", [])
            decision_data = final_state.get("final_decision", {})
            
            entity_extraction = EntityExtraction(**entities)
            search_result_objects = [SearchResult(**result) for result in search_results]
            
            decision_type = decision_data.get("decision", DecisionType.INSUFFICIENT_INFO.value)
            try:
                decision_enum = DecisionType(decision_type)
            except ValueError:
                decision_enum = DecisionType.INSUFFICIENT_INFO
            
            return QueryResponse(
                decision=decision_enum,
                confidence=decision_data.get("confidence", 0.0),
                reasoning=decision_data.get("reasoning", "No reasoning provided."),
                entities_found=entity_extraction,
                search_results=search_result_objects,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return QueryResponse(
                decision=DecisionType.INSUFFICIENT_INFO,
                confidence=0.0,
                reasoning=f"System error: {str(e)}",
                entities_found=EntityExtraction(),
                search_results=[],
                processing_time=time.time() - start_time
            )

coordinator = AgentCoordinator()
