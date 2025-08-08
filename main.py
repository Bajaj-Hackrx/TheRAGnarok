import logging
import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env file from the current directory (project root)
load_dotenv()

# --- CORRECTED IMPORTS FROM THE 'app' PACKAGE ---
from app.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    HackRXRequest,
    HackRXResponse,
    ErrorResponse,
    PolicyQuestionsRequest,
    PolicyQuestionsResponse
)
from app.services import chroma_service, ingestion_service, hackrx_service, llm_service
from app.agents import coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Key Verification Dependency ---
API_KEY = os.getenv("API_KEY")

async def verify_api_key(authorization: str = Header(None)):
    """A dependency that checks the Authorization header for a valid bearer token."""
    if not API_KEY:
        logger.warning("API_KEY not set in .env file. Endpoint is unsecured.")
        return

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing"
        )
    
    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic RAG System...")
    yield
    logger.info("Shutting down Agentic RAG System...")

app = FastAPI(
    title="Agentic RAG System & Hackathon Endpoint",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hackathon Specific Endpoint ---
@app.post(
    "/hackrx/run", 
    response_model=HackRXResponse, 
    tags=["Hackathon"],
    dependencies=[Depends(verify_api_key)]
)
async def run_hackathon_submission(request: HackRXRequest):
    """The official endpoint for the HackRx 6.0 challenge."""
    logger.info(f"Received /hackrx/run request for document: {request.documents[0]}")
    try:
        response = await hackrx_service.process_request(request)
        return response
    except Exception as e:
        logger.error(f"Critical error in /hackrx/run endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")

# --- General Purpose Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy"}

@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    file_content = await file.read()
    chunks, metadatas, chunk_ids = ingestion_service.process_uploaded_file(file_content, file.filename)
    await chroma_service.add_documents(chunks, metadatas, chunk_ids)
    return UploadResponse(
        message=f"Document '{file.filename}' processed.",
        document_id=metadatas[0].get("document_id", "unknown") if metadatas else "unknown",
        chunks_processed=len(chunks)
    )

@app.post("/query/", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    return await coordinator.process_query(query=request.question)

# --- Policy Questions Endpoint (from merge) ---
@app.post("/policy-questions/", response_model=PolicyQuestionsResponse, tags=["Insurance Policy"])
async def answer_policy_questions(request: PolicyQuestionsRequest) -> PolicyQuestionsResponse:
    """Answer specific National Parivar Mediclaim Plus Policy questions."""
    try:
        start_time = time.time()
        logger.info(f"Processing {len(request.questions)} policy questions...")
        
        context_chunks = []
        if request.use_context:
            try:
                search_results = await chroma_service.search_similar(
                    query="National Parivar Mediclaim Plus Policy insurance coverage benefits",
                    n_results=5
                )
                context_chunks = [result.content for result in search_results]
                logger.info(f"Retrieved {len(context_chunks)} context chunks from database")
            except Exception as e:
                logger.warning(f"Failed to retrieve context from database: {e}")
        
        # This assumes an 'answer_policy_questions' method exists in your llm_service
        # You may need to add this method if it doesn't exist.
        result = await llm_service.answer_policy_questions(
            questions=request.questions,
            context_chunks=context_chunks if context_chunks else None
        )
        
        processing_time = time.time() - start_time
        average_confidence = sum(result["confidence_scores"]) / len(result["confidence_scores"]) if result["confidence_scores"] else 0.0
        
        logger.info(f"Policy questions processed in {processing_time:.2f}s with avg confidence: {average_confidence:.2f}")
        
        return PolicyQuestionsResponse(
            answers=result["answers"],
            confidence_scores=result["confidence_scores"],
            inferences=result["inferences"],
            average_confidence=average_confidence,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Policy questions processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Policy questions processing failed: {str(e)}")

@app.get("/status/", tags=["System"])
async def get_system_status():
    return {"status": "operational"}

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
