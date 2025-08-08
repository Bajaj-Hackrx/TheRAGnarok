import logging
import sys
import os
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
    ErrorResponse
)
from app.services import chroma_service, ingestion_service, hackrx_service
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

@app.delete("/documents/{document_url:path}", tags=["Cache Management"])
async def delete_document_cache(document_url: str):
    """Delete cached document chunks for a specific URL."""
    try:
        doc_id = hackrx_service._create_document_id(document_url)
        deleted_count = await chroma_service.delete_by_filter({"document_id": doc_id})
        return {
            "message": "Successfully deleted cached chunks for document",
            "document_id": doc_id,
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to delete document cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete cache: {str(e)}")

@app.delete("/documents/clear-all", tags=["Cache Management"])
async def clear_all_cache():
    """Clear the entire document cache in ChromaDB."""
    try:
        deleted_count = await chroma_service.clear_collection()
        return {
            "message": "Successfully cleared all cached documents",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/chunks/", tags=["Cache Management"])
async def view_all_chunks(limit: Optional[int] = None):
    """View all chunks stored in ChromaDB."""
    try:
        return await chroma_service.get_all_chunks(limit=limit)
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")

@app.get("/chunks/document/{document_url:path}", tags=["Cache Management"])
async def view_chunks_by_document(document_url: str, limit: Optional[int] = None):
    """View chunks for a specific document URL."""
    try:
        doc_id = hackrx_service._create_document_id(document_url)
        chunks_data = await chroma_service.get_chunks_by_document(doc_id, limit=limit)
        return {
            "document_url": document_url,
            "document_id": doc_id,
            **chunks_data
        }
    except Exception as e:
        logger.error(f"Failed to retrieve chunks for document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks for document: {str(e)}")

@app.get("/chunks/stats", tags=["Cache Management"])
async def get_chunks_statistics():
    """Get statistics about the chunks in ChromaDB."""
    try:
        total_count = chroma_service.get_collection_count()
        sample_chunks = await chroma_service.get_all_chunks()
        document_ids = set(
            chunk["metadata"]["document_id"]
            for chunk in sample_chunks.get("chunks", [])
            if "document_id" in chunk.get("metadata", {})
        )
        return {
            "total_chunks": total_count,
            "unique_documents_in_sample": len(document_ids),
            "collection_name": chroma_service.collection.name if chroma_service.collection else "N/A",
        }
    except Exception as e:
        logger.error(f"Failed to get chunks statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
