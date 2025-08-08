import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings

from ..models.config import settings
from ..models.schemas import SearchResult

logger = logging.getLogger(__name__)

class ChromaDBService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize ChromaDB client and collection with a local embedding model."""
        try:
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            
            # --- THIS IS THE CRITICAL CHANGE ---
            # Switch back to the reliable local SentenceTransformerEmbeddingFunction
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model_name
            )
            # --- END OF CHANGE ---

            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized with collection: {settings.chroma_collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    async def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Add document chunks to the vector database in batches."""
        try:
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                self.collection.add(
                    documents=chunks[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
            logger.info(f"Successfully added {len(chunks)} document chunks to ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            return False
    
    async def search_similar(
        self, 
        query: str, 
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in the collection."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        chunk_id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        similarity_score=1.0 - results['distances'][0][i], # Convert cosine distance to similarity
                        metadata=results['metadatas'][0][i] or {}
                    ))
            return search_results
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}")
            return []

    async def delete_by_filter(self, where_filter: dict) -> int:
        """Delete documents matching the filter."""
        try:
            results = self.collection.get(where=where_filter)
            count = len(results['ids'])
            if count > 0:
                self.collection.delete(where=where_filter)
            logger.info(f"Deleted {count} chunks matching filter: {where_filter}")
            return count
        except Exception as e:
            logger.error(f"Failed to delete by filter {where_filter}: {e}")
            raise

    async def clear_collection(self) -> int:
        """Clear the entire collection."""
        try:
            count = self.collection.count()
            if count > 0:
                self.client.delete_collection(name=self.collection.name)
                self.collection = self.client.get_or_create_collection(
                    name=settings.chroma_collection_name,
                    embedding_function=self.embedding_function
                )
            logger.info(f"Cleared entire collection: {count} chunks deleted")
            return count
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    async def get_all_chunks(self, limit: Optional[int] = None, where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get all chunks from the collection with optional filtering and limiting."""
        try:
            results = self.collection.get(limit=limit, where=where_filter, include=["documents", "metadatas"])
            chunks_data = [
                {
                    "chunk_id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i] or {}
                }
                for i in range(len(results['ids']))
            ]
            return {"chunks": chunks_data}
        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            raise

    async def get_chunks_by_document(self, document_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get all chunks for a specific document."""
        return await self.get_all_chunks(limit=limit, where_filter={"document_id": document_id})

# Global ChromaDB service instance
chroma_service = ChromaDBService()
