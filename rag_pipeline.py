import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "index"))
sys.path.append(str(Path(__file__).parent / "retrieval"))
sys.path.append(str(Path(__file__).parent / "generation"))

# Local imports
from index.vectorstore import DrugVectorStore
from retrieval.multi_query_retriever import DrugMultiQueryRetriever
from generation.drug_llm import DrugLLM

load_dotenv()


class DrugRAGPipeline:
    """
    Complete Drug RAG Pipeline
    Combines vector search, multi-query retrieval, and LLM generation
    """
    
    def __init__(self, 
                 vector_db_name: str = "drug_vector_db",
                 model_name: str = "gpt-4o-mini",
                 log_level: str = "INFO"):
        """
        Initialize the complete RAG pipeline
        
        Args:
            vector_db_name: Name of the vector database to use
            model_name: OpenAI model name for generation
            log_level: Logging level
        """
        
        self.vector_db_name = vector_db_name
        self.model_name = model_name
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.llm = None
        
        # Pipeline statistics
        self.stats = {
            "queries_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0,
            "last_query_time": None
        }
        
        # Initialize pipeline
        self.initialize_pipeline()
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"drug_rag_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("DrugRAGPipeline")
        self.logger.info("Drug RAG Pipeline initialized")
    
    def initialize_pipeline(self):
        """Initialize all pipeline components"""
        
        try:
            self.logger.info(f"Initializing Drug RAG Pipeline...")
            
            # 1. Initialize Vector Store
            self.logger.info(f"Loading vector store: {self.vector_db_name}")
            self.vector_store = DrugVectorStore(
                db_name=self.vector_db_name,
                embedding_model="openai"
            )
            
            if not self.vector_store.load_vectorstore():
                # Try main database as fallback
                self.logger.warning(f"Test database not found, trying main database...")
                self.vector_store = DrugVectorStore(
                    db_name="drug_vector_db",
                    embedding_model="openai"
                )
                
                if not self.vector_store.load_vectorstore():
                    raise ValueError("No vector database found. Please create one first.")
            
            self.logger.info("✅ Vector store loaded successfully")
            
            # 2. Initialize Multi-Query Retriever
            self.logger.info("Setting up multi-query retriever...")
            self.retriever = DrugMultiQueryRetriever(self.vector_store, model_name=self.model_name)
            self.logger.info("✅ Multi-query retriever initialized")
            
            # 3. Initialize LLM Generator
            self.logger.info(f"Setting up LLM generator: {self.model_name}")
            self.llm = DrugLLM(model_name=self.model_name)
            self.logger.info("✅ LLM generator initialized")
            
            self.logger.info("🚀 Drug RAG Pipeline ready!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def query(self, 
              question: str, 
              k: int = 5,
              include_sources: bool = True,
              response_format: str = "comprehensive") -> Dict[str, Any]:
        """
        Process a complete drug query through the RAG pipeline
        
        Args:
            question: User's drug-related question
            k: Number of documents to retrieve
            include_sources: Whether to include source documents
            response_format: "simple", "comprehensive", or "structured"
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        
        start_time = datetime.now()
        self.stats["queries_processed"] += 1
        
        try:
            self.logger.info(f"Processing query: {question}")
            
            # 1. Retrieve relevant documents using multi-query
            self.logger.info("🔍 Retrieving relevant documents...")
            retrieved_docs = self.retriever.retrieve_documents(question, k=k)
            
            if not retrieved_docs:
                self.logger.warning("No documents retrieved")
                return self._create_error_response("No relevant documents found", question)
            
            self.logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
            
            # 2. Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # 3. Generate answer using LLM
            self.logger.info("🧠 Generating answer...")
            
            if response_format == "comprehensive":
                generation_result = self.llm.generate_comprehensive_answer(question, context)
                answer = generation_result["text_answer"]
                query_type = generation_result["query_type"]
                structured_data = generation_result.get("structured_data")
            elif response_format == "structured":
                structured_data = self.llm.generate_structured_safety_info(question, context)
                answer = f"Structured data generated for {structured_data.get('drug_name', 'drug query')}"
                query_type = "structured"
            else:  # simple
                answer = self.llm.generate_answer(question, context, "general")
                query_type = "general"
                structured_data = None
            
            # 4. Prepare response
            response_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "question": question,
                "answer": answer,
                "query_type": query_type,
                "response_time_seconds": response_time,
                "documents_retrieved": len(retrieved_docs),
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "success": True
            }
            
            # Add structured data if available
            if structured_data:
                response["structured_data"] = structured_data
            
            # Add source documents if requested
            if include_sources:
                response["sources"] = self._format_sources(retrieved_docs)
            
            # Update statistics
            self.stats["successful_queries"] += 1
            self._update_stats(response_time)
            
            self.logger.info(f"✅ Query processed successfully in {response_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            self.stats["failed_queries"] += 1
            return self._create_error_response(str(e), question)
    
    def _prepare_context(self, documents: List) -> str:
        """Prepare context string from retrieved documents"""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            active_ingredient = doc.metadata.get('active_ingredient', 'Unknown')
            form = doc.metadata.get('form', 'Unknown')
            marketing_status = doc.metadata.get('marketing_status', 'Unknown')
            
            # Format document
            doc_text = f"Document {i}:\n"
            doc_text += f"Drug: {drug_name}\n"
            doc_text += f"Active Ingredient: {active_ingredient}\n"
            doc_text += f"Form: {form}\n"
            doc_text += f"Status: {marketing_status}\n"
            doc_text += f"Content: {doc.page_content}\n"
            
            context_parts.append(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, documents: List) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        
        sources = []
        for i, doc in enumerate(documents, 1):
            source = {
                "source_id": i,
                "drug_name": doc.metadata.get('drug_name', 'Unknown'),
                "active_ingredient": doc.metadata.get('active_ingredient', 'Unknown'),
                "form": doc.metadata.get('form', 'Unknown'),
                "marketing_status": doc.metadata.get('marketing_status', 'Unknown'),
                "application_no": doc.metadata.get('application_no', 'Unknown'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)
        
        return sources
    
    def _create_error_response(self, error_message: str, question: str) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            "question": question,
            "answer": f"I apologize, but I encountered an error: {error_message}",
            "error": error_message,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }
    
    def _update_stats(self, response_time: float):
        """Update pipeline statistics"""
        
        # Update average response time
        total_successful = self.stats["successful_queries"]
        current_avg = self.stats["average_response_time"]
        
        self.stats["average_response_time"] = (
            (current_avg * (total_successful - 1) + response_time) / total_successful
        )
        
        self.stats["last_query_time"] = datetime.now().isoformat()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        
        vector_stats = self.vector_store.get_stats() if self.vector_store else {}
        
        return {
            "pipeline_stats": self.stats,
            "vector_store_stats": vector_stats,
            "model_info": self.llm.get_model_info() if self.llm else {},
            "database_name": self.vector_db_name
        }
    
    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        
        self.logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            self.logger.info(f"Processing batch query {i}/{len(questions)}")
            result = self.query(question, **kwargs)
            results.append(result)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check pipeline health status"""
        
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check vector store
            if self.vector_store and self.vector_store.vectorstore:
                health_status["components"]["vector_store"] = "healthy"
            else:
                health_status["components"]["vector_store"] = "unhealthy"
                health_status["status"] = "degraded"
            
            # Check retriever
            if self.retriever:
                health_status["components"]["retriever"] = "healthy"
            else:
                health_status["components"]["retriever"] = "unhealthy"
                health_status["status"] = "degraded"
            
            # Check LLM
            if self.llm:
                health_status["components"]["llm"] = "healthy"
            else:
                health_status["components"]["llm"] = "unhealthy"
                health_status["status"] = "degraded"
            
            # Check API key
            if os.getenv("OPENAI_API_KEY"):
                health_status["components"]["openai_api"] = "configured"
            else:
                health_status["components"]["openai_api"] = "missing"
                health_status["status"] = "unhealthy"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


def main():
    """Test the complete Drug RAG Pipeline"""
    
    print("🚀 Testing Complete Drug RAG Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = DrugRAGPipeline()
        
        # Health check
        print("\n🏥 Pipeline Health Check:")
        health = pipeline.health_check()
        print(f"Status: {health['status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status}")
        
        if health['status'] != 'healthy':
            print("⚠️  Pipeline not fully healthy. Some features may not work.")
        
        # Test questions
        test_questions = [
            "What are the interactions between amoxicillin and ibuprofen?",
            "What is the FDA approval status of Ozempic?",
            "What are the side effects of metformin?",
            "Which drugs are available for diabetes treatment?",
            "What forms is acetaminophen available in?"
        ]
        
        print(f"\n🧪 Testing with {len(test_questions)} questions...")
        
        # Process each question
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i}/{len(test_questions)} ---")
            print(f"❓ Question: {question}")
            
            # Process query
            result = pipeline.query(
                question, 
                k=3, 
                include_sources=True,
                response_format="comprehensive"
            )
            
            # Display results
            if result["success"]:
                print(f"✅ Success ({result['response_time_seconds']:.2f}s)")
                print(f"🏷️  Query Type: {result['query_type']}")
                print(f"📄 Documents: {result['documents_retrieved']}")
                print(f"💬 Answer: {result['answer'][:150]}...")
                
                if result.get('structured_data'):
                    print(f"📊 Structured Data: Available")
                
                print(f"📚 Sources: {len(result.get('sources', []))} documents")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Pipeline statistics
        print(f"\n📊 Pipeline Statistics:")
        stats = pipeline.get_pipeline_stats()
        for category, data in stats.items():
            print(f"\n{category}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
        
        print(f"\n✅ End-to-End Pipeline testing complete!")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        print(f"💡 Make sure you have:")
        print(f"   - Vector database created (python index/create_vectorstore.py)")
        print(f"   - OpenAI API key in .env file")
        print(f"   - All required dependencies installed")


if __name__ == "__main__":
    main() 