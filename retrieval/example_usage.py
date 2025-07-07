"""
Example Usage: Drug RAG Multi-Query Retrieval System
Shows how to use the system step by step
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from index.vectorstore import DrugVectorStore
from multi_query_retriever import DrugMultiQueryRetriever

load_dotenv()


def example_basic_usage():
    """Example 1: Basic Multi-Query Retrieval"""
    print("📚 Example 1: Basic Multi-Query Retrieval")
    print("=" * 50)
    
    # Step 1: Load your vector store
    print("1️⃣ Loading vector store...")
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",  # or "drug_vector_db"
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        print("❌ Vector store not found. Please create it first.")
        return
    
    # Step 2: Create multi-query retriever
    print("2️⃣ Setting up multi-query retriever...")
    retriever = DrugMultiQueryRetriever(vector_store)
    
    # Step 3: Ask a question and retrieve documents
    question = "What are the side effects of antibiotics?"
    print(f"3️⃣ Retrieving documents for: '{question}'")
    
    docs = retriever.retrieve_documents(question, k=5)
    
    # Step 4: Display results
    print(f"\n📋 Retrieved {len(docs)} relevant documents:")
    for i, doc in enumerate(docs[:5], 1):
        drug_name = doc.metadata.get('drug_name', 'Unknown')
        active_ingredient = doc.metadata.get('active_ingredient', 'Unknown')
        form = doc.metadata.get('form', 'Unknown')
        print(f"  {i}. {drug_name} ({active_ingredient}) - {form}")


def example_full_rag_pipeline():
    """Example 2: Full RAG Pipeline with Answer Generation"""
    print("\n📚 Example 2: Full RAG Pipeline")
    print("=" * 50)
    
    # Load vector store
    print("1️⃣ Loading vector store...")
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        print("❌ Vector store not found. Please create it first.")
        return
    
    # Create retriever
    print("2️⃣ Setting up RAG pipeline...")
    retriever = DrugMultiQueryRetriever(vector_store)
    
    # Ask questions and get full answers
    questions = [
        "What medications are approved for diabetes treatment?",
        "Are there any generic versions of insulin available?",
        "What are the different forms of ibuprofen?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n3️⃣ Question {i}: {question}")
        
        # Get full RAG answer
        answer = retriever.answer_question(question)
        
        print(f"🤖 Generated Answer:")
        print(f"   {answer[:200]}..." if len(answer) > 200 else f"   {answer}")


def example_custom_queries():
    """Example 3: Custom Query Types"""
    print("\n📚 Example 3: Different Types of Drug Queries")
    print("=" * 50)
    
    # Load vector store
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        print("❌ Vector store not found.")
        return
    
    retriever = DrugMultiQueryRetriever(vector_store)
    
    # Different query types
    query_types = {
        "Drug Interaction": "What drugs interact with warfarin?",
        "Dosage Form": "What are the available forms of acetaminophen?",  
        "Active Ingredient": "Which drugs contain the active ingredient metformin?",
        "FDA Status": "What is the approval status of biosimilar drugs?",
        "Brand vs Generic": "What are the generic names for Tylenol?"
    }
    
    for query_type, question in query_types.items():
        print(f"\n🔍 {query_type} Query: {question}")
        
        # Just retrieve documents (faster than full RAG)
        docs = retriever.retrieve_documents(question, k=3)
        
        print(f"📋 Top {len(docs[:3])} results:")
        for i, doc in enumerate(docs[:3], 1):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            marketing_status = doc.metadata.get('marketing_status', 'Unknown')
            print(f"  {i}. {drug_name} - Status: {marketing_status}")


def main():
    """Run all examples"""
    print("🚀 Drug RAG Multi-Query System - Usage Examples")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Please create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=your_key_here")
        return
    
    # Run examples
    try:
        example_basic_usage()
        example_full_rag_pipeline() 
        example_custom_queries()
        
        print(f"\n✅ All examples completed!")
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   ✅ Multi-Query Generation (5 perspectives per question)")
        print(f"   ✅ Document Retrieval with Deduplication")
        print(f"   ✅ Full RAG Pipeline with Answer Generation")
        print(f"   ✅ Different Query Types (interactions, forms, ingredients)")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print(f"💡 Make sure you have a vector store created first:")
        print(f"   python index/create_vectorstore.py")


if __name__ == "__main__":
    main() 