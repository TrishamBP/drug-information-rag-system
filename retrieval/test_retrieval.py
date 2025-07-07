"""
Simple test script for Multi-Query Retrieval
Tests each component step by step
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from index.vectorstore import DrugVectorStore
from retrieval.multi_query_retriever import DrugMultiQueryRetriever

load_dotenv()


def test_basic_retrieval():
    """Test basic retrieval functionality"""
    print("🧪 Step 1: Testing Basic Vector Store Retrieval")
    print("=" * 50)
    
    # Load vector store
    print("📁 Loading vector store...")
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",  # Use test database first
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        print("❌ Test vector store not found.")
        print("💡 Trying main database...")
        
        # Try main database
        vector_store = DrugVectorStore(
            db_name="drug_vector_db",
            embedding_model="openai"
        )
        
        if not vector_store.load_vectorstore():
            print("❌ No vector store found. Please create one first:")
            print("   python index/create_vectorstore.py")
            return False
    
    print("✅ Vector store loaded successfully!")
    
    # Test basic retrieval
    print("\n🔍 Testing basic similarity search...")
    test_query = "amoxicillin antibiotic"
    
    try:
        results = vector_store.vectorstore.similarity_search(test_query, k=3)
        print(f"✅ Found {len(results)} results for '{test_query}'")
        
        for i, doc in enumerate(results, 1):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            form = doc.metadata.get('form', 'Unknown')
            print(f"  {i}. {drug_name} - {form}")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic retrieval failed: {e}")
        return False


def test_multi_query_generation():
    """Test multi-query generation without retrieval"""
    print("\n🧪 Step 2: Testing Multi-Query Generation")
    print("=" * 50)
    
    # Load vector store
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        vector_store = DrugVectorStore(
            db_name="drug_vector_db", 
            embedding_model="openai"
        )
        if not vector_store.load_vectorstore():
            print("❌ No vector store available")
            return False
    
    try:
        # Create retriever
        retriever = DrugMultiQueryRetriever(vector_store)
        
        # Test query generation
        test_question = "What are the side effects of amoxicillin?"
        print(f"🔍 Original question: {test_question}")
        
        # Generate queries manually to see the process
        print("\n🧠 Generating alternative queries...")
        queries = retriever.generate_queries.invoke({"question": test_question})
        
        print(f"📝 Generated {len(queries)} alternative queries:")
        for i, query in enumerate(queries, 1):
            if query.strip():
                print(f"  {i}. {query.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-query generation failed: {e}")
        return False


def test_full_multi_query_retrieval():
    """Test full multi-query retrieval process"""
    print("\n🧪 Step 3: Testing Full Multi-Query Retrieval")
    print("=" * 50)
    
    # Load vector store
    vector_store = DrugVectorStore(
        db_name="test_drug_vector_db",
        embedding_model="openai"
    )
    
    if not vector_store.load_vectorstore():
        vector_store = DrugVectorStore(
            db_name="drug_vector_db",
            embedding_model="openai"
        )
        if not vector_store.load_vectorstore():
            print("❌ No vector store available")
            return False
    
    try:
        # Create retriever
        retriever = DrugMultiQueryRetriever(vector_store)
        
        # Test questions
        test_questions = [
            "What drugs are used for diabetes?",
            "Side effects of antibiotics",
            "FDA approved pain medications"
        ]
        
        for question in test_questions:
            print(f"\n--- Testing: {question} ---")
            
            # Retrieve documents
            docs = retriever.retrieve_documents(question, k=3)
            
            print(f"📋 Retrieved {len(docs)} unique documents:")
            for i, doc in enumerate(docs[:5], 1):  # Show first 5
                drug_name = doc.metadata.get('drug_name', 'Unknown')
                active_ingredient = doc.metadata.get('active_ingredient', 'Unknown')
                form = doc.metadata.get('form', 'Unknown')
                print(f"  {i}. {drug_name} ({active_ingredient}) - {form}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full retrieval failed: {e}")
        return False


def main():
    """Run all retrieval tests"""
    print("🚀 Drug RAG Multi-Query Retrieval Testing")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Please set up your .env file with OpenAI API key")
        return
    
    # Run tests step by step
    tests = [
        test_basic_retrieval,
        test_multi_query_generation, 
        test_full_multi_query_retrieval
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print(f"\n🎉 All tests passed! Multi-Query Retrieval is working!")
        print(f"💡 Next step: Test full RAG chain with:")
        print(f"   python index/multi_query_retriever.py")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 