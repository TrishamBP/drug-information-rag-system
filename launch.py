import os
import sys
from pathlib import Path


def check_environment():
    """Check if environment is properly set up"""
    
    print("🔧 Checking Environment...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        print("💡 Please create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=your_key_here")
        return False
    
    print("✅ OpenAI API key found")
    
    # Check if vector database exists
    test_db = Path("test_drug_vector_db")
    main_db = Path("drug_vector_db")
    
    if test_db.exists() or main_db.exists():
        print("✅ Vector database found")
        return True
    else:
        print("❌ No vector database found")
        print("💡 Please create a vector database first:")
        print("   python index/create_vectorstore.py")
        return False


def show_menu():
    """Show the main menu"""
    
    print("\n🏥 Drug RAG System Launcher")
    print("=" * 40)
    print("1. 🌐 Launch Gradio Web Interface")
    print("2. 🧪 Test RAG Pipeline (Command Line)")
    print("3. 🔍 Test Retrieval Only")
    print("4. 🤖 Test LLM Generation Only")
    print("5. 📊 Check System Status")
    print("6. 📚 View Sample Questions")
    print("7. 🚪 Exit")
    print("=" * 40)


def launch_gradio_ui():
    """Launch the Gradio web interface"""
    
    print("🌐 Launching Gradio Web Interface...")
    
    try:
        import drug_rag_ui
        drug_rag_ui.main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")


def test_rag_pipeline():
    """Test the complete RAG pipeline from command line"""
    
    print("🧪 Testing Complete RAG Pipeline...")
    
    try:
        from rag_pipeline import DrugRAGPipeline
        
        # Initialize pipeline
        pipeline = DrugRAGPipeline()
        
        # Test questions
        test_questions = [
            "What are the side effects of amoxicillin?",
            "What drugs interact with warfarin?",
            "What is the FDA approval status of Ozempic?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i}/{len(test_questions)} ---")
            print(f"❓ Question: {question}")
            
            result = pipeline.query(question, k=3, include_sources=False)
            
            if result["success"]:
                print(f"✅ Success ({result['response_time_seconds']:.2f}s)")
                print(f"💬 Answer: {result['answer'][:200]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print("\n📊 Pipeline Statistics:")
        stats = pipeline.get_pipeline_stats()
        pipeline_stats = stats.get("pipeline_stats", {})
        print(f"  Queries Processed: {pipeline_stats.get('queries_processed', 0)}")
        print(f"  Success Rate: {pipeline_stats.get('successful_queries', 0)}/{pipeline_stats.get('queries_processed', 0)}")
        
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")


def test_retrieval():
    """Test retrieval functionality only"""
    
    print("🔍 Testing Retrieval Module...")
    
    try:
        from retrieval.test_retrieval import main as test_retrieval_main
        test_retrieval_main()
    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")


def test_llm_generation():
    """Test LLM generation only"""
    
    print("🤖 Testing LLM Generation...")
    
    try:
        from generation.drug_llm import main as test_llm_main
        test_llm_main()
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")


def check_system_status():
    """Check overall system status"""
    
    print("📊 Checking System Status...")
    
    try:
        from rag_pipeline import DrugRAGPipeline
        
        pipeline = DrugRAGPipeline()
        health = pipeline.health_check()
        
        print(f"\n🏥 System Status: {health['status'].upper()}")
        print("Components:")
        for component, status in health['components'].items():
            emoji = "✅" if status in ["healthy", "configured"] else "❌"
            print(f"  {emoji} {component.replace('_', ' ').title()}: {status}")
        
        # Get statistics
        stats = pipeline.get_pipeline_stats()
        pipeline_stats = stats.get("pipeline_stats", {})
        
        print(f"\n📈 Statistics:")
        print(f"  Total Queries: {pipeline_stats.get('queries_processed', 0)}")
        print(f"  Successful: {pipeline_stats.get('successful_queries', 0)}")
        print(f"  Failed: {pipeline_stats.get('failed_queries', 0)}")
        print(f"  Avg Response Time: {pipeline_stats.get('average_response_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")


def view_sample_questions():
    """Display sample questions"""
    
    print("📚 Sample Questions...")
    
    try:
        with open("sample_questions.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Show first few categories
        lines = content.split('\n')
        for line in lines[:50]:  # First 50 lines
            print(line)
        
        print("\n... (see sample_questions.md for complete list)")
        
    except FileNotFoundError:
        print("❌ sample_questions.md not found")
        
        # Show a few built-in examples
        sample_questions = [
            "What are the side effects of amoxicillin?",
            "What drugs interact with warfarin?",
            "What is the FDA approval status of Ozempic?",
            "What forms is acetaminophen available in?",
            "Which drugs are used for diabetes treatment?"
        ]
        
        print("📚 Quick Sample Questions:")
        for i, q in enumerate(sample_questions, 1):
            print(f"  {i}. {q}")


def main():
    """Main launcher function"""
    
    print("🚀 Drug RAG System Launcher")
    print("Starting system checks...")
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Environment setup incomplete.")
        print("Please fix the issues above before proceeding.")
        return
    
    print("✅ Environment checks passed!")
    
    # Main interaction loop
    while True:
        show_menu()
        
        try:
            choice = input("\n🔸 Select an option (1-7): ").strip()
            
            if choice == "1":
                launch_gradio_ui()
            elif choice == "2":
                test_rag_pipeline()
            elif choice == "3":
                test_retrieval()
            elif choice == "4":
                test_llm_generation()
            elif choice == "5":
                check_system_status()
            elif choice == "6":
                view_sample_questions()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")
            
            # Wait for user before showing menu again
            if choice != "7":
                input("\n⏸️  Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main() 