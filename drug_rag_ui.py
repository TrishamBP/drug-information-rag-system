import os
import sys
import json
import gradio as gr
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import the complete pipeline
from rag_pipeline import DrugRAGPipeline

load_dotenv()


class DrugRAGInterface:
    """
    Gradio interface for the Drug RAG system
    """
    
    def __init__(self):
        """Initialize the RAG interface"""
        
        self.pipeline = None
        self.initialize_pipeline()
        
        # Sample questions organized by category
        self.sample_questions = self.get_sample_questions()
        
    def initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        
        try:
            self.pipeline = DrugRAGPipeline(
                vector_db_name="test_drug_vector_db",
                model_name="gpt-4o-mini"
            )
            return True
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            return False
    
    def get_sample_questions(self) -> Dict[str, List[str]]:
        """Get comprehensive sample questions for testing"""
        
        return {
            "Drug Interactions": [
                "What are the interactions between amoxicillin and ibuprofen?",
                "Can I take warfarin with acetaminophen?",
                "What drugs interact with metformin?",
                "Are there any contraindications with aspirin and blood thinners?",
                "What happens if I combine antibiotics with alcohol?"
            ],
            
            "Side Effects & Safety": [
                "What are the side effects of amoxicillin?",
                "Is metformin safe during pregnancy?",
                "What are the common adverse reactions to ibuprofen?",
                "What warnings should I know about acetaminophen?",
                "Are there any serious side effects with antibiotics?"
            ],
            
            "FDA Status & Approval": [
                "What is the FDA approval status of Ozempic?",
                "Is insulin approved for Type 1 diabetes?",
                "What drugs are FDA approved for hypertension?",
                "Which antibiotics have recent FDA approvals?",
                "What is the regulatory status of biosimilar drugs?"
            ],
            
            "Drug Forms & Dosages": [
                "What forms is acetaminophen available in?",
                "What are the available strengths of metformin?",
                "How is insulin administered?",
                "What dosage forms does ibuprofen come in?",
                "Are there liquid forms of antibiotics available?"
            ],
            
            "Generic vs Brand": [
                "What is the generic name for Tylenol?",
                "Are there generic versions of insulin available?",
                "What's the difference between Advil and ibuprofen?",
                "Which drugs have generic alternatives?",
                "Is there a generic version of Ozempic?"
            ],
            
            "Therapeutic Uses": [
                "What drugs are used for diabetes treatment?",
                "Which medications treat high blood pressure?",
                "What antibiotics are used for respiratory infections?",
                "What pain medications are available over-the-counter?",
                "Which drugs are used for heart conditions?"
            ],
            
            "Active Ingredients": [
                "What is the active ingredient in Tylenol?",
                "Which drugs contain acetaminophen?",
                "What active ingredients are in combination cold medications?",
                "What is the main component of aspirin?",
                "Which antibiotics contain penicillin?"
            ],
            
            "Drug Classes": [
                "What drugs belong to the ACE inhibitor class?",
                "Which medications are beta-blockers?",
                "What are the different types of antibiotics?",
                "Which drugs are considered NSAIDs?",
                "What medications are in the statin class?"
            ]
        }
    
    def process_query(self, 
                     question: str, 
                     response_format: str = "Comprehensive",
                     k_documents: int = 5,
                     include_sources: bool = True) -> Tuple[str, str, str]:
        """
        Process a drug query and return formatted results
        
        Returns:
            Tuple of (main_answer, sources_info, metadata_info)
        """
        
        if not self.pipeline:
            return "❌ Pipeline not initialized. Please check your setup.", "", ""
        
        if not question.strip():
            return "⚠️ Please enter a question about drugs or medications.", "", ""
        
        try:
            # Map UI format to pipeline format
            format_mapping = {
                "Simple": "simple",
                "Comprehensive": "comprehensive", 
                "Structured Data": "structured"
            }
            
            pipeline_format = format_mapping.get(response_format, "comprehensive")
            
            # Process the query
            result = self.pipeline.query(
                question=question,
                k=k_documents,
                include_sources=include_sources,
                response_format=pipeline_format
            )
            
            if not result["success"]:
                return f"❌ Error: {result.get('error', 'Unknown error')}", "", ""
            
            # Format main answer
            main_answer = self._format_main_answer(result)
            
            # Format sources
            sources_info = self._format_sources_info(result) if include_sources else ""
            
            # Format metadata
            metadata_info = self._format_metadata_info(result)
            
            return main_answer, sources_info, metadata_info
            
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}", "", ""
    
    def _format_main_answer(self, result: Dict[str, Any]) -> str:
        """Format the main answer section"""
        
        answer = result.get("answer", "No answer generated")
        query_type = result.get("query_type", "general")
        
        # Add structured data if available
        structured_data = result.get("structured_data")
        if structured_data:
            answer += f"\n\n📊 **Structured Information:**\n"
            
            if isinstance(structured_data, dict):
                for key, value in structured_data.items():
                    if key == "error":
                        continue
                    
                    if isinstance(value, list) and value:
                        answer += f"**{key.replace('_', ' ').title()}:** {', '.join(value)}\n"
                    elif value and value != "Not specified":
                        answer += f"**{key.replace('_', ' ').title()}:** {value}\n"
        
        return f"🔍 **Query Type:** {query_type.title()}\n\n💬 **Answer:**\n{answer}"
    
    def _format_sources_info(self, result: Dict[str, Any]) -> str:
        """Format the sources information"""
        
        sources = result.get("sources", [])
        if not sources:
            return "No source documents available."
        
        sources_text = f"📚 **Sources ({len(sources)} documents):**\n\n"
        
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            drug_name = source.get("drug_name", "Unknown")
            active_ingredient = source.get("active_ingredient", "Unknown")
            form = source.get("form", "Unknown")
            status = source.get("marketing_status", "Unknown")
            app_no = source.get("application_no", "Unknown")
            preview = source.get("content_preview", "No preview available")
            
            sources_text += f"**Source {i}:** {drug_name}\n"
            sources_text += f"- **Active Ingredient:** {active_ingredient}\n"
            sources_text += f"- **Form:** {form}\n"
            sources_text += f"- **Status:** {status}\n"
            sources_text += f"- **Application No:** {app_no}\n"
            sources_text += f"- **Preview:** {preview}\n\n"
        
        return sources_text
    
    def _format_metadata_info(self, result: Dict[str, Any]) -> str:
        """Format the metadata information"""
        
        response_time = result.get("response_time_seconds", 0)
        docs_retrieved = result.get("documents_retrieved", 0)
        model_used = result.get("model_used", "Unknown")
        timestamp = result.get("timestamp", "Unknown")
        
        metadata = f"⚡ **Performance:**\n"
        metadata += f"- **Response Time:** {response_time:.2f} seconds\n"
        metadata += f"- **Documents Retrieved:** {docs_retrieved}\n"
        metadata += f"- **Model Used:** {model_used}\n"
        metadata += f"- **Timestamp:** {timestamp}\n"
        
        return metadata
    
    def get_pipeline_status(self) -> str:
        """Get current pipeline status"""
        
        if not self.pipeline:
            return "❌ Pipeline not initialized"
        
        try:
            health = self.pipeline.health_check()
            status = health["status"]
            
            status_text = f"🏥 **Pipeline Status:** {status.upper()}\n\n"
            
            for component, comp_status in health["components"].items():
                emoji = "✅" if comp_status in ["healthy", "configured"] else "❌"
                status_text += f"{emoji} **{component.replace('_', ' ').title()}:** {comp_status}\n"
            
            # Add statistics
            stats = self.pipeline.get_pipeline_stats()
            pipeline_stats = stats.get("pipeline_stats", {})
            
            status_text += f"\n📊 **Statistics:**\n"
            status_text += f"- **Queries Processed:** {pipeline_stats.get('queries_processed', 0)}\n"
            status_text += f"- **Successful:** {pipeline_stats.get('successful_queries', 0)}\n"
            status_text += f"- **Failed:** {pipeline_stats.get('failed_queries', 0)}\n"
            status_text += f"- **Average Response Time:** {pipeline_stats.get('average_response_time', 0):.2f}s\n"
            
            return status_text
            
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
    
    def get_sample_questions_text(self) -> str:
        """Get formatted sample questions"""
        
        questions_text = "💡 **Sample Questions by Category:**\n\n"
        
        for category, questions in self.sample_questions.items():
            questions_text += f"**{category}:**\n"
            for i, question in enumerate(questions[:3], 1):  # Show first 3 per category
                questions_text += f"{i}. {question}\n"
            questions_text += "\n"
        
        return questions_text
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .status-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="Drug RAG System") as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🏥 Drug Information RAG System</h1>
                <p>Ask questions about medications, drug interactions, FDA approvals, and more!</p>
            </div>
            """)
            
            # Main interface
            with gr.Row():
                # Left column - Query interface
                with gr.Column(scale=2):
                    
                    # Query input
                    question_input = gr.Textbox(
                        label="💬 Enter your drug-related question:",
                        placeholder="e.g., What are the side effects of amoxicillin?",
                        lines=2
                    )
                    
                    # Settings
                    with gr.Row():
                        response_format = gr.Dropdown(
                            choices=["Simple", "Comprehensive", "Structured Data"],
                            value="Comprehensive",
                            label="Response Format"
                        )
                        
                        k_documents = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Documents to Retrieve"
                        )
                    
                    include_sources = gr.Checkbox(
                        label="Include Source Documents",
                        value=True
                    )
                    
                    # Submit button
                    submit_btn = gr.Button("🔍 Search Drug Information", variant="primary")
                    
                    # Quick question buttons
                    gr.HTML("<h3>🚀 Quick Questions:</h3>")
                    
                    with gr.Row():
                        quick_btns = []
                        for i, (category, questions) in enumerate(list(self.sample_questions.items())[:4]):
                            btn = gr.Button(f"{questions[0][:30]}...", size="sm")
                            quick_btns.append((btn, questions[0]))
                
                # Right column - Status and help
                with gr.Column(scale=1):
                    
                    # Pipeline status
                    status_display = gr.Markdown(
                        value=self.get_pipeline_status(),
                        label="System Status"
                    )
                    
                    # Refresh status button
                    refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
                    
                    # Sample questions
                    sample_questions_display = gr.Markdown(
                        value=self.get_sample_questions_text(),
                        label="Sample Questions"
                    )
            
            # Results section
            gr.HTML("<h2>📋 Results</h2>")
            
            with gr.Row():
                # Main answer
                with gr.Column(scale=2):
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="Enter a question above to get started!"
                    )
                
                # Sources and metadata
                with gr.Column(scale=1):
                    sources_output = gr.Markdown(
                        label="Sources",
                        value=""
                    )
                    
                    metadata_output = gr.Markdown(
                        label="Metadata",
                        value=""
                    )
            
            # Event handlers
            submit_btn.click(
                fn=self.process_query,
                inputs=[question_input, response_format, k_documents, include_sources],
                outputs=[answer_output, sources_output, metadata_output]
            )
            
            # Quick question buttons
            for btn, question in quick_btns:
                btn.click(
                    fn=lambda q=question: q,
                    outputs=question_input
                )
            
            # Refresh status
            refresh_status_btn.click(
                fn=self.get_pipeline_status,
                outputs=status_display
            )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 20px; color: #6c757d;">
                <p>🔬 Drug RAG System - Built with LangChain, OpenAI, and Gradio</p>
                <p>⚠️ This is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
            </div>
            """)
        
        return interface


def main():
    """Launch the Drug RAG Gradio interface"""
    
    print("🚀 Launching Drug RAG System Interface...")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Please create a .env file with your OpenAI API key")
        return
    
    try:
        # Create interface
        rag_interface = DrugRAGInterface()
        
        if not rag_interface.pipeline:
            print("❌ Failed to initialize RAG pipeline")
            print("💡 Make sure you have a vector database created:")
            print("   python index/create_vectorstore.py")
            return
        
        # Create and launch Gradio interface
        interface = rag_interface.create_interface()
        
        print("✅ Interface created successfully!")
        print("🌐 Launching web interface...")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main() 