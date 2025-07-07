import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()


class DrugSafetyInfo(BaseModel):
    """Structured model for drug safety information"""
    drug_name: str = Field(description="Name of the drug")
    safety_level: str = Field(description="Overall safety level: Safe/Caution/Warning")
    contraindications: List[str] = Field(description="List of contraindications")
    side_effects: List[str] = Field(description="Common side effects")
    interactions: List[str] = Field(description="Known drug interactions")
    warnings: List[str] = Field(description="Important warnings")


class DrugInteractionInfo(BaseModel):
    """Structured model for drug interaction information"""
    primary_drug: str = Field(description="Primary drug name")
    interacting_drugs: List[str] = Field(description="List of interacting drugs")
    interaction_severity: str = Field(description="Severity: Mild/Moderate/Severe")
    interaction_effects: List[str] = Field(description="Effects of the interaction")
    recommendations: List[str] = Field(description="Clinical recommendations")


class DrugGeneralInfo(BaseModel):
    """Structured model for general drug information"""
    drug_name: str = Field(description="Drug name")
    active_ingredient: str = Field(description="Active ingredient")
    drug_class: str = Field(description="Therapeutic class")
    indications: List[str] = Field(description="Medical indications")
    dosage_forms: List[str] = Field(description="Available dosage forms")
    fda_status: str = Field(description="FDA approval status")
    generic_available: bool = Field(description="Whether generic version is available")


class DrugLLM:
    """
    Drug-focused LLM generation with structured outputs
    Handles different types of drug queries with appropriate formatting
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            max_tokens=2000
        )
        
        # Set up different prompt templates
        self.setup_prompts()
        
    def setup_prompts(self):
        """Set up different prompt templates for various drug query types"""
        
        # General drug information prompt
        self.general_prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable pharmaceutical assistant. Based on the provided drug information context, answer the user's question with accurate, helpful information.

Focus on providing:
- Clear, factual information about drugs
- Proper drug names (brand and generic)
- FDA approval status and regulatory information
- Available dosage forms and strengths
- Basic safety information when relevant

If the context doesn't contain enough information to answer completely, clearly state what information is missing.

Context:
{context}

Question: {question}

Provide a comprehensive but concise answer:""")

        # Drug safety focused prompt with structured output
        self.safety_prompt = ChatPromptTemplate.from_template("""
You are a clinical pharmacist analyzing drug safety information. Based on the provided context, extract structured safety information about the drug(s) mentioned.

Context:
{context}

Question: {question}

Analyze the safety profile and provide structured information about:
1. Overall safety assessment
2. Contraindications (when NOT to use)
3. Common side effects
4. Drug interactions
5. Important warnings

Format your response as a clear, structured analysis. If specific safety data is not available in the context, state that clearly.""")

        # Drug interaction specific prompt
        self.interaction_prompt = ChatPromptTemplate.from_template("""
You are a clinical pharmacist specializing in drug interactions. Analyze the provided context for drug interaction information.

Context:
{context}

Question: {question}

Provide detailed information about:
1. Which drugs are mentioned and their potential interactions
2. Severity of any interactions (Mild/Moderate/Severe)
3. Clinical effects of the interactions
4. Recommendations for healthcare providers
5. Patient counseling points

If no interaction data is available in the context, clearly state this and recommend consulting additional resources.""")

        # Structured JSON output prompt
        self.json_prompt = ChatPromptTemplate.from_template("""
You are a pharmaceutical data analyst. Extract structured information from the provided drug context.

Context:
{context}

Question: {question}

Extract and format the following information as a JSON object:
- drug_name: Primary drug name
- active_ingredient: Active pharmaceutical ingredient
- drug_class: Therapeutic classification
- indications: List of medical uses
- dosage_forms: Available formulations
- fda_status: Regulatory approval status
- safety_notes: Key safety information

Respond with valid JSON only. If information is not available, use "Not specified" as the value.""")

    def generate_answer(self, question: str, context: str, response_type: str = "general") -> str:
        """
        Generate an answer based on the question type
        
        Args:
            question: User's question
            context: Retrieved document context
            response_type: Type of response ("general", "safety", "interaction", "json")
            
        Returns:
            Generated answer
        """
        
        try:
            # Select appropriate prompt based on response type
            if response_type == "safety":
                prompt = self.safety_prompt
            elif response_type == "interaction":
                prompt = self.interaction_prompt
            elif response_type == "json":
                prompt = self.json_prompt
            else:
                prompt = self.general_prompt
            
            # Create generation chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured_safety_info(self, question: str, context: str) -> Dict[str, Any]:
        """Generate structured safety information"""
        
        try:
            # Use JSON mode for structured output
            structured_llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            safety_json_prompt = ChatPromptTemplate.from_template("""
Extract drug safety information from the context and format as JSON.

Context: {context}
Question: {question}

Provide a JSON response with this structure:
{{
    "drug_name": "name of the drug",
    "safety_level": "Safe/Caution/Warning",
    "contraindications": ["list", "of", "contraindications"],
    "side_effects": ["common", "side", "effects"],
    "interactions": ["known", "drug", "interactions"],
    "warnings": ["important", "warnings"]
}}

If specific information is not available, use empty arrays or "Not specified".""")

            chain = safety_json_prompt | structured_llm | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Parse JSON response
            return json.loads(response)
            
        except Exception as e:
            return {
                "error": f"Failed to generate structured safety info: {str(e)}",
                "drug_name": "Unknown",
                "safety_level": "Not specified",
                "contraindications": [],
                "side_effects": [],
                "interactions": [],
                "warnings": []
            }
    
    def detect_query_type(self, question: str) -> str:
        """
        Detect the type of drug query to use appropriate response format
        
        Returns: "general", "safety", "interaction", "dosage", "approval"
        """
        
        question_lower = question.lower()
        
        # Safety-related keywords
        safety_keywords = ["side effect", "adverse", "contraindication", "warning", "safe", "danger", "risk"]
        if any(keyword in question_lower for keyword in safety_keywords):
            return "safety"
        
        # Interaction keywords
        interaction_keywords = ["interact", "combination", "together", "with", "and"]
        if any(keyword in question_lower for keyword in interaction_keywords):
            return "interaction"
        
        # Dosage keywords
        dosage_keywords = ["dose", "dosage", "how much", "strength", "mg", "ml"]
        if any(keyword in question_lower for keyword in dosage_keywords):
            return "dosage"
        
        # Approval/regulatory keywords
        approval_keywords = ["fda", "approval", "approved", "status", "regulatory"]
        if any(keyword in question_lower for keyword in approval_keywords):
            return "approval"
        
        return "general"
    
    def generate_comprehensive_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate a comprehensive answer with both text and structured data
        
        Returns:
            Dictionary with text answer, query type, and structured data
        """
        
        # Detect query type
        query_type = self.detect_query_type(question)
        
        # Generate text answer
        text_answer = self.generate_answer(question, context, query_type)
        
        # Generate structured data for safety queries
        structured_data = None
        if query_type == "safety":
            structured_data = self.generate_structured_safety_info(question, context)
        
        return {
            "question": question,
            "query_type": query_type,
            "text_answer": text_answer,
            "structured_data": structured_data,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": "OpenAI",
            "structured_output_support": True,
            "safety_analysis": True,
            "interaction_analysis": True
        }


def main():
    """Test the Drug LLM generation system"""
    
    print("🧪 Testing Drug LLM Generation System")
    print("=" * 60)
    
    # Initialize LLM
    drug_llm = DrugLLM()
    
    # Test context (mock drug information)
    test_context = """
    Amoxicillin is a penicillin-type antibiotic used to treat bacterial infections.
    Common side effects include nausea, vomiting, and diarrhea.
    It should not be used in patients allergic to penicillin.
    Drug interactions may occur with warfarin and methotrexate.
    FDA approved for various bacterial infections.
    Available in capsule, tablet, and liquid forms.
    """
    
    # Test questions
    test_questions = [
        "What are the side effects of amoxicillin?",
        "Can amoxicillin interact with warfarin?",
        "What is the FDA approval status of amoxicillin?",
        "What forms is amoxicillin available in?"
    ]
    
    print(f"🔧 Model info: {drug_llm.get_model_info()}")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        print(f"❓ Question: {question}")
        
        # Test comprehensive answer generation
        result = drug_llm.generate_comprehensive_answer(question, test_context)
        
        print(f"🏷️  Query Type: {result['query_type']}")
        print(f"💬 Answer: {result['text_answer'][:200]}...")
        
        if result['structured_data']:
            print(f"📊 Structured Data Available: Yes")
            print(f"   Safety Level: {result['structured_data'].get('safety_level', 'N/A')}")
    
    print(f"\n✅ LLM Generation testing complete!")


if __name__ == "__main__":
    main() 