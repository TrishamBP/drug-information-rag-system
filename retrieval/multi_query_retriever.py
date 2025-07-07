"""
Multi-Query Retrieval Module for Drug RAG System
Implements advanced retrieval strategies adapted from the user's Jupyter notebook
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from index.vectorstore import DrugVectorStore

# Load environment variables
load_dotenv()


class DrugMultiQueryRetriever:
    """
    Multi-Query Retrieval for Drug Information
    Generates multiple perspectives of drug-related queries for better retrieval
    """
    
    def __init__(self, vector_store: DrugVectorStore, model_name: str = "gpt-4o-mini"):
        self.vector_store = vector_store
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Get the basic retriever from vector store
        if not vector_store.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        
        self.retriever = vector_store.vectorstore.as_retriever()
        
        # Set up multi-query prompt template for drug queries
        self.setup_drug_query_prompt()
        self.setup_rag_prompt()
        
    def setup_drug_query_prompt(self):
        """Set up the prompt template for generating multiple drug query perspectives"""
        
        template = """You are an AI assistant specialized in pharmaceutical and drug information. 
Your task is to generate five different versions of the given user question to retrieve relevant 
drug information from a medical database. By generating multiple perspectives on the user question, 
your goal is to help overcome limitations of similarity search when finding drug information.

Focus on different aspects like:
- Drug names (brand names, generic names, active ingredients)
- Medical conditions and indications  
- Drug interactions and contraindications
- Dosage forms and administration routes
- Regulatory and approval information

Provide these alternative questions separated by newlines.
Original question: {question}"""

        self.prompt_perspectives = ChatPromptTemplate.from_template(template)
        
        # Create the query generation chain
        self.generate_queries = (
            self.prompt_perspectives 
            | self.llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
    
    def setup_rag_prompt(self):
        """Set up the RAG prompt template for drug questions"""
        
        template = """You are a knowledgeable pharmaceutical assistant. Answer the following question based on the provided drug information context. 

Provide accurate, helpful information about:
- Drug names and active ingredients
- Medical indications and uses
- Dosage forms and strengths  
- Regulatory status and approval information
- Safety considerations when relevant

If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        self.rag_prompt = ChatPromptTemplate.from_template(template)
    
    def get_unique_union(self, documents: List[List]) -> List:
        """
        Get unique union of retrieved documents
        Removes duplicates from multiple retrieval results
        """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        
        # Return as Document objects
        return [loads(doc) for doc in unique_docs]
    
    def retrieve_documents(self, question: str, k: int = 5) -> List:
        """
        Retrieve documents using multi-query strategy
        
        Args:
            question: Original user question
            k: Number of documents to retrieve per query
            
        Returns:
            List of unique retrieved documents
        """
        print(f"🔍 Original question: {question}")
        
        # Generate multiple query perspectives
        print("🧠 Generating query variations...")
        queries = self.generate_queries.invoke({"question": question})
        
        # Filter out empty queries
        queries = [q.strip() for q in queries if q.strip()]
        
        print(f"📝 Generated {len(queries)} query variations:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        
        # Retrieve documents for each query
        print("🔎 Retrieving documents for each query...")
        all_documents = []
        
        for i, query in enumerate(queries):
            try:
                docs = self.retriever.invoke(query)
                all_documents.append(docs[:k])  # Limit to k documents per query
                print(f"  Query {i+1}: Retrieved {len(docs[:k])} documents")
            except Exception as e:
                print(f"  Query {i+1}: Error - {e}")
                continue
        
        # Get unique union of all retrieved documents
        unique_docs = self.get_unique_union(all_documents)
        
        print(f"✅ Total unique documents retrieved: {len(unique_docs)}")
        return unique_docs
    
    def create_retrieval_chain(self):
        """Create the complete multi-query retrieval chain"""
        
        retrieval_chain = self.generate_queries | self.retriever.map() | self.get_unique_union
        return retrieval_chain
    
    def create_rag_chain(self):
        """Create the complete RAG chain with multi-query retrieval"""
        
        # Create retrieval chain
        retrieval_chain = self.create_retrieval_chain()
        
        # Create final RAG chain
        final_rag_chain = (
            {"context": retrieval_chain, 
             "question": itemgetter("question")} 
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return final_rag_chain
    
    def answer_question(self, question: str) -> str:
        """
        Answer a drug-related question using multi-query RAG
        
        Args:
            question: User's question about drugs
            
        Returns:
            Generated answer based on retrieved documents
        """
        print(f"\n{'='*60}")
        print(f"🏥 Drug RAG Query: {question}")
        print(f"{'='*60}")
        
        try:
            # Create and invoke RAG chain
            rag_chain = self.create_rag_chain()
            answer = rag_chain.invoke({"question": question})
            
            print(f"\n💊 Answer: {answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_retrieval_stats(self) -> dict:
        """Get statistics about the retrieval system"""
        
        stats = self.vector_store.get_stats()
        stats.update({
            "retrieval_model": self.model_name,
            "multi_query_enabled": True,
            "queries_per_question": 5
        })
        
        return stats


def main():
    """Test the Multi-Query Retrieval system"""
    
    print("🧪 Testing Drug Multi-Query Retrieval System")
    print("=" * 60)
    
    # Load vector store
    print("📁 Loading vector store...")
    vector_store = DrugVectorStore(
        db_name="drug_vector_db",
        embedding_model="openai"
    )
    
    # Load existing vector store
    if not vector_store.load_vectorstore():
        print("❌ Vector store not found. Please create it first:")
        print("   python index/create_vectorstore.py")
        return
    
    # Create multi-query retriever
    print("🔧 Setting up multi-query retriever...")
    retriever = DrugMultiQueryRetriever(vector_store)
    
    # Test questions
    test_questions = [
        "What are the side effects of amoxicillin?",
        "Which drugs are used for diabetes treatment?", 
        "What is the difference between generic and brand name drugs?",
        "Are there any drug interactions with warfarin?",
        "What FDA approval status does Ozempic have?"
    ]
    
    print(f"\n🔍 Testing with {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        
        # Test document retrieval only
        docs = retriever.retrieve_documents(question, k=3)
        
        print(f"📋 Sample retrieved documents:")
        for j, doc in enumerate(docs[:3], 1):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            form = doc.metadata.get('form', 'Unknown')
            print(f"  {j}. {drug_name} ({form})")
        
        # Test full RAG answer
        print(f"\n🤖 Generating answer...")
        answer = retriever.answer_question(question)
        print(f"📝 Answer length: {len(answer.split())} words")
    
    # Print stats
    print(f"\n📊 Retrieval System Statistics:")
    stats = retriever.get_retrieval_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Multi-Query Retrieval testing complete!")


if __name__ == "__main__":
    main() 