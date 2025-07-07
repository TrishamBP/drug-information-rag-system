import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import pandas as pd

from vectorstore import DrugVectorStore


class VectorVisualizer:
    """Visualize drug vector embeddings using t-SNE and Plotly"""
    
    def __init__(self, vector_store: DrugVectorStore):
        self.vector_store = vector_store
        
    def get_vectors_and_metadata(self) -> Tuple[np.ndarray, List[Dict]]:
        """Extract vectors and metadata from the vector store"""
        if not self.vector_store.vectorstore:
            raise ValueError("Vector store not initialized")
        
        collection = self.vector_store.vectorstore._collection
        result = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']
        
        return vectors, documents, metadatas
    
    def create_tsne_visualization(self, 
                                n_components: int = 2, 
                                random_state: int = 42,
                                perplexity: int = 30) -> Tuple[np.ndarray, List[Dict]]:
        """Create t-SNE reduced vectors for visualization"""
        vectors, documents, metadatas = self.get_vectors_and_metadata()
        
        print(f"Reducing {vectors.shape[0]:,} vectors from {vectors.shape[1]:,} to {n_components} dimensions using t-SNE...")
        
        # Create t-SNE model
        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        reduced_vectors = tsne.fit_transform(vectors)
        
        print(f"✅ t-SNE reduction complete")
        return reduced_vectors, documents, metadatas
    
    def plot_2d_scatter(self, 
                       reduced_vectors: np.ndarray, 
                       metadatas: List[Dict],
                       documents: List[str],
                       color_by: str = "form",
                       title: str = "2D Drug Vector Store Visualization") -> go.Figure:
        """Create 2D scatter plot using Plotly"""
        
        # Extract color mapping data
        color_values = [metadata.get(color_by, 'Unknown') for metadata in metadatas]
        drug_names = [metadata.get('drug_name', 'Unknown') for metadata in metadatas]
        forms = [metadata.get('form', 'Unknown') for metadata in metadatas]
        sponsors = [metadata.get('sponsor_name', 'Unknown') for metadata in metadatas]
        
        # Create hover text
        hover_texts = []
        for i, (drug, form, sponsor, doc) in enumerate(zip(drug_names, forms, sponsors, documents)):
            hover_text = f"Drug: {drug}<br>Form: {form}<br>Sponsor: {sponsor}<br>Text: {doc[:100]}..."
            hover_texts.append(hover_text)
        
        # Create the scatter plot
        fig = go.Figure(data=go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                opacity=0.8,
                colorscale='viridis'
            ),
            text=hover_texts,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='x', yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        
        return fig
    
    def plot_3d_scatter(self, 
                       color_by: str = "form",
                       title: str = "3D Drug Vector Store Visualization") -> go.Figure:
        """Create 3D scatter plot using Plotly"""
        
        # Get 3D t-SNE reduction
        reduced_vectors, documents, metadatas = self.create_tsne_visualization(n_components=3)
        
        # Extract color mapping data
        color_values = [metadata.get(color_by, 'Unknown') for metadata in metadatas]
        drug_names = [metadata.get('drug_name', 'Unknown') for metadata in metadatas]
        forms = [metadata.get('form', 'Unknown') for metadata in metadatas]
        sponsors = [metadata.get('sponsor_name', 'Unknown') for metadata in metadatas]
        
        # Create hover text
        hover_texts = []
        for i, (drug, form, sponsor, doc) in enumerate(zip(drug_names, forms, sponsors, documents)):
            hover_text = f"Drug: {drug}<br>Form: {form}<br>Sponsor: {sponsor}<br>Text: {doc[:100]}..."
            hover_texts.append(hover_text)
        
        # Create the 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                opacity=0.8,
                colorscale='viridis'
            ),
            text=hover_texts,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        
        return fig
    
    def analyze_clusters(self, reduced_vectors: np.ndarray, metadatas: List[Dict]) -> pd.DataFrame:
        """Analyze clusters in the reduced vector space"""
        
        # Create DataFrame for analysis
        df_data = {
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'drug_name': [m.get('drug_name', 'Unknown') for m in metadatas],
            'form': [m.get('form', 'Unknown') for m in metadatas],
            'marketing_status': [m.get('marketing_status', 'Unknown') for m in metadatas],
            'sponsor_name': [m.get('sponsor_name', 'Unknown') for m in metadatas]
        }
        
        df = pd.DataFrame(df_data)
        
        # Analyze form distribution
        print("📊 Form Distribution:")
        form_counts = df['form'].value_counts()
        for form, count in form_counts.head(10).items():
            print(f"  {form}: {count:,}")
        
        # Analyze sponsor distribution
        print(f"\n📊 Top Sponsors:")
        sponsor_counts = df['sponsor_name'].value_counts()
        for sponsor, count in sponsor_counts.head(10).items():
            print(f"  {sponsor}: {count:,}")
        
        # Analyze marketing status
        print(f"\n📊 Marketing Status Distribution:")
        status_counts = df['marketing_status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count:,}")
        
        return df
    
    def save_visualization(self, 
                          fig: go.Figure, 
                          filename: str = "drug_vectors_visualization.html"):
        """Save Plotly figure as HTML"""
        fig.write_html(filename)
        print(f"✅ Visualization saved to {filename}")
    
    def create_form_comparison_plot(self) -> go.Figure:
        """Create a plot comparing different drug forms"""
        reduced_vectors, documents, metadatas = self.create_tsne_visualization(n_components=2)
        
        # Group by form
        form_data = {}
        for i, metadata in enumerate(metadatas):
            form = metadata.get('form', 'Unknown')
            if form not in form_data:
                form_data[form] = {'x': [], 'y': [], 'drugs': []}
            
            form_data[form]['x'].append(reduced_vectors[i, 0])
            form_data[form]['y'].append(reduced_vectors[i, 1])
            form_data[form]['drugs'].append(metadata.get('drug_name', 'Unknown'))
        
        # Create figure with subplots for each form
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, (form, data) in enumerate(form_data.items()):
            if len(data['x']) > 10:  # Only show forms with enough data points
                fig.add_trace(go.Scatter(
                    x=data['x'],
                    y=data['y'],
                    mode='markers',
                    name=form,
                    marker=dict(color=colors[i % len(colors)], size=6),
                    text=data['drugs'],
                    hovertemplate=f"<b>{form}</b><br>Drug: %{{text}}<extra></extra>"
                ))
        
        fig.update_layout(
            title="Drug Forms Distribution in Vector Space",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            width=1000,
            height=700
        )
        
        return fig


def main():
    """Main function for testing visualization"""
    
    # Load or create vector store
    vector_store = DrugVectorStore(
        db_name="drug_vector_db",
        embedding_model="openai"
    )
    
    # Try to load existing vector store, create if doesn't exist
    if not vector_store.load_vectorstore():
        print("No existing vector store found. Creating new one...")
        jsonl_path = "data/processed/fda_documents.jsonl"
        documents = vector_store.load_documents_from_jsonl(jsonl_path)
        vector_store.create_vectorstore(documents)
    
    # Create visualizer
    visualizer = VectorVisualizer(vector_store)
    
    print("Creating 2D visualization...")
    # Create 2D visualization
    reduced_vectors, documents, metadatas = visualizer.create_tsne_visualization(n_components=2)
    
    # Create and show 2D plot
    fig_2d = visualizer.plot_2d_scatter(reduced_vectors, metadatas, documents, color_by="form")
    visualizer.save_visualization(fig_2d, "drug_vectors_2d.html")
    
    # Create form comparison plot
    print("Creating form comparison plot...")
    fig_forms = visualizer.create_form_comparison_plot()
    visualizer.save_visualization(fig_forms, "drug_forms_comparison.html")
    
    # Analyze clusters
    print("Analyzing vector clusters...")
    df = visualizer.analyze_clusters(reduced_vectors, metadatas)
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    fig_3d = visualizer.plot_3d_scatter(color_by="form")
    visualizer.save_visualization(fig_3d, "drug_vectors_3d.html")
    
    print("✅ All visualizations complete!")
    print("📄 Generated files:")
    print("  - drug_vectors_2d.html")
    print("  - drug_vectors_3d.html") 
    print("  - drug_forms_comparison.html")


if __name__ == "__main__":
    main() 