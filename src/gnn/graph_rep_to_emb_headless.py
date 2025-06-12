import os
import networkx as nx
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

# Import your existing graph builder class
from graph_representation_headless import SimpleUIGraph

def save_graphs_for_embedding(csv_path, output_dir="graph_data", proximity_threshold=0.2):
    """
    Process UI data and save NetworkX graphs to pickle files
    
    Args:
        csv_path (str): Path to CSV file with UI element data
        output_dir (str): Directory to save graph pickle files
        proximity_threshold (float): Threshold for connecting nodes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graph builder
    graph_builder = SimpleUIGraph(csv_path, proximity_threshold=proximity_threshold)
    
    # Get all image names
    image_names = graph_builder.df['Image Name'].unique()
    print(f"Found {len(image_names)} unique images")
    
    # Process each image
    for image_name in tqdm(image_names, desc="Processing images"):
        # Create graph
        G = graph_builder.create_graph(image_name)
        
        if G and len(G) > 0:
            # Save graph to pickle file
            output_path = os.path.join(output_dir, f"{Path(image_name).stem}_graph.gpickle")
            
            # Use pickle to save the graph (compatible with all versions)
            with open(output_path, 'wb') as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Optional: Also save visualization
            # viz_path = os.path.join(output_dir, f"{Path(image_name).stem}_graph.png")
            # graph_builder.visualize_graph(G, title=f"UI Graph: {image_name}", save_path=viz_path)
            
    print(f"Saved {len(image_names)} graphs to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Save NetworkX graphs for embedding")
    parser.add_argument("--csv_path", type=str, default="parsed_output_final.csv", 
                        help="Path to CSV file with UI element data")
    parser.add_argument("--output_dir", type=str, default="graph_data", 
                        help="Directory to save graph pickle files")
    parser.add_argument("--proximity_threshold", type=float, default=0.2, 
                        help="Threshold for connecting nodes")
    
    args = parser.parse_args()
    
    # Save graphs
    save_graphs_for_embedding(
        args.csv_path,
        args.output_dir,
        args.proximity_threshold
    )
    
    print("All graphs saved successfully!")

if __name__ == "__main__":
    main()
    
# Direct Mode: python src/gnn/graph_rep_to_emb_headless.py
# CLI Mode: python src/gnn/graph_rep_to_emb_headless.py --csv_path your_file.csv --output_dir your_output_dir --proximity_threshold 0.2