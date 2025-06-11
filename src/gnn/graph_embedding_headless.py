import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class SimpleGNN(nn.Module):
    """
    Simple Graph Neural Network using Graph Convolutional Network (GCN) layers
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2):
        """
        Initialize the GNN
        
        Args:
            input_dim (int): Dimension of input node features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output embeddings
            num_layers (int): Number of GCN layers
        """
        super(SimpleGNN, self).__init__()
        
        # First layer: input to hidden
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Middle layers (if any)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Last layer: hidden to output
        self.conv_out = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x (Tensor): Node features [num_nodes, input_dim]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            batch (Tensor): Batch vector [num_nodes]
            
        Returns:
            embeddings (Tensor): Node or graph embeddings
        """
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Middle layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Last layer
        x = self.conv_out(x, edge_index)
        
        # If batch is provided, create graph-level embeddings
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class UIGraphEmbedder:
    """
    Class to create embeddings from UI element graphs using GNN
    """
    def __init__(self, hidden_dim=128, output_dim=64, num_layers=2, device=None):
        """
        Initialize the embedder
        
        Args:
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output embeddings
            num_layers (int): Number of GNN layers
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Model will be initialized when we know the input dimension
        self.model = None
        
    def convert_networkx_to_pyg(self, G):
        """
        Convert NetworkX graph to PyTorch Geometric data
        
        Args:
            G (nx.Graph): NetworkX graph
            
        Returns:
            data (torch_geometric.data.Data): PyG data object
        """
        # Extract node features
        node_features = []
        
        # Map from node ID to index
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # Process nodes to create feature matrix
        for node in G.nodes():
            features = []
            
            # Extract node attributes and convert to numeric features
            node_data = G.nodes[node]
            
            # Type (one-hot encode)
            if 'type' in node_data:
                if node_data['type'] == 'text':
                    features.append(1.0)  # text
                    features.append(0.0)  # icon
                else:  # icon or other
                    features.append(0.0)  # text
                    features.append(1.0)  # icon
            else:
                features.extend([0.0, 0.0])  # Unknown type
            
            # Interactivity (boolean to float)
            if 'interactive' in node_data:
                if isinstance(node_data['interactive'], str):
                    # Convert string representation to boolean
                    interactive = node_data['interactive'].upper() == 'TRUE'
                else:
                    interactive = bool(node_data['interactive'])
                features.append(float(interactive))
            else:
                features.append(0.0)  # Not interactive
            
            # Position (normalized)
            if 'position' in node_data:
                # Normalize positions to [0, 1] range
                # Assuming positions are in pixels and typical screen might be up to 1920x1080
                x, y = node_data['position']
                features.append(float(x) / 1920.0)
                features.append(float(y) / 1080.0)
            else:
                features.extend([0.0, 0.0])  # No position
            
            # Convert to tensor
            node_features.append(features)
        
        # Create edge index
        edge_index = []
        for src, dst in G.edges():
            edge_index.append([node_mapping[src], node_mapping[dst]])
            # Add reverse edge for undirected graph
            edge_index.append([node_mapping[dst], node_mapping[src]])
        
        # Convert to PyG data
        x = torch.tensor(node_features, dtype=torch.float)
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data = Data(x=x, edge_index=edge_index)
        else:
            # Handle case with no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def initialize_model(self, input_dim):
        """
        Initialize the GNN model
        
        Args:
            input_dim (int): Dimension of input node features
        """
        self.model = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers
        ).to(self.device)
    
    def generate_embeddings(self, graphs, normalize=True):
        """
        Generate embeddings for a list of graphs
        
        Args:
            graphs (dict): Dictionary of NetworkX graphs keyed by image name
            normalize (bool): Whether to normalize the embeddings
            
        Returns:
            embeddings (dict): Dictionary of embeddings keyed by image name
        """
        # Convert NetworkX graphs to PyG data
        pyg_data = {}
        for img_name, G in tqdm(graphs.items(), desc="Converting graphs to PyG format"):
            pyg_data[img_name] = self.convert_networkx_to_pyg(G)
        
        # Check if we have any graphs
        if not pyg_data:
            raise ValueError("No valid graphs provided")
        
        # Initialize model if needed
        input_dim = next(iter(pyg_data.values())).x.shape[1]
        if self.model is None:
            self.initialize_model(input_dim)
            print(f"Initialized model with input dimension: {input_dim}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate embeddings
        embeddings = {}
        with torch.no_grad():
            for img_name, data in tqdm(pyg_data.items(), desc="Generating embeddings"):
                # Move data to device
                data = data.to(self.device)
                
                # Generate embedding by averaging node embeddings
                node_embeddings = self.model(data.x, data.edge_index)
                graph_embedding = node_embeddings.mean(dim=0)
                
                # Move to CPU and convert to numpy
                embeddings[img_name] = graph_embedding.cpu().numpy()
        
        # Normalize embeddings if requested
        if normalize:
            # Stack all embeddings
            all_embeddings = np.stack(list(embeddings.values()))
            
            # Calculate mean and std
            mean = np.mean(all_embeddings, axis=0)
            std = np.std(all_embeddings, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            
            # Normalize
            for img_name in embeddings:
                embeddings[img_name] = (embeddings[img_name] - mean) / std
        
        return embeddings
    
    def save_embeddings(self, embeddings, output_dir):
        """
        Save embeddings to disk
        
        Args:
            embeddings (dict): Dictionary of embeddings keyed by image name
            output_dir (str): Directory to save embeddings
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as NPZ
        np.savez_compressed(
            os.path.join(output_dir, "ui_embeddings.npz"),
            **embeddings
        )
        
        # Save metadata as JSON
        metadata = {
            "num_embeddings": len(embeddings),
            "embedding_dim": self.output_dim,
            "image_names": list(embeddings.keys()),
            "model_config": {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers
            }
        }
        
        with open(os.path.join(output_dir, "embeddings_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved embeddings and metadata to {output_dir}")
    
    def visualize_embeddings(self, embeddings, output_dir):
        """
        Visualize embeddings using PCA
    
        Args:
            embeddings (dict): Dictionary of embeddings keyed by image name
            output_dir (str): Directory to save visualization
        """
        os.makedirs(output_dir, exist_ok=True)
    
        # Extract image names and embeddings
        image_names = list(embeddings.keys())
        embeddings_array = np.array([embeddings[img] for img in image_names])
    
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
    
        # Create plot
        plt.figure(figsize=(12, 10))
    
        # Extract frame indices for coloring (assuming format like 'frame_X.png')
        frame_indices = []
        for img_name in image_names:
            try:
                # Try to extract a number from the image name
                frame_index = int(Path(img_name).stem.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                # If extraction fails, use the position in the list
                frame_index = image_names.index(img_name)
            frame_indices.append(frame_index)
    
        # Plot points with color based on sequence
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=frame_indices,
            cmap='viridis',
            alpha=0.8,
            s=100
        )
    
        # Add frame indices as annotations
        for i, img_name in enumerate(image_names):
            try:
                # Try to extract a frame index from the image name
                frame_index = Path(img_name).stem.split('_')[1].split('.')[0]
            except IndexError:
                # If extraction fails, use the original name
                frame_index = Path(img_name).stem
            plt.annotate(
                frame_index,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9
            )
    
        # Add a colorbar to show the sequence
        plt.colorbar(scatter, label='Frame Sequence')

        plt.title(f"PCA Visualization of {len(embeddings)} UI Graph Embeddings")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
    
        # Save plot
        plt.savefig(os.path.join(output_dir, "embeddings_pca.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
        # Create interactive HTML visualization if plotly is available
        try:
            import plotly.express as px
            import plotly.io as pio

            # Create dataframe for plotly
            df = pd.DataFrame({
                'PC1': embeddings_2d[:, 0],
                'PC2': embeddings_2d[:, 1],
                'Image': [Path(img).stem for img in image_names],
                'FrameIndex': frame_indices
            })
        
            # Create interactive plot
            fig = px.scatter(
                df, x='PC1', y='PC2', 
                color='FrameIndex', hover_name='Image',
                title=f"PCA Visualization of {len(embeddings)} UI Graph Embeddings"
            )

            # Save as HTML
            pio.write_html(fig, os.path.join(output_dir, "embeddings_pca_interactive.html"))
            print("Created interactive visualization")
        
        except ImportError:
            print("Plotly not available, skipping interactive visualization")
        
        print(f"Saved visualization to {output_dir}")

def load_graphs_from_directory(graph_dir):
    """
    Load NetworkX graphs from pickle files in a directory
    
    Args:
        graph_dir (str): Directory containing graph pickle files
        
    Returns:
        graphs (dict): Dictionary of NetworkX graphs keyed by image name
    """
    graphs = {}
    
    # Check if directory exists
    if not os.path.exists(graph_dir):
        raise ValueError(f"Directory not found: {graph_dir}")
    
    # Load graphs from pickle files
    for filename in os.listdir(graph_dir):
        if filename.endswith(".gpickle"):
            img_name = filename.replace("_graph.gpickle", "")
            graph_path = os.path.join(graph_dir, filename)
            
            try:
                # Use pickle directly for compatibility
                with open(graph_path, 'rb') as f:
                    G = pickle.load(f)
                graphs[img_name] = G
            except Exception as e:
                print(f"Error loading graph {filename}: {e}")
    
    return graphs

def main():
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for UI graphs")
    parser.add_argument("--graph_dir", type=str, required=True, help="Directory containing graph pickle files")
    parser.add_argument("--output_dir", type=str, default="ui_embeddings", help="Directory to save embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimension of hidden layers")
    parser.add_argument("--output_dim", type=int, default=64, help="Dimension of output embeddings")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load graphs
    graphs = load_graphs_from_directory(args.graph_dir)
    print(f"Loaded {len(graphs)} graphs")
    
    # Create embedder
    embedder = UIGraphEmbedder(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        device=args.device
    )
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(graphs)
    
    # Save embeddings
    embedder.save_embeddings(embeddings, args.output_dir)
    
    # Visualize embeddings
    embedder.visualize_embeddings(embeddings, args.output_dir)
    
    print("Done!")
    
if __name__ == "__main__":
    main()
    
    
# Direct mode: python src/gnn/graph_embedding_headless.py
# CLI Mode: python src/gnn/graph_embedding_headless.py --graph_dir path/to/graphs --output_dir path/to/output