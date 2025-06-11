import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ast
import os
from pathlib import Path

class SimpleUIGraph:
    """A simple class to build graph representations of UI screens"""
    
    def __init__(self, csv_path, proximity_threshold=0.2):
        """
        Initialize with data source
        
        Args:
            csv_path: Path to CSV file with parsed UI data
            proximity_threshold: Maximum distance to connect elements with edges
        """
        self.csv_path = csv_path
        self.proximity_threshold = proximity_threshold
        self.df = pd.read_csv(csv_path)
        
        # Print basic stats
        print(f"Loaded {len(self.df)} UI elements from {len(self.df['Image Name'].unique())} images")
        
    def parse_bbox(self, bbox_str):
        """Parse bounding box string into coordinates"""
        try:
            bbox = ast.literal_eval(bbox_str)
            return bbox
        except:
            return None
            
    def get_elements_for_image(self, image_name):
        """Get all elements for a specific image"""
        return self.df[self.df['Image Name'] == image_name]
        
    def create_graph(self, image_name):
        """
        Create a graph representation for a specific image
        
        Args:
            image_name: Name of the image to process
            
        Returns:
            NetworkX graph with UI elements as nodes and proximity as edges
        """
        # Get elements for this image
        elements = self.get_elements_for_image(image_name)
        
        if len(elements) == 0:
            print(f"No elements found for {image_name}")
            return None
            
        # Create graph
        G = nx.Graph()
        
        # Process each element
        for idx, element in elements.iterrows():
            # Extract properties
            element_id = element['Element ID']
            element_type = element['Type']
            interactivity = element['Interactivity']
            content = element['Content']
            
            # Parse bounding box
            if 'Bounding Box' in element:
                bbox = self.parse_bbox(element['Bounding Box'])
            else:
                continue
                
            if not bbox:
                continue
                
            # Calculate center point (for node positioning)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Add node with properties
            G.add_node(
                element_id,
                type=element_type,
                interactive=interactivity,
                content=content,
                position=(center_x, center_y),
                bbox=bbox
            )
            
        # Add edges based on proximity
        nodes = list(G.nodes(data=True))
        for i in range(len(nodes)):
            node1_id, node1_data = nodes[i]
            pos1 = node1_data.get('position')
            
            if not pos1:
                continue
                
            for j in range(i+1, len(nodes)):
                node2_id, node2_data = nodes[j]
                pos2 = node2_data.get('position')
                
                if not pos2:
                    continue
                    
                # Calculate Euclidean distance
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Normalize
                normalized_distance = distance / 1000.0 
                
                # Add edge if elements are close
                if normalized_distance < self.proximity_threshold:
                    G.add_edge(node1_id, node2_id, weight=1.0-normalized_distance)
        
        print(f"Created graph with {len(G)} nodes and {G.number_of_edges()} edges")
        return G
        
    def visualize_graph(self, G, title=None, save_path=None, figsize=(12, 10)):
        """
        Visualize the graph
        
        Args:
            G: NetworkX graph to visualize
            title: Plot title
            save_path: Path to save image (if None, display only)
            figsize: Figure size in inches
        """
        if not G or len(G) == 0:
            print("Empty graph, nothing to visualize")
            return
            
        plt.figure(figsize=figsize)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'position')
        
        # If no positions, use spring layout
        if not pos:
            pos = nx.spring_layout(G)
            
        # Get node types and interactivity
        node_types = [G.nodes[n]['type'] for n in G.nodes()]
        
        # Fix for interactive detection - handle different boolean representations
        node_interactive = []
        for n in G.nodes():
            inter = G.nodes[n].get('interactive', 'FALSE')
            # Convert various forms of TRUE/FALSE to boolean
            if isinstance(inter, str):
                inter = inter.upper() == 'TRUE'
            node_interactive.append(inter)
        
        # Create colors - fix the MatplotlibDeprecationWarning
        unique_types = list(set(node_types))
        # Use the new recommended way of getting colormaps
        color_map = plt.colormaps['tab10']
        type_to_idx = {t: i % 10 for i, t in enumerate(unique_types)}  # Ensure index is within colormap range
        node_colors = [color_map(type_to_idx[t]) for t in node_types]
        
        # Create node sizes - larger for interactive elements
        node_sizes = [100 + (200 if inter else 0) for inter in node_interactive]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors, 
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=0.5,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Draw labels only for interactive nodes
        interactive_nodes = [n for n, inter in zip(G.nodes(), node_interactive) if inter]
        if interactive_nodes:
            # Get content or element ID for labels
            labels = {}
            for n in interactive_nodes:
                content = G.nodes[n].get('content', '')
                # Truncate long content
                if content and len(content) > 15:
                    content = content[:12] + '...'
                # If no content, use element type
                if not content:
                    content = G.nodes[n].get('type', '')
                labels[n] = content
            
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=8,
                font_color='black',
                font_weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
        else:
            print("No interactive nodes found to label")
        
        # Add legend for node types
        handles = []
        for t in unique_types:
            idx = type_to_idx[t]
            handle = plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=color_map(idx),
                markersize=8,
                label=t
            )
            handles.append(handle)
        
        # Add indicator for interactive nodes
        interactive_handle = plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='gray',
            markersize=12,
            label='Interactive Element'
        )
        handles.append(interactive_handle)
            
        plt.legend(handles=handles, loc='upper right')
        
        plt.title(title or "UI Element Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def analyze_image(self, image_name, output_dir="ui_graphs"):
        """Create and visualize graph for an image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create graph
        G = self.create_graph(image_name)
        
        if G and len(G) > 0:
            # Visualize graph
            save_path = os.path.join(output_dir, f"{Path(image_name).stem}_graph.png")
            self.visualize_graph(
                G,
                title=f"UI Graph: {image_name}",
                save_path=save_path
            )
            print(f"Graph visualization saved to {save_path}")
            return G
        else:
            print(f"Failed to create graph for {image_name}")
            return None
            
    def analyze_sample_images(self, n=5, output_dir="ui_graphs"):
        """
        Analyze a sample of images
        
        Args:
            n: Number of images to process. Use -1 to process all images.
            output_dir: Directory to save visualizations
        """
        image_names = self.df['Image Name'].unique()
        print(f"Found {len(image_names)} unique images")
        
        # If n is -1, process all images
        images_to_process = image_names if n == -1 else image_names[:n]
        
        for image_name in images_to_process:
            print(f"\nProcessing {image_name}")
            self.analyze_image(image_name, output_dir)

# Example usage
if __name__ == "__main__":
    # Create graph builder
    graph_builder = SimpleUIGraph("parsed_output_final.csv")
    
    # Analyze all images
    # To process all images, use n=-1 to indicate no limit
    graph_builder.analyze_sample_images(n=-1)