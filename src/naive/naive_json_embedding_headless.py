import os
import json
import pandas as pd
import numpy as np
import ast
import re
import argparse
import sys
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def validate_csv_path(csv_path):
    """Validate CSV file path and prompt for correction if needed."""
    while True:
        if not csv_path:
            csv_path = input("Please enter the CSV file path: ").strip()
            continue
            
        if not os.path.exists(csv_path):
            print(f"Error: CSV file '{csv_path}' not found.")
            csv_path = input("Please enter a valid CSV file path: ").strip()
            continue
            
        if not csv_path.lower().endswith('.csv'):
            print(f"Error: '{csv_path}' is not a CSV file.")
            csv_path = input("Please enter a valid CSV file path (.csv): ").strip()
            continue
            
        return csv_path

def validate_output_path(output_path, csv_path):
    """Validate output directory path and create if needed."""
    if not output_path:
        # Default: create output folder next to CSV file
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(csv_dir, f"output_embeddings_{csv_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    return output_path

def load_model_with_fallback(model_name):
    """Load SentenceTransformer model with fallback handling."""
    try:
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {str(e)}")
        
        # Try fallback models
        fallback_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2']
        
        for fallback_model in fallback_models:
            if fallback_model != model_name:
                try:
                    print(f"Trying fallback model: {fallback_model}")
                    model = SentenceTransformer(fallback_model)
                    print(f"Successfully loaded fallback model: {fallback_model}")
                    return model
                except Exception as fallback_e:
                    print(f"Fallback model '{fallback_model}' also failed: {str(fallback_e)}")
                    continue
        
        # If all models fail
        error_msg = f"Failed to load embedding model. Please check your internet connection and try again."
        print(f"Error: {error_msg}")
        print("Available models to try: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2")
        sys.exit(1)

def parse_list_string(list_str):
    """Parse string representation of lists like [x, y, w, h]"""
    try:
        return ast.literal_eval(list_str)
    except (SyntaxError, ValueError):
        # Alternative parsing using regex
        matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", list_str)
        if len(matches) >= 4:
            return [float(matches[i]) for i in range(4)]
        return [0, 0, 0, 0]  # Default values if parsing fails

def parse_tuple_string(tuple_str):
    """Parse string representation of tuples like (r, g, b)"""
    try:
        return list(ast.literal_eval(tuple_str))
    except (SyntaxError, ValueError):
        # Alternative parsing using regex
        matches = re.findall(r"\d+", tuple_str)
        if len(matches) >= 3:
            return [int(matches[i]) for i in range(3)]
        return [0, 0, 0]  # Default values if parsing fails

def parse_csv(csv_path):
    """Read the CSV file and organize data by frame"""
    df = pd.read_csv(csv_path)
    
    # Group by image name (frame)
    frames = {}
    for img_name, group in df.groupby('Image Name'):
        elements = []
        for _, row in group.iterrows():
            try:
                bounding_box = parse_list_string(row['Bounding Box'])
                normalized_bbox = parse_list_string(row['Normalized Bounding Box'])
                dominant_color = parse_tuple_string(row['Dominant Color'])
            except Exception as e:
                print(f"Error parsing row: {e}")
                bounding_box = [0, 0, 0, 0]
                normalized_bbox = [0, 0, 0, 0]
                dominant_color = [0, 0, 0]
                
            # Handle potential missing values
            ocr_confidence = float(row['OCR Confidence']) if pd.notna(row['OCR Confidence']) else 0.0
            iou_with_previous = float(row['IOU with Previous']) if pd.notna(row['IOU with Previous']) else 0.0
            content = str(row['Content']) if pd.notna(row['Content']) else ""
            
            # Create element dictionary
            element = {
                "element_id": row['Element ID'],
                "type": row['Type'],
                "bounding_box": bounding_box,
                "normalized_bbox": normalized_bbox,
                "interactivity": bool(row['Interactivity']),
                "interaction_type": row['Interaction Type'],
                "content": content,
                "ocr_confidence": ocr_confidence,
                "iou_with_previous": iou_with_previous,
                "dominant_color": dominant_color
            }
            elements.append(element)
        frames[img_name] = elements
    return frames

def element_to_json_string(element):
    """Convert element dictionary to a standardized JSON string"""
    x, y, w, h = element['normalized_bbox']
    r, g, b = element['dominant_color']
    
    standard_element = {
        "id": element['element_id'],
        "type": element['type'],
        "position": {
            "x": x,
            "y": y,
            "width": w,
            "height": h
        },
        "interactivity": {
            "is_interactive": element['interactivity'],
            "interaction_type": element["interaction_type"]
        },
        "appearance": {
            "dominant_color": [r, g, b]
        },
        "content": element['content']
    }
    
    return json.dumps(standard_element)

def process_csv(csv_path, model_name, output_dir=None):
    """
    Process CSV file and generate embeddings for each element.
    
    Args:
        csv_path (str): Path to CSV file
        model_name (str): SentenceTransformer model name
        output_dir (str): Output directory (default: auto-generated)
    
    Returns:
        dict: Processing results with embeddings info
    """
    # Validate inputs
    csv_path = validate_csv_path(csv_path)
    output_dir = validate_output_path(output_dir, csv_path)
    
    print(f"Processing CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize the model with fallback handling
    model = load_model_with_fallback(model_name)
    
    # Parse the CSV file
    frames = parse_csv(csv_path)
    
    # Sort frame names
    frame_names = sorted(frames.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"Processing {len(frame_names)} frames...")
    
    # Create embeddings for each frame
    frame_embeddings = []
    processed_count = 0
    
    for i, frame_name in enumerate(frame_names):
        elements = frames[frame_name]
        
        # Skip if no elements in frame
        if not elements:
            print(f"Warning: No elements found in frame {frame_name}")
            frame_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue
        
        # Convert each element to JSON string
        element_jsons = [element_to_json_string(element) for element in elements]
        
        # Create embeddings for each element
        element_embeddings = [model.encode(json_str) for json_str in element_jsons]
        
        # Use simple mean pooling for frame embedding
        frame_embedding = np.mean(element_embeddings, axis=0)
        frame_embeddings.append(frame_embedding)
        
        processed_count += 1
        if processed_count % 10 == 0:  # Progress update every 10 frames
            print(f"Processed {processed_count}/{len(frame_names)} frames...")
    
    embeddings_array = np.array(frame_embeddings)
    print(f"Created embeddings with shape: {embeddings_array.shape}")
    
    # Save embeddings
    embeddings_file = os.path.join(output_dir, "frame_embeddings.npz")
    save_embeddings(embeddings_array, frame_names, embeddings_file, model)
    
    # Create visualization
    viz_file = os.path.join(output_dir, "embeddings_viz.png")
    visualize_embeddings(embeddings_array, frame_names, viz_file)
    
    print(f"Processing complete! Results saved to: {output_dir}")
    
    return {
        "status": "success",
        "embeddings_file": embeddings_file,
        "visualization_file": viz_file,
        "output_directory": output_dir,
        "input_csv": csv_path,
        "model_used": model_name,
        "embedding_shape": embeddings_array.shape,
        "num_frames": len(frame_names)
    }

def save_embeddings(embeddings, frame_names, output_path, model=None):
    """Save embeddings and frame names to file"""
    np.savez(
        output_path,
        embeddings=embeddings,
        frame_names=frame_names
    )
    print(f"Saved embeddings to {output_path}")
        
    # Saving metadata
    metadata = {
        "embedding_model": model.__class__.__name__ if model else "SentenceTransformer",
        "model_name": model._modules['0'].auto_model.config._name_or_path if model else "all-MiniLM-L6-v2",
        "embedding_dimension": embeddings.shape[1],
        "frame_embedding_dimension": embeddings.shape[1],
        "num_frames": len(frame_names),
        "frame_names": frame_names.tolist() if isinstance(frame_names, np.ndarray) else frame_names
    }
    
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f: 
        json.dump(metadata, f, indent=2)
        
    print(f"Saved metadata to {metadata_path}")

def visualize_embeddings(embeddings, frame_names, output_path=None):
    """Create a PCA visualization of the embeddings"""
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=range(len(frame_names)),
        cmap='viridis',
        alpha=0.8,
        s=100
    )
    
    # Add frame indices for reference
    for i, name in enumerate(frame_names):
        frame_index = name.split('_')[1].split('.')[0]
        plt.annotate(frame_index, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9)
        
    # Add a colorbar to show the sequence
    plt.colorbar(scatter, label='Frame Sequence')
        
    plt.title("PCA Visualization of Frame Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
        
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate embeddings for UI elements from CSV data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'csv_path',
        nargs='?',
        help='Path to the CSV file containing parsed UI elements'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for embeddings (default: auto-generated next to CSV)'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model to use for embeddings'
    )
    
    args = parser.parse_args()
    
    # If no CSV path provided, prompt for it
    csv_path = args.csv_path
    if not csv_path:
        csv_path = input("Enter CSV file path: ").strip()
    
    # Process CSV and generate embeddings
    result = process_csv(csv_path, args.model, args.output)
    
    if result["status"] == "error":
        sys.exit(1)
    
    return result

if __name__ == "__main__":
    main()
    
# This code can be run as a script directly or with CLI arguments.
# Interactive mode (It will prompt for CSV path if not provided): python src/naive/naive_json_embedding_headless.py
# CLI mode (With arguments): python src/naive/naive_json_embedding_headless.py path/to/csv.csv -o output_dir -m model_name
