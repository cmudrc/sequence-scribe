import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_gnn_embeddings(npz_file: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    
    try:
        # Load the NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract embeddings and image names
        embeddings_dict = {}
        for key in data.files:
            embeddings_dict[key] = data[key]
        
        # Load metadata file to get proper ordering of images
        metadata_file = os.path.join(os.path.dirname(npz_file), "embeddings_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            image_names = metadata.get("image_names", list(embeddings_dict.keys()))
        else:
            # If metadata doesn't exist, use dictionary keys
            image_names = list(embeddings_dict.keys())
            
        # Ensure image_names are strings
        image_names = [str(name) for name in image_names]
        
        print(f"Loaded {len(embeddings_dict)} embeddings with dimension {next(iter(embeddings_dict.values())).shape}")
        print(f"Image names: {image_names[:5]}... (total: {len(image_names)})")
        
        return embeddings_dict, image_names
    
    except Exception as e:
        print(f"Error loading GNN embeddings from {npz_file}: {e}")
        raise

def create_sequential_embeddings(
    embeddings_dict: Dict[str, np.ndarray], 
    image_names: List[str]
) -> np.ndarray:
    
    # Gather embeddings in the correct sequence
    embeddings_list = []
    
    for img_name in image_names:
        if img_name in embeddings_dict:
            embeddings_list.append(embeddings_dict[img_name])
        else:
            print(f"Warning: No embedding found for {img_name}")
    
    # Convert to numpy array
    return np.array(embeddings_list)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    
    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms

def detect_outliers(embeddings: np.ndarray, z_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    
    # Calculate Z-scores for each dimension
    z_scores = np.abs(stats.zscore(embeddings, axis=0))
    
    # Find outliers across any dimension
    outlier_mask = np.any(z_scores > z_threshold, axis=1)
    outlier_indices = np.where(outlier_mask)[0]
    
    # Create cleaned embeddings (with outliers set to mean values)
    cleaned_embeddings = embeddings.copy()
    if len(outlier_indices) > 0:
        # Replace outliers with mean values
        dimension_means = np.mean(embeddings[~outlier_mask], axis=0)
        cleaned_embeddings[outlier_mask] = dimension_means
    
    return outlier_indices, cleaned_embeddings

def plot_embedding_distributions(
    embeddings: np.ndarray, 
    output_path: Optional[str] = None,
    n_dims: int = 5,  # Plot first n dimensions
    title: str = "Embedding Distributions"
) -> None:
    
    # Select a subset of dimensions to plot
    n_dims = min(n_dims, embeddings.shape[1])
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))
    
    # Handle single dimension case
    if n_dims == 1:
        axes = [axes]
    
    for i in range(n_dims):
        sns.histplot(embeddings[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding distributions plot to {output_path}")
    else:
        plt.show()
    plt.close()

def reduce_dimensions(
    train_embeddings: np.ndarray, 
    test_embeddings: np.ndarray, 
    n_components: int = 20,
    apply_standard_scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    
    # Ensure n_components is not larger than the input dimension
    n_components = min(n_components, train_embeddings.shape[1])
    
    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    pca.fit(train_embeddings)
    
    # Transform both training and test data
    train_reduced = pca.transform(train_embeddings)
    test_reduced = pca.transform(test_embeddings)
    
    # Apply standard scaling if requested
    scaler = None
    if apply_standard_scaling:
        scaler = StandardScaler()
        train_reduced = scaler.fit_transform(train_reduced)
        test_reduced = scaler.transform(test_reduced)
    
    # Calculate explained variance
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Reduced dimensions from {train_embeddings.shape[1]} to {n_components}")
    print(f"Explained variance: {explained_variance:.2f}%")
    
    return train_reduced, test_reduced, pca, scaler

def train_hmm(
    embeddings: np.ndarray, 
    n_states: int = 3, 
    covariance_type: str = "diag", 
    n_iter: int = 100, 
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[hmm.GaussianHMM, float]:
    
    # Create and train the model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit the model and get log likelihood
    model.fit(embeddings)
    log_likelihood = model.score(embeddings)
    
    return model, log_likelihood

def evaluate_hmm(
    model: hmm.GaussianHMM, 
    embeddings: np.ndarray, 
    frame_names: List[str]
) -> Dict[str, Any]:
    
    # Get the most likely state sequence
    hidden_states = model.predict(embeddings)
    
    # Get log likelihood
    log_likelihood = model.score(embeddings)
    
    # Get transition matrix
    transition_matrix = model.transmat_
    
    # Get state means
    state_means = model.means_
    
    # Create frame to state mapping
    frame_to_state = {frame: state for frame, state in zip(frame_names, hidden_states)}
    
    # Calculate average time spent in each state
    state_counts = np.bincount(hidden_states, minlength=model.n_components)
    state_proportions = state_counts / len(hidden_states)
    
    # Calculate state transition frequencies
    state_transitions = []
    for i in range(len(hidden_states) - 1):
        state_transitions.append((hidden_states[i], hidden_states[i+1]))
        
    transition_counts = {}
    for from_state, to_state in state_transitions:
        key = (from_state, to_state)
        transition_counts[key] = transition_counts.get(key, 0) + 1
    
    # Calculate state durations (how long each state lasts)
    state_durations = []
    current_state = hidden_states[0]
    current_duration = 1
    
    for i in range(1, len(hidden_states)):
        if hidden_states[i] == current_state:
            current_duration += 1
        else:
            state_durations.append((current_state, current_duration))
            current_state = hidden_states[i]
            current_duration = 1
            
    # Add the last sequence
    state_durations.append((current_state, current_duration))
    
    # Organize durations by state
    durations_by_state = {}
    for state, duration in state_durations:
        if state not in durations_by_state:
            durations_by_state[state] = []
        durations_by_state[state].append(duration)
    
    # Return evaluation metrics
    return {
        "log_likelihood": log_likelihood,
        "hidden_states": hidden_states,
        "transition_matrix": transition_matrix,
        "state_means": state_means,
        "frame_to_state": frame_to_state,
        "state_proportions": state_proportions,
        "transition_counts": transition_counts,
        "state_durations": durations_by_state
    }

def plot_transition_matrix(
    transition_matrix: np.ndarray, 
    output_path: Optional[str] = None,
    title: str = "HMM Transition Matrix"
) -> None:
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis', 
        xticklabels=range(transition_matrix.shape[1]),
        yticklabels=range(transition_matrix.shape[0])
    )
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition matrix plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_sequence(
    hidden_states: np.ndarray, 
    frame_names: List[str], 
    output_path: Optional[str] = None,
    title: str = "HMM State Sequence"
) -> None:
    
    plt.figure(figsize=(12, 6))
    
    # Extract frame indices for x-axis if frame names have format "frame_XXXX.jpg"
    try:
        x = [int(name.split('_')[1].split('.')[0]) for name in frame_names]
    except (IndexError, ValueError):
        x = range(len(hidden_states))
    
    plt.plot(x, hidden_states, 'o-', markersize=8)
    
    # Show a reasonable number of x-ticks
    if len(x) > 10:
        step = max(1, len(x) // 10)
        plt.xticks(x[::step], rotation=45, ha='right')
    else:
        plt.xticks(x, rotation=45, ha='right')
        
    plt.yticks(range(max(hidden_states) + 1))
    plt.grid(alpha=0.3)
    plt.xlabel("Frame")
    plt.ylabel("State")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state sequence plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_means(
    state_means: np.ndarray, 
    output_path: Optional[str] = None,
    title: str = "HMM State Means"
) -> None:
    
    # If there are too many dimensions, use PCA to reduce to 2D
    if state_means.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_means = pca.fit_transform(state_means)
    else:
        reduced_means = state_means
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_means[:, 0], 
        reduced_means[:, 1], 
        c=range(len(state_means)), 
        cmap='viridis', 
        s=100, 
        alpha=0.8
    )
    
    # Add state labels
    for i, (x, y) in enumerate(reduced_means):
        plt.annotate(f"State {i}", (x, y), fontsize=12, ha='center')
    
    plt.grid(alpha=0.3)
    plt.title(title)
    
    if state_means.shape[1] > 2:
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
    else:
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state means plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_durations(
    state_durations: Dict[int, List[int]],
    output_path: Optional[str] = None,
    title: str = "State Durations"
) -> None:
    
    num_states = len(state_durations)
    fig, axes = plt.subplots(1, num_states, figsize=(num_states * 4, 5))
    
    # Handle single state case
    if num_states == 1:
        axes = [axes]
    
    for i, state in enumerate(sorted(state_durations.keys())):
        durations = state_durations[state]
        axes[i].hist(durations, bins=min(20, max(5, len(durations) // 3)), alpha=0.7)
        axes[i].set_title(f"State {state}")
        axes[i].set_xlabel("Duration (frames)")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.05)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state durations plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_pca_variance(
    pca: PCA,
    output_path: Optional[str] = None,
    title: str = "PCA Explained Variance"
) -> None:
    
    plt.figure(figsize=(10, 6))
    
    # Plot explained variance ratio
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-', markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
    
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA variance plot to {output_path}")
    else:
        plt.show()
    plt.close()

def save_evaluation_results(
    evaluation: Dict[str, Any], 
    output_path: str, 
    embedding_info: Dict[str, Any]
) -> None:
    
    # Convert numpy arrays to lists for JSON serialization
    results = {
        "embedding_info": embedding_info,
        "log_likelihood": float(evaluation["log_likelihood"]),
        "state_proportions": evaluation["state_proportions"].tolist(),
        "transition_matrix": evaluation["transition_matrix"].tolist(),
        "num_frames": len(evaluation["hidden_states"]),
        "hidden_states": evaluation["hidden_states"].tolist()
    }
    
    # Add frame to state mapping
    frame_states = {}
    for frame, state in evaluation["frame_to_state"].items():
        frame_states[str(frame)] = int(state)
    results["frame_to_state"] = frame_states
    
    # Format transition counts for readability
    transition_counts = {}
    for (from_state, to_state), count in evaluation["transition_counts"].items():
        transition_counts[f"{from_state}->{to_state}"] = count
    results["transition_counts"] = transition_counts
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {output_path}")

def process_gnn_embeddings(
    embedding_file: str,
    output_dir: str,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    train_ratio: float = 2/3,
    pca_components: int = 20,
    normalize: bool = True,
    apply_standard_scaling: bool = True,
    detect_and_handle_outliers: bool = True,
    z_threshold: float = 3.0
) -> Dict[str, Any]:
    
    print(f"\nProcessing {embedding_file}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load GNN embeddings
    embeddings_dict, image_names = load_gnn_embeddings(embedding_file)
    
    # Create sequential embeddings array
    embeddings = create_sequential_embeddings(embeddings_dict, image_names)
    print(f"Created sequential embeddings with shape {embeddings.shape}")
    
    # Plot original embedding distributions
    orig_dist_plot = os.path.join(output_dir, "original_embedding_distributions.png")
    plot_embedding_distributions(embeddings, orig_dist_plot, title="Original GNN Embedding Distributions")
    
    # Normalization step (if enabled)
    if normalize:
        print("Normalizing embeddings to unit length...")
        embeddings = normalize_embeddings(embeddings)
        
        # Plot normalized embedding distributions
        norm_dist_plot = os.path.join(output_dir, "normalized_embedding_distributions.png")
        plot_embedding_distributions(embeddings, norm_dist_plot, title="Normalized GNN Embedding Distributions")
    
    # Outlier detection (if enabled)
    if detect_and_handle_outliers:
        print(f"Detecting outliers using Z-score threshold of {z_threshold}...")
        outlier_indices, cleaned_embeddings = detect_outliers(embeddings, z_threshold)
        
        if len(outlier_indices) > 0:
            print(f"Detected {len(outlier_indices)} outliers: {outlier_indices}")
            print("Replacing outliers with dimension means...")
            embeddings = cleaned_embeddings
            
            # Plot cleaned embedding distributions
            cleaned_dist_plot = os.path.join(output_dir, "cleaned_embedding_distributions.png")
            plot_embedding_distributions(embeddings, cleaned_dist_plot, title="Cleaned GNN Embedding Distributions")
        else:
            print("No outliers detected.")
    
    # Split data sequentially
    split_index = int(len(embeddings) * train_ratio)
    train_embeddings = embeddings[:split_index]
    train_frames = image_names[:split_index]
    test_embeddings = embeddings[split_index:]
    test_frames = image_names[split_index:]
    
    print(f"Split data: {len(train_embeddings)} training frames, {len(test_embeddings)} testing frames")
    
    # Apply dimensionality reduction
    print(f"Applying PCA dimensionality reduction to {pca_components} components...")
    train_reduced, test_reduced, pca_model, scaler = reduce_dimensions(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        n_components=pca_components,
        apply_standard_scaling=apply_standard_scaling
    )
    
    # Plot PCA explained variance
    pca_variance_plot = os.path.join(output_dir, "pca_explained_variance.png")
    plot_pca_variance(pca_model, pca_variance_plot)
    
    # Plot reduced embedding distributions
    reduced_dist_plot = os.path.join(output_dir, "reduced_embedding_distributions.png")
    plot_embedding_distributions(
        train_reduced, reduced_dist_plot, 
        n_dims=min(5, train_reduced.shape[1]),
        title="Reduced GNN Embedding Distributions" + (" (with Standard Scaling)" if apply_standard_scaling else "")
    )
    
    # Train HMM on dimensionality-reduced training data
    print(f"Training HMM with {n_states} states...")
    model, train_log_likelihood = train_hmm(
        embeddings=train_reduced,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter
    )
    print(f"Training complete. Train log likelihood: {train_log_likelihood:.2f}")
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_evaluation = evaluate_hmm(model, train_reduced, train_frames)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_evaluation = evaluate_hmm(model, test_reduced, test_frames)
    test_log_likelihood = test_evaluation["log_likelihood"]
    print(f"Test log likelihood: {test_log_likelihood:.2f}")
    
    # Collect embedding info
    embedding_info = {
        "file": embedding_file,
        "type": "GNN Embeddings",
        "n_states": n_states,
        "covariance_type": covariance_type,
        "original_embedding_shape": embeddings.shape,
        "reduced_embedding_shape": train_reduced.shape,
        "pca_components": pca_components,
        "pca_explained_variance": float(sum(pca_model.explained_variance_ratio_) * 100),
        "normalization_applied": normalize,
        "standard_scaling_applied": apply_standard_scaling,
        "outlier_detection_applied": detect_and_handle_outliers,
        "outlier_z_threshold": z_threshold if detect_and_handle_outliers else None,
        "outliers_detected": len(outlier_indices) if detect_and_handle_outliers else None,
        "train_ratio": train_ratio,
        "train_frames": len(train_frames),
        "test_frames": len(test_frames)
    }
    
    # Generate and save plots for the full model
    print("Generating plots...")
    
    # Transition matrix plot
    transition_plot_file = os.path.join(output_dir, f"transition_matrix.png")
    plot_transition_matrix(
        model.transmat_, 
        transition_plot_file,
        title=f"HMM Transition Matrix (n_states={n_states})"
    )
    
    # Markov chain graph plot
    markov_graph_file = os.path.join(output_dir, f"markov_chain_graph.png")
    plot_hmm_graph(
        model.transmat_,
        markov_graph_file,
        title=f"HMM Markov Chain (n_states={n_states})"
    )
    
    # State means plot
    means_plot_file = os.path.join(output_dir, f"state_means.png")
    plot_state_means(
        model.means_, 
        means_plot_file,
        title=f"HMM State Means (n_states={n_states})"
    )
    
    # State sequence plot for training data
    train_sequence_plot_file = os.path.join(output_dir, f"train_state_sequence.png")
    plot_state_sequence(
        train_evaluation["hidden_states"], 
        train_frames, 
        train_sequence_plot_file,
        title=f"HMM State Sequence - Training Data (n_states={n_states})"
    )
    
    # State sequence plot for test data
    test_sequence_plot_file = os.path.join(output_dir, f"test_state_sequence.png")
    plot_state_sequence(
        test_evaluation["hidden_states"], 
        test_frames, 
        test_sequence_plot_file,
        title=f"HMM State Sequence - Test Data (n_states={n_states})"
    )
    
    # Plot state durations
    durations_plot_file = os.path.join(output_dir, f"state_durations.png")
    plot_state_durations(
        train_evaluation["state_durations"],
        durations_plot_file,
        title=f"State Durations (n_states={n_states})"
    )
    
    # Save evaluation results
    train_results_file = os.path.join(output_dir, "train_evaluation_results.json")
    save_evaluation_results(train_evaluation, train_results_file, {**embedding_info, "dataset": "train"})
    
    test_results_file = os.path.join(output_dir, "test_evaluation_results.json")
    save_evaluation_results(test_evaluation, test_results_file, {**embedding_info, "dataset": "test"})
    
    # Create comparison summary
    summary = {
        "embedding_type": "GNN Embeddings",
        "n_states": n_states,
        "covariance_type": covariance_type,
        "pca_components": pca_components,
        "pca_explained_variance": float(sum(pca_model.explained_variance_ratio_) * 100),
        "normalization_applied": normalize,
        "standard_scaling_applied": apply_standard_scaling,
        "outlier_detection_applied": detect_and_handle_outliers,
        "train_log_likelihood": float(train_log_likelihood),
        "test_log_likelihood": float(test_log_likelihood),
        "train_frames": len(train_frames),
        "test_frames": len(test_frames)
    }
    
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing complete for {embedding_file}")
    
    return {
        "file": embedding_file,
        "train_log_likelihood": train_log_likelihood,
        "test_log_likelihood": test_log_likelihood,
        "n_states": n_states,
        "pca_components": pca_components
    }

def compare_state_sequences(
    text_embedding_file: Optional[str] = None,
    gnn_embedding_file: Optional[str] = None,
    output_dir: str = "comparison_results",
    n_states: int = 3,
    pca_components: int = 20
) -> None:
    
    if text_embedding_file is None and gnn_embedding_file is None:
        print("Error: At least one embedding file must be provided")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    text_states = None
    gnn_states = None
    
    # Process text embeddings if provided
    if text_embedding_file and os.path.exists(text_embedding_file):
        print(f"Processing text embeddings from {text_embedding_file}...")
        text_output_dir = os.path.join(output_dir, "text_embeddings")
        os.makedirs(text_output_dir, exist_ok=True)
        
        # Use the existing function from naive_hmm.py to load text embeddings
        try:
            # Load text embeddings
            text_data = np.load(text_embedding_file)
            if 'embeddings' in text_data:
                text_embeddings = text_data['embeddings']
                text_frame_names = text_data['frame_names'].tolist()
            elif 'observations' in text_data:
                text_embeddings = text_data['observations']
                text_frame_names = text_data['frame_names'].tolist()
            else:
                # Try to infer format by checking all keys
                print("Warning: Unknown text embedding format. Trying to infer...")
                if len(text_data.files) > 0:
                    # Use the first key as embeddings
                    key = text_data.files[0]
                    text_embeddings = text_data[key]
                    text_frame_names = [f"frame_{i}" for i in range(len(text_embeddings))]
            
            # Process text embeddings with PCA
            # Apply PCA
            if text_embeddings.shape[1] > pca_components:
                pca = PCA(n_components=pca_components)
                text_reduced = pca.fit_transform(text_embeddings)
            else:
                text_reduced = text_embeddings
                
            # Train HMM on text embeddings
            text_model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            text_model.fit(text_reduced)
            
            # Get the state sequence
            text_states = text_model.predict(text_reduced)
            
        except Exception as e:
            print(f"Error processing text embeddings: {e}")
            text_states = None
    
    # Process GNN embeddings if provided
    if gnn_embedding_file and os.path.exists(gnn_embedding_file):
        print(f"Processing GNN embeddings from {gnn_embedding_file}...")
        gnn_output_dir = os.path.join(output_dir, "gnn_embeddings")
        os.makedirs(gnn_output_dir, exist_ok=True)
        
        try:
            # Load GNN embeddings
            gnn_dict, gnn_frame_names = load_gnn_embeddings(gnn_embedding_file)
            gnn_embeddings = create_sequential_embeddings(gnn_dict, gnn_frame_names)
            
            # Apply PCA
            if gnn_embeddings.shape[1] > pca_components:
                pca = PCA(n_components=pca_components)
                gnn_reduced = pca.fit_transform(gnn_embeddings)
            else:
                gnn_reduced = gnn_embeddings
                
            # Train HMM on GNN embeddings
            gnn_model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            gnn_model.fit(gnn_reduced)
            
            # Get the state sequence
            gnn_states = gnn_model.predict(gnn_reduced)
            
        except Exception as e:
            print(f"Error processing GNN embeddings: {e}")
            gnn_states = None
    
    # Compare state sequences if both are available
    if text_states is not None and gnn_states is not None:
        # Make sure they have the same length for comparison
        min_length = min(len(text_states), len(gnn_states))
        text_states = text_states[:min_length]
        gnn_states = gnn_states[:min_length]
        
        # Calculate state matching
        # This is a simplified approach - in practice, you might want to use
        # more sophisticated methods like adjusted Rand index or normalized mutual information
        matching = np.zeros((n_states, n_states), dtype=int)
        for i in range(min_length):
            matching[text_states[i], gnn_states[i]] += 1
            
        # Plot state matching matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matching, 
            annot=True, 
            fmt='d', 
            cmap='viridis', 
            xticklabels=[f"GNN State {i}" for i in range(n_states)],
            yticklabels=[f"Text State {i}" for i in range(n_states)]
        )
        plt.title("State Matching Matrix: Text Embeddings vs. GNN Embeddings")
        plt.tight_layout()
        
        matching_plot_file = os.path.join(output_dir, "state_matching_matrix.png")
        plt.savefig(matching_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot state sequences together
        plt.figure(figsize=(15, 8))
        
        try:
            x = [int(name.split('_')[1].split('.')[0]) for name in gnn_frame_names[:min_length]]
        except (IndexError, ValueError):
            x = range(min_length)
        
        plt.plot(x, text_states, 'o-', label='Text Embeddings States', alpha=0.7, markersize=8)
        plt.plot(x, gnn_states, 's-', label='GNN Embeddings States', alpha=0.7, markersize=8)
        
        if len(x) > 10:
            step = max(1, len(x) // 10)
            plt.xticks(x[::step], rotation=45, ha='right')
        else:
            plt.xticks(x, rotation=45, ha='right')
            
        plt.yticks(range(n_states))
        plt.grid(alpha=0.3)
        plt.xlabel("Frame")
        plt.ylabel("State")
        plt.title(f"HMM State Sequence Comparison (n_states={n_states})")
        plt.legend()
        
        sequence_plot_file = os.path.join(output_dir, "state_sequence_comparison.png")
        plt.savefig(sequence_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate agreement percentage
        agreement = np.sum(text_states == gnn_states) / min_length * 100
        print(f"State agreement: {agreement:.2f}%")
        
        # Save comparison results
        comparison_results = {
            "text_embedding_file": text_embedding_file,
            "gnn_embedding_file": gnn_embedding_file,
            "n_states": n_states,
            "pca_components": pca_components,
            "sequence_length": min_length,
            "state_agreement_percentage": float(agreement),
            "state_matching_matrix": matching.tolist()
        }
        
        comparison_file = os.path.join(output_dir, "comparison_results.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to {output_dir}")
    else:
        print("Could not compare state sequences: one or both state sequences are unavailable")

def plot_hmm_graph(
    transition_matrix: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "HMM Markov Chain",
    min_prob: float = 0.01,  # Minimum probability to show an edge
    node_size: int = 2000,
    arrow_size: int = 20,
    layout: str = "spring"  # Options: spring, circular, spectral
) -> None:
    """
    Plot a directed graph visualization of the HMM transition matrix.
    
    Args:
        transition_matrix: The transition matrix of the HMM
        output_path: Path to save the plot
        title: Title of the plot
        min_prob: Minimum probability to display an edge
        node_size: Size of nodes in the graph
        arrow_size: Size of arrows in the graph
        layout: Graph layout algorithm
    """
    import networkx as nx
    import matplotlib.patches as patches
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (states)
    n_states = transition_matrix.shape[0]
    for i in range(n_states):
        G.add_node(i, label=f"State {i}")
    
    # Add edges with weights (transition probabilities)
    for i in range(n_states):
        for j in range(n_states):
            prob = transition_matrix[i, j]
            if prob > min_prob:  # Only show significant transitions
                G.add_edge(i, j, weight=prob, label=f"{prob:.2f}")
    
    # Create figure with specific axes layout
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position nodes
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:  # Default to spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)  # Added seed for reproducibility
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                          node_color="#8ecae6", alpha=1.0, ax=ax)  # Changed to more vibrant node color
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight="bold", ax=ax)  # Increased font size
    
    # Define edge colors and widths based on weight
    edge_colors = [G[u][v]['weight'] for u, v in G.edges() if u != v]  # Exclude self-loops
    # Make edges thicker for better visibility
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges() if u != v]  # Exclude self-loops
    
    # Create edge list excluding self-loops
    edges_no_selfloops = [(u, v) for u, v in G.edges() if u != v]
    
    # Use a more vibrant colormap
    cmap = plt.cm.YlOrRd  # Changed to YlOrRd which has more contrast
    vmin = min(edge_colors) if edge_colors else 0
    vmax = max(edge_colors) if edge_colors else 1
    
    # Draw edges with increased contrast and visibility - only non-self-loops
    if edges_no_selfloops:
        edges = nx.draw_networkx_edges(G, pos, edgelist=edges_no_selfloops, width=edge_widths, 
                                      edge_color=edge_colors, edge_cmap=cmap,
                                      edge_vmin=vmin, edge_vmax=vmax,
                                      arrowsize=arrow_size+5,  # Increased arrow size
                                      connectionstyle='arc3,rad=0.2',  # More curved edges
                                      arrowstyle='-|>',
                                      ax=ax)
    
    # Add edge labels (probabilities) with a background for better readability
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() if u != v}  # Skip self-loops here
    # Add white background to edge labels
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, 
                                bbox=bbox_props, font_weight='bold', ax=ax)
    
    # Add colorbar with improved visibility
    if edge_colors:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition Probability", fontsize=14, fontweight='bold')  # Increased font size
        cbar.ax.tick_params(labelsize=12)  # Larger tick labels
    
    # Self-loops (transitions to same state) with improved styling and a single text label
    for i in range(n_states):
        if G.has_edge(i, i):
            self_loop_weight = G[i][i]['weight']
            rad = 0.3
            
            # Create a looped arrow with more prominent color
            color_val = cmap(self_loop_weight)
            
            arrow = patches.FancyArrowPatch(
                pos[i], pos[i],
                connectionstyle=f'arc3,rad={rad}',
                arrowstyle='-|>',
                mutation_scale=25,  # Increased size
                lw=self_loop_weight*12,  # Thicker line
                color=color_val,
                zorder=0
            )
            ax.add_patch(arrow)
            
            # Add self-loop label with white background for better visibility
            # Position the label at the top of the loop
            label_pos = (pos[i][0], pos[i][1] + rad + 0.05)
            
            # Create white background for self-loop label
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            ax.text(label_pos[0], label_pos[1], f"{self_loop_weight:.2f}", 
                   fontsize=11, ha='center', va='center', 
                   bbox=bbox_props, fontweight='bold')
    
    ax.set_title(title, fontsize=18, fontweight='bold')  # Increased font size
    ax.axis('off')
    
    # Add margin to avoid cutting off node contents
    plt.tight_layout(pad=1.2)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved HMM graph visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_hmm_results(
    hmm_dir: str,
    embed_type: str = "GNN",
    title_prefix: str = "GNN Embedding"
) -> None:
    
    try:
        # Load evaluation summary
        summary_file = os.path.join(hmm_dir, "evaluation_summary.json")
        if not os.path.exists(summary_file):
            print(f"Error: Summary file not found at {summary_file}")
            return
            
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            
        # Load train and test results
        train_file = os.path.join(hmm_dir, "train_evaluation_results.json")
        test_file = os.path.join(hmm_dir, "test_evaluation_results.json")
        
        with open(train_file, 'r') as f:
            train_results = json.load(f)
            
        with open(test_file, 'r') as f:
            test_results = json.load(f)
        
        # Extract key metrics
        n_states = summary["n_states"]
        train_ll = summary["train_log_likelihood"]
        test_ll = summary["test_log_likelihood"]
        pca_variance = summary.get("pca_explained_variance", 0)
        
        # Summary plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(['Train', 'Test'], [train_ll, test_ll], color=['blue', 'orange'])
        plt.title(f"{title_prefix} Log Likelihood")
        plt.ylabel("Log Likelihood")
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        state_props = np.array(train_results["state_proportions"])
        plt.pie(
            state_props, 
            labels=[f"State {i}\n({prop:.1%})" for i, prop in enumerate(state_props)],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        plt.title(f"{title_prefix} State Distribution")
        
        plt.tight_layout()
        summary_plot_file = os.path.join(hmm_dir, f"{embed_type.lower()}_hmm_summary.png")
        plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated summary plot for {embed_type} HMM results")
        
    except Exception as e:
        print(f"Error plotting HMM results: {e}")

def main():
    
    parser = argparse.ArgumentParser(description="Train and evaluate HMM models on GNN-based UI embeddings")
    parser.add_argument("--input_dir", type=str, default="ui_embeddings",
                        help="Directory containing embedding NPZ files")
    parser.add_argument("--output_dir", type=str, default="gnn_hmm_results",
                        help="Directory to save HMM results")
    parser.add_argument("--embeddings_file", type=str, default="ui_embeddings.npz",
                        help="Name of the embeddings NPZ file")
    parser.add_argument("--n_states", type=int, default=3,
                        help="Number of hidden states for the HMM")
    parser.add_argument("--covariance_type", type=str, default="diag",
                        choices=["spherical", "tied", "diag", "full"],
                        help="Type of covariance matrix for the HMM")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="Maximum number of iterations for EM algorithm")
    parser.add_argument("--train_ratio", type=float, default=2/3,
                        help="Ratio of data to use for training (default: 2/3)")
    parser.add_argument("--pca_components", type=int, default=20,
                        help="Number of PCA components to use (default: 20)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable normalization of embeddings")
    parser.add_argument("--no_standard_scaling", action="store_true",
                        help="Disable standard scaling after PCA")
    parser.add_argument("--no_outlier_detection", action="store_true",
                        help="Disable outlier detection and handling")
    parser.add_argument("--z_threshold", type=float, default=3.0,
                        help="Z-score threshold for outlier detection (default: 3.0)")
    parser.add_argument("--compare_with_text", type=str, default=None,
                        help="Path to text embeddings NPZ file for comparison")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get embedding file path
    embedding_file = os.path.join(args.input_dir, args.embeddings_file)
    
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found at {embedding_file}")
        return
    
    print(f"Found embedding file: {embedding_file}")
    
    # Process with train-test split and preprocessing
    result = process_gnn_embeddings(
        embedding_file=embedding_file,
        output_dir=args.output_dir,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        train_ratio=args.train_ratio,
        pca_components=args.pca_components,
        normalize=not args.no_normalize,
        apply_standard_scaling=not args.no_standard_scaling,
        detect_and_handle_outliers=not args.no_outlier_detection,
        z_threshold=args.z_threshold
    )
    
    # Generate summary plots
    plot_hmm_results(args.output_dir, "GNN", "GNN Embedding")
    
    # Compare with text embeddings if provided
    if args.compare_with_text:
        print(f"\nComparing with text embeddings: {args.compare_with_text}")
        compare_output_dir = os.path.join(args.output_dir, "comparison_with_text")
        compare_state_sequences(
            text_embedding_file=args.compare_with_text,
            gnn_embedding_file=embedding_file,
            output_dir=compare_output_dir,
            n_states=args.n_states,
            pca_components=args.pca_components
        )
    
    print("\nGNN-HMM analysis complete!")
    print(f"All results saved to {args.output_dir}")
    print(f"Train log likelihood: {result['train_log_likelihood']:.2f}")
    print(f"Test log likelihood: {result['test_log_likelihood']:.2f}")
    print(f"PCA components used: {result['pca_components']}")

if __name__ == "__main__":
    main()


# Direct mode: python src/gnn/graph_hmm_headless.py
# CLI Mode: python src/gnn/graph_hmm_headless.py --input_dir path/to/embeddings --output_dir path/to/results --embeddings_file ui_embeddings.npz --covariance_type diag