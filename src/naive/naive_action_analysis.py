import os
import json
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.patches as patches
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Default paths
DEFAULT_EMBEDDINGS_PATH = "./embeddings.npz"
DEFAULT_OUTPUT_FOLDER = "./output"

def validate_embeddings_path(embeddings_path):
    """Validate embeddings file path and prompt for correction if needed."""
    while True:
        if not embeddings_path:
            embeddings_path = input("Please enter the embeddings file path (.npz): ").strip()
            continue
            
        if not os.path.exists(embeddings_path):
            print(f"Error: Embeddings file '{embeddings_path}' not found.")
            embeddings_path = input("Please enter a valid embeddings file path: ").strip()
            continue
            
        if not embeddings_path.lower().endswith('.npz'):
            print(f"Error: '{embeddings_path}' is not an NPZ file.")
            embeddings_path = input("Please enter a valid embeddings file path (.npz): ").strip()
            continue
            
        return embeddings_path

def validate_output_path(output_path, embeddings_path):
    """Validate output directory path and create if needed."""
    if not output_path:
        # Default: create output folder next to embeddings file
        embeddings_dir = os.path.dirname(embeddings_path)
        embeddings_name = os.path.splitext(os.path.basename(embeddings_path))[0]
        output_path = os.path.join(embeddings_dir, f"output_hmm_{embeddings_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    return output_path

def load_embeddings(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings from a .npz file."""
    try:
        data = np.load(file_path)
        if 'embeddings' in data:
            embeddings = data['embeddings']
            frame_names = data['frame_names']
        elif 'observations' in data:
            embeddings = data['observations']
            frame_names = data['frame_names']
        else:
            raise ValueError(f"Unknown data format in {file_path}")
        
        # Convert frame_names to list if it's a numpy array
        if isinstance(frame_names, np.ndarray):
            frame_names = [str(name) for name in frame_names]
            
        return embeddings, frame_names
    except Exception as e:
        print(f"Error loading embeddings from {file_path}: {e}")
        raise

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length (L2 norm)."""
    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms

def detect_outliers(embeddings: np.ndarray, z_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outliers using Z-score."""
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

def preprocess_embeddings(
    embeddings: np.ndarray,
    normalize: bool = True,
    detect_outliers_flag: bool = True,
    z_threshold: float = 3.0
) -> Tuple[np.ndarray, List[int]]:
    """Preprocess embeddings with normalization and outlier detection."""
    outlier_indices = []
    
    # Normalization
    if normalize:
        print("Normalizing embeddings to unit length...")
        embeddings = normalize_embeddings(embeddings)
    
    # Outlier detection
    if detect_outliers_flag:
        print(f"Detecting outliers using Z-score threshold of {z_threshold}...")
        outlier_indices, embeddings = detect_outliers(embeddings, z_threshold)
        
        if len(outlier_indices) > 0:
            print(f"Detected {len(outlier_indices)} outliers: {outlier_indices}")
            print("Replacing outliers with dimension means...")
        else:
            print("No outliers detected.")
    
    return embeddings, outlier_indices

def reduce_dimensions(
    train_embeddings: np.ndarray, 
    test_embeddings: np.ndarray, 
    n_components: int = 20,
    apply_standard_scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """Reduce dimensionality of embeddings using PCA with optional standard scaling."""
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

def train_hidden_markov_model(
    action_sequence: Union[str, np.ndarray],
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    train_ratio: float = 2/3,
    pca_components: int = 20,
    normalize: bool = True,
    apply_standard_scaling: bool = True,
    detect_outliers_flag: bool = True,
    z_threshold: float = 3.0,
    output_dir: str = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a comprehensive Hidden Markov Model with preprocessing and evaluation.
    
    Args:
        action_sequence: Path to embeddings file or embeddings array
        n_states: Number of HMM states
        covariance_type: HMM covariance type
        n_iter: Maximum iterations for training
        train_ratio: Training data ratio
        pca_components: Number of PCA components
        normalize: Whether to normalize embeddings
        apply_standard_scaling: Whether to apply standard scaling
        detect_outliers_flag: Whether to detect outliers
        z_threshold: Z-score threshold for outlier detection
        output_dir: Output directory for results
        random_state: Random state for reproducibility
        verbose: Whether to print verbose output
    
    Returns:
        Dict containing model and evaluation results
    """
    # Load embeddings if path is provided
    if isinstance(action_sequence, str):
        embeddings_path = validate_embeddings_path(action_sequence)
        embeddings, frame_names = load_embeddings(embeddings_path)
        if output_dir is None:
            output_dir = validate_output_path(None, embeddings_path)
    else:
        embeddings = action_sequence
        frame_names = [f"frame_{i:04d}" for i in range(len(embeddings))]
        if output_dir is None:
            output_dir = "./hmm_output"
            os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loaded embeddings with shape {embeddings.shape} for {len(frame_names)} frames")
    
    # Preprocess embeddings
    embeddings, outlier_indices = preprocess_embeddings(
        embeddings, normalize, detect_outliers_flag, z_threshold
    )
    
    # Split data sequentially
    split_index = int(len(embeddings) * train_ratio)
    train_embeddings = embeddings[:split_index]
    train_frames = frame_names[:split_index]
    test_embeddings = embeddings[split_index:]
    test_frames = frame_names[split_index:]
    
    print(f"Split data: {len(train_embeddings)} training frames, {len(test_embeddings)} testing frames")
    
    # Apply dimensionality reduction
    print(f"Applying PCA dimensionality reduction to {pca_components} components...")
    train_reduced, test_reduced, pca_model, scaler = reduce_dimensions(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        n_components=pca_components,
        apply_standard_scaling=apply_standard_scaling
    )
    
    # Train HMM
    print(f"Training HMM with {n_states} states...")
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    model.fit(train_reduced)
    train_log_likelihood = model.score(train_reduced)
    test_log_likelihood = model.score(test_reduced)
    
    print(f"Training complete. Train log likelihood: {train_log_likelihood:.2f}")
    print(f"Test log likelihood: {test_log_likelihood:.2f}")
    
    # Evaluate model
    train_evaluation = evaluate_hmm(model, train_reduced, train_frames)
    test_evaluation = evaluate_hmm(model, test_reduced, test_frames)
    
    # Save results
    results = {
        "model": model,
        "train_evaluation": train_evaluation,
        "test_evaluation": test_evaluation,
        "pca_model": pca_model,
        "scaler": scaler,
        "train_log_likelihood": train_log_likelihood,
        "test_log_likelihood": test_log_likelihood,
        "outlier_indices": outlier_indices,
        "embedding_info": {
            "n_states": n_states,
            "covariance_type": covariance_type,
            "original_shape": embeddings.shape,
            "reduced_shape": train_reduced.shape,
            "pca_components": pca_components,
            "normalize": normalize,
            "detect_outliers": detect_outliers_flag,
            "train_ratio": train_ratio
        }
    }
    
    return results

def train_markov_model(
    action_sequence: Union[str, np.ndarray],
    n_states: int = 3,
    covariance_type: str = "diag", 
    n_iter: int = 100,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a simplified Markov model (Gaussian HMM without comprehensive preprocessing).
    
    Args:
        action_sequence: Path to embeddings file or embeddings array
        n_states: Number of states
        covariance_type: Covariance type
        n_iter: Maximum iterations
        random_state: Random state
    
    Returns:
        Dict containing transition matrix and basic model info
    """
    # Load embeddings if path is provided
    if isinstance(action_sequence, str):
        embeddings_path = validate_embeddings_path(action_sequence)
        embeddings, frame_names = load_embeddings(embeddings_path)
    else:
        embeddings = action_sequence
        frame_names = [f"frame_{i:04d}" for i in range(len(embeddings))]
    
    print(f"Training simplified Markov model with {n_states} states...")
    
    # Create and train basic HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    
    model.fit(embeddings)
    log_likelihood = model.score(embeddings)
    
    # Get basic evaluation
    hidden_states = model.predict(embeddings)
    
    return {
        "transition_matrix": model.transmat_,
        "model": model,
        "hidden_states": hidden_states,
        "log_likelihood": log_likelihood,
        "state_means": model.means_,
        "frame_names": frame_names
    }

def evaluate_hmm(
    model: hmm.GaussianHMM, 
    embeddings: np.ndarray, 
    frame_names: List[str]
) -> Dict[str, Any]:
    """Evaluate a trained HMM model."""
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
    
    # Calculate state durations
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
            
    state_durations.append((current_state, current_duration))
    
    # Organize durations by state
    durations_by_state = {}
    for state, duration in state_durations:
        if state not in durations_by_state:
            durations_by_state[state] = []
        durations_by_state[state].append(duration)
    
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

def plot_categorical_model_distribution(
    token_frequencies: Union[Dict, np.ndarray], 
    action_names: List[str] = None,
    output_path: str = None,
    title: str = "State Frequency Distribution"
) -> matplotlib.figure.Figure:
    """Plot distribution of states/tokens."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(token_frequencies, dict):
        states = list(token_frequencies.keys())
        frequencies = list(token_frequencies.values())
    else:
        states = list(range(len(token_frequencies)))
        frequencies = token_frequencies
    
    if action_names is None:
        action_names = [f"State {i}" for i in states]
    
    bars = ax.bar(states, frequencies, alpha=0.8, color='skyblue')
    ax.set_xlabel("State")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xticks(states)
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{freq:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state distribution plot to {output_path}")
    
    return fig

def plot_markov_model_matrix(
    action_sequence: Union[np.ndarray, Dict], 
    action_names: List[str] = None,
    output_path: str = None,
    title: str = "Markov Model Transition Matrix"
) -> matplotlib.figure.Figure:
    """Plot transition matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if isinstance(action_sequence, dict) and 'transition_matrix' in action_sequence:
        transition_matrix = action_sequence['transition_matrix']
    else:
        transition_matrix = action_sequence
    
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis', 
        xticklabels=action_names or range(transition_matrix.shape[1]),
        yticklabels=action_names or range(transition_matrix.shape[0]),
        ax=ax
    )
    
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition matrix plot to {output_path}")
    
    return fig

def plot_hidden_markov_model_matrices(
    action_sequence: Dict, 
    action_names: List[str] = None,
    output_path: str = None,
    title: str = "Hidden Markov Model Analysis"
) -> matplotlib.figure.Figure:
    """Plot comprehensive HMM analysis including transition matrix, state sequence, and state means."""
    # Create subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data from results
    if 'model' in action_sequence:
        model = action_sequence['model']
        transition_matrix = model.transmat_
        state_means = model.means_
    else:
        transition_matrix = action_sequence.get('transition_matrix')
        state_means = action_sequence.get('state_means')
    
    evaluation = action_sequence.get('train_evaluation', action_sequence)
    hidden_states = evaluation.get('hidden_states', [])
    frame_names = action_sequence.get('frame_names', [])
    
    # Plot 1: Transition Matrix
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis',
        xticklabels=action_names or range(transition_matrix.shape[1]),
        yticklabels=action_names or range(transition_matrix.shape[0]),
        ax=ax1
    )
    ax1.set_title("Transition Matrix")
    ax1.set_xlabel("Next State")
    ax1.set_ylabel("Current State")
    
    # Plot 2: State Sequence
    ax2 = plt.subplot(2, 3, 2)
    if len(hidden_states) > 0:
        try:
            x = [int(name.split('_')[1].split('.')[0]) for name in frame_names]
        except (IndexError, ValueError):
            x = range(len(hidden_states))
        
        ax2.plot(x, hidden_states, 'o-', markersize=4)
        ax2.set_title("State Sequence")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("State")
        ax2.grid(alpha=0.3)
    
    # Plot 3: State Means (PCA if high-dimensional)
    ax3 = plt.subplot(2, 3, 3)
    if state_means is not None and len(state_means) > 0:
        if state_means.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_means = pca.fit_transform(state_means)
        else:
            reduced_means = state_means
        
        scatter = ax3.scatter(
            reduced_means[:, 0], 
            reduced_means[:, 1], 
            c=range(len(state_means)), 
            cmap='viridis', 
            s=100, 
            alpha=0.8
        )
        
        for i, (x, y) in enumerate(reduced_means):
            ax3.annotate(f"S{i}", (x, y), fontsize=10, ha='center')
        
        ax3.set_title("State Means")
        ax3.grid(alpha=0.3)
    
    # Plot 4: State Proportions
    ax4 = plt.subplot(2, 3, 4)
    if 'state_proportions' in evaluation:
        proportions = evaluation['state_proportions']
        states = range(len(proportions))
        bars = ax4.bar(states, proportions, alpha=0.8, color='lightcoral')
        ax4.set_title("State Proportions")
        ax4.set_xlabel("State")
        ax4.set_ylabel("Proportion")
        ax4.set_xticks(states)
        
        for bar, prop in zip(bars, proportions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prop:.2f}', ha='center', va='bottom')
    
    # Plot 5: State Durations
    ax5 = plt.subplot(2, 3, 5)
    if 'state_durations' in evaluation:
        durations = evaluation['state_durations']
        if durations:
            all_durations = []
            state_labels = []
            for state, durs in durations.items():
                all_durations.extend(durs)
                state_labels.extend([f"S{state}"] * len(durs))
            
            if all_durations:
                unique_states = sorted(durations.keys())
                state_data = [durations[state] for state in unique_states]
                ax5.boxplot(state_data, labels=[f"S{s}" for s in unique_states])
                ax5.set_title("State Duration Distribution")
                ax5.set_xlabel("State")
                ax5.set_ylabel("Duration (frames)")
    
    # Plot 6: Markov Chain Graph
    ax6 = plt.subplot(2, 3, 6)
    G = nx.DiGraph()
    n_states = transition_matrix.shape[0]
    
    # Add nodes and edges
    for i in range(n_states):
        G.add_node(i)
    
    for i in range(n_states):
        for j in range(n_states):
            prob = transition_matrix[i, j]
            if prob > 0.01:  # Only show significant transitions
                G.add_edge(i, j, weight=prob)
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', ax=ax6)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax6)
    
    edges = [(u, v) for u, v in G.edges() if u != v]
    if edges:
        edge_weights = [G[u][v]['weight'] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_weights, 
                              arrowsize=15, arrowstyle='->', ax=ax6)
    
    ax6.set_title("Markov Chain Graph")
    ax6.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive HMM analysis to {output_path}")
    
    return fig

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save all results to files."""
    # Save evaluation results
    for dataset in ['train', 'test']:
        if f'{dataset}_evaluation' in results:
            evaluation = results[f'{dataset}_evaluation']
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
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
            json_results["frame_to_state"] = frame_states
            
            # Format transition counts
            transition_counts = {}
            for (from_state, to_state), count in evaluation["transition_counts"].items():
                transition_counts[f"{from_state}->{to_state}"] = count
            json_results["transition_counts"] = transition_counts
            
            # Save to JSON
            results_file = os.path.join(output_dir, f"{dataset}_evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Saved {dataset} evaluation results to {results_file}")
    
    # Save summary
    summary = {
        "embedding_info": results.get("embedding_info", {}),
        "train_log_likelihood": float(results.get("train_log_likelihood", 0)),
        "test_log_likelihood": float(results.get("test_log_likelihood", 0)),
        "outliers_detected": len(results.get("outlier_indices", []))
    }
    
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis summary to {summary_file}")

def process_embeddings_analysis(
    embeddings_file: str,
    output_dir: str = None,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    train_ratio: float = 2/3,
    pca_components: int = 20,
    normalize: bool = True,
    apply_standard_scaling: bool = True,
    detect_outliers_flag: bool = True,
    z_threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Main orchestrating function for comprehensive embeddings analysis.
    Similar to train_test_sequential_split from the headless version.
    """
    # Validate inputs
    embeddings_file = validate_embeddings_path(embeddings_file)
    output_dir = validate_output_path(output_dir, embeddings_file)
    
    print(f"Processing embeddings: {embeddings_file}")
    print(f"Output directory: {output_dir}")
    
    # Train comprehensive HMM
    results = train_hidden_markov_model(
        action_sequence=embeddings_file,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        train_ratio=train_ratio,
        pca_components=pca_components,
        normalize=normalize,
        apply_standard_scaling=apply_standard_scaling,
        detect_outliers_flag=detect_outliers_flag,
        z_threshold=z_threshold,
        output_dir=output_dir
    )
    
    # Generate all plots
    print("Generating plots...")
    
    # Plot transition matrix
    transition_plot = os.path.join(output_dir, "transition_matrix.png")
    plot_markov_model_matrix(
        results['train_evaluation']['transition_matrix'],
        output_path=transition_plot,
        title=f"HMM Transition Matrix (n_states={n_states})"
    )
    
    # Plot state proportions
    proportions_plot = os.path.join(output_dir, "state_proportions.png")
    plot_categorical_model_distribution(
        results['train_evaluation']['state_proportions'],
        output_path=proportions_plot,
        title="State Frequency Distribution"
    )
    
    # Plot comprehensive analysis
    comprehensive_plot = os.path.join(output_dir, "comprehensive_hmm_analysis.png")
    plot_hidden_markov_model_matrices(
        {**results, 'frame_names': results['train_evaluation']['frame_to_state'].keys()},
        output_path=comprehensive_plot,
        title="Comprehensive HMM Analysis"
    )
    
    # Save all results
    save_results(results, output_dir)
    
    print(f"Analysis complete! Results saved to: {output_dir}")
    
    return {
        "status": "success",
        "embeddings_file": embeddings_file,
        "output_directory": output_dir,
        "train_log_likelihood": results['train_log_likelihood'],
        "test_log_likelihood": results['test_log_likelihood'],
        "n_states": n_states,
        "pca_components": pca_components
    }

def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Analyze UI element embeddings using Hidden Markov Models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python naive_action_analysis.py
  python naive_action_analysis.py --embeddings ./embeddings.npz
  python naive_action_analysis.py --embeddings ./embeddings.npz --output ./results
  python naive_action_analysis.py -e ./embeddings.npz -o ./results --n_states 5

Default paths:
  Embeddings: ./embeddings.npz
  Output: ./output
        '''
    )
    
    parser.add_argument(
        '--embeddings', '-e',
        type=str,
        help=f'Path to the input embeddings file (.npz) (default: {DEFAULT_EMBEDDINGS_PATH})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help=f'Path to the output folder (default: {DEFAULT_OUTPUT_FOLDER})'
    )
    
    parser.add_argument(
        '--n_states',
        type=int,
        default=3,
        help='Number of hidden states for the HMM (default: 3)'
    )
    
    parser.add_argument(
        '--covariance_type',
        default="diag",
        choices=["spherical", "tied", "diag", "full"],
        help='Type of covariance matrix for the HMM (default: diag)'
    )
    
    parser.add_argument(
        '--n_iter',
        type=int,
        default=100,
        help='Maximum number of iterations for EM algorithm (default: 100)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=2/3,
        help='Ratio of data to use for training (default: 0.667)'
    )
    
    parser.add_argument(
        '--pca_components',
        type=int,
        default=20,
        help='Number of PCA components to use (default: 20)'
    )
    
    parser.add_argument(
        '--no_normalize',
        action="store_true",
        help='Disable normalization of embeddings'
    )
    
    parser.add_argument(
        '--no_standard_scaling',
        action="store_true",
        help='Disable standard scaling after PCA'
    )
    
    parser.add_argument(
        '--no_outlier_detection',
        action="store_true",
        help='Disable outlier detection and handling'
    )
    
    parser.add_argument(
        '--z_threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    
    return parser

def main():
    """Main function to handle both CLI and interactive modes."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Determine embeddings path
    if args.embeddings:
        embeddings_path = args.embeddings
    else:
        print("No embeddings path provided via command line.")
        embeddings_path = get_user_input("Enter embeddings file path (.npz)", DEFAULT_EMBEDDINGS_PATH)
    
    # Validate embeddings path
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file '{embeddings_path}' does not exist.")
        print("Exiting due to invalid embeddings path.")
        sys.exit(1)
    
    if not embeddings_path.lower().endswith('.npz'):
        print(f"Error: '{embeddings_path}' is not an NPZ file.")
        print("Exiting due to invalid file format.")
        sys.exit(1)
    
    # Determine output folder
    if args.output:
        output_folder = args.output
    else:
        if not args.embeddings:  # Only prompt if not provided via CLI
            output_folder = get_user_input("Enter output folder path", DEFAULT_OUTPUT_FOLDER)
        else:
            output_folder = DEFAULT_OUTPUT_FOLDER
    
    # If output folder is not provided and no default, create one based on embeddings location
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(embeddings_path), "action_analysis_results")
        print(f"Using default output folder: {output_folder}")
    
    print(f"Processing embeddings: {embeddings_path}")
    print(f"Output folder: {output_folder}")
    print(f"HMM Configuration:")
    print(f"  - States: {args.n_states}")
    print(f"  - Covariance type: {args.covariance_type}")
    print(f"  - PCA components: {args.pca_components}")
    print(f"  - Train ratio: {args.train_ratio:.3f}")
    
    try:
        # Process embeddings with comprehensive analysis
        results = process_embeddings_analysis(
            embeddings_file=embeddings_path,
            output_dir=output_folder,
            n_states=args.n_states,
            covariance_type=args.covariance_type,
            n_iter=args.n_iter,
            train_ratio=args.train_ratio,
            pca_components=args.pca_components,
            normalize=not args.no_normalize,
            apply_standard_scaling=not args.no_standard_scaling,
            detect_outliers_flag=not args.no_outlier_detection,
            z_threshold=args.z_threshold
        )
        
        print("\nAnalysis Summary:")
        print(f"Train log likelihood: {results['train_log_likelihood']:.2f}")
        print(f"Test log likelihood: {results['test_log_likelihood']:.2f}")
        print(f"Results saved to: {results['output_directory']}")
        
    except Exception as e:
        print(f"Error processing embeddings: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    
# This script can be run directly or via CLI arguments.
# Run: python src/naive/naive_action_analysis.py
# CLI Mode: python src/naive/naive_action_analysis.py --embeddings path/to/embeddings.npz --output path/to/output
