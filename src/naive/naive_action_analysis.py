import matplotlib.figure
import hmmlearn.hmm


def train_hidden_markov_model(
    action_sequence: list[list[int]],
) -> hmmlearn.hmm.CategoricalHMM:

    model = hmmlearn.hmm.CategoricalHMM(n_components=4)

    model.fit(action_sequence)

    return model


def train_markov_model(
    action_sequence: list[list[int]],
) -> list[list[float]]:
    # Estimate transition matrix for every discrete token in action_sequence
    token_transitions = {}
    for action in action_sequence:
        for i, token in enumerate(action):
            if token not in token_transitions:
                token_transitions[token] = {}
            if i == 0:
                continue
            if action[i - 1] not in token_transitions[token]:
                token_transitions[token][action[i - 1]] = 0
            token_transitions[token][action[i - 1]] += 1

    # Normalize transitions to probabilities
    for token in token_transitions:
        total_transitions = sum(token_transitions[token].values())
        for prev_token in token_transitions[token]:
            token_transitions[token][prev_token] /= total_transitions

    return token_transitions


def train_categorical_model(
    action_sequence: list[list[int]],
) -> dict[int, float]:
    # Estimate frequency for every discrete token in action_sequence
    token_frequencies = {}
    for action in action_sequence:
        for token in action:
            if token not in token_frequencies:
                token_frequencies[token] = 0
            token_frequencies[token] += 1

    # Normalize frequencies to probabilities
    total_tokens = sum(token_frequencies.values())
    for token in token_frequencies:
        token_frequencies[token] /= total_tokens

    return token_frequencies


def plot_categorical_model_distribution(
    token_frequencies: list[float], action_names: list[str] = None
) -> matplotlib.figure.Figure:
    return matplotlib.figure.Figure()


def plot_markov_model_matrix(
    action_sequence: list[list[int]], action_names: list[str] = None
) -> matplotlib.figure.Figure:
    return matplotlib.figure.Figure()


def plot_hidden_markov_model_matrices(
    action_sequence: list[list[int]], action_names: list[str] = None
) -> matplotlib.figure.Figure:
    return matplotlib.figure.Figure()
