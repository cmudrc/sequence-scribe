import matplotlib.figure


def train_markov_model(action_sequence: list[list[int]]) -> list[list[float]]:

    return [[0.0]]


def train_hidden_markov_model(
    action_sequence: list[list[int]],
) -> [list[list[float]], list[list[float]]]:
    return [[0.0]], [[0.0]]


def plot_markov_model_matrix(
    action_sequence: list[list[int]], action_names: list[str] = None
) -> matplotlib.figure.Figure:
    return matplotlib.figure.Figure()


def plot_hidden_markov_model_matrices(
    action_sequence: list[list[int]], action_names: list[str] = None
) -> matplotlib.figure.Figure:
    return matplotlib.figure.Figure()
