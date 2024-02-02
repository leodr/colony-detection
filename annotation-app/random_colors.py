import numpy as np


def get_n_random_colors(n=1, seed=42) -> np.ndarray:
    """Generates n unique random colors with a seed.

    Args:
        n (int, optional): The amount of unique colors to generate. Defaults to 1.
        seed (int, optional): The seed to use. Defaults to 42.

    Returns:
        ndarray: An array of the unique random colors.
    """

    np.random.seed(seed)

    unique_colors = []

    while len(unique_colors) < n:
        color = tuple(np.random.randint(3, 256, size=3))

        if color not in unique_colors:
            unique_colors.append(color)

    colors = np.array(unique_colors)

    return colors
