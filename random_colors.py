import numpy as np


def get_nth_random_color(n=0, seed=42):
    """
    Generate the n-th random color using a static seed.

    Parameters:
    n (int): The position in the sequence of random colors to return. Starts from 0.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    numpy.ndarray: An array containing the RGB values of the color.
    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Generate n random colors
    colors = np.random.randint(0, 256, size=(n + 1, 3))

    # Return the n-th color
    return colors[n]
