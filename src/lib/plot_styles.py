import matplotlib.pyplot as plt

from src.lib.globals import STYLE_FILEPATH


def publication_style() -> None:
    plt.style.use(STYLE_FILEPATH)
