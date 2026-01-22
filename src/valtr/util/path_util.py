import pathlib


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent.parent


def get_paper_plot_dir() -> pathlib.Path:
    return get_root_dir() / "paper" / "plots"
