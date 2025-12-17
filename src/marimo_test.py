import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _(np, plt):
    x = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x, np.sin(x))
    return


if __name__ == "__main__":
    app.run()
