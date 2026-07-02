import pandas as pd
import numpy as np
import itertools


class HyperparameterGrid:

    def __init__(self, grid_with_results: dict = None):

        # sweep variables
        if grid_with_results is None:
            self.base_learning_rates = []
            self.base_init_stddevs = []

            self.update_sweep_variables()

        # if grid_with_results exists
        else:
            # TODO: read out grid and optimal lr and init var
            pass

    def update_sweep_variables(self, variables: list = None):
        self.potential_sweep_variables = {
            "base_lr": self.base_learning_rates,
            "base_init_stddev": self.base_init_stddevs,
        }
        variables = self.potential_sweep_variables.keys()
        self.sweep_variables = {v: self.potential_sweep_variables[v] for v in variables}

    def populate_naive_grid(self, n_lrs: int = 5, n_init_stddevs: int = 5):
        """
        Naive grid with parameter spacing log2 base.
        """
        min_lr_exponent = -10  # e.g. 2^{-10}
        max_lr_exponent = -3  # e.g. 2^{-2}

        learning_rates = np.logspace(min_lr_exponent, max_lr_exponent, n_lrs, base=2)

        min_init_stddev_exponent = -2
        max_init_stddev_exponent = 2

        init_stddevs = np.logspace(
            min_init_stddev_exponent, max_init_stddev_exponent, n_init_stddevs, base=2
        )

        self.base_learning_rates = learning_rates
        self.base_init_stddevs = init_stddevs



        return {"learning_rates": learning_rates, "init_stddevs": init_stddevs}


if __name__ == "__main__":
    grid = HyperparameterGrid()
    grid.populate_naive_grid(n_lrs=10, n_init_stddevs=5)
    combos = list(itertools.product(grid.base_learning_rates, grid.base_init_stddevs))
    pd.DataFrame(combos, columns=["base_lr", "base_init_stddev"]).to_csv(
        "grid_manifest.csv", index=False
    )
    print(len(combos))  # use this to size --array=0-(N-1)
