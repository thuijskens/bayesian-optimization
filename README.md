# Bayesian optimization with Gaussian processes

This repository contains Python code for Bayesian optimization using Gaussian processes. It contains two directories:

* `python`: Contains two python scripts `gp.py` and `plotters.py`, that contain the optimization code, and utility functions to plot iterations of the algorithm, respectively.
* `ipython-notebooks`: Contains an IPython notebook that uses the Bayesian algorithm to tune the hyperparameters of a support vector machine on a dummy classification task.

The signature of the optimization function is:

```python
bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                      gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7)
```

and its docstring is:

```
bayesian_optimisation

  Uses Gaussian Processes to optimise the loss function `sample_loss`.

  Arguments:
  ----------
      n_iters: integer.
          Number of iterations to run the search algorithm.
      sample_loss: function.
          Function to be optimised.
      bounds: array-like, shape = [n_params, 2].
          Lower and upper bounds on the parameters of the function `sample_loss`.
      x0: array-like, shape = [n_pre_samples, n_params].
          Array of initial points to sample the loss function for. If None, randomly
          samples from the loss function.
      n_pre_samples: integer.
          If x0 is None, samples `n_pre_samples` initial points from the loss function.
      gp_params: dictionary.
          Dictionary of parameters to pass on to the underlying Gaussian Process.
      random_search: integer.
          Flag that indicates whether to perform random search or L-BFGS-B optimisation
          over the acquisition function.
      alpha: double.
          Variance of the error term of the GP.
      epsilon: double.
          Precision tolerance for floats.
```
