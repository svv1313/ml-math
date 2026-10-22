# Machine Learning Foundations

Jupyter notebooks covering the math behind machine learning: linear algebra, calculus, and gradient-based optimization, implemented with NumPy, PyTorch, and scikit-learn. Each notebook has an "Open in Colab" badge at the top.

## File structure

```
ml-math/
├── Readme.md
└── notebooks/
    ├── trace_operation.ipynb                    # linear algebra
    ├── matrix_operation_for_ml.ipynb            # linear algebra
    ├── moore_penrose_pseudoinverse.ipynb        # linear algebra
    ├── principal_component_analysis.ipynb       # linear algebra
    ├── delta_method.ipynb                       # calculus
    ├── partial_derivatives.ipynb                # calculus
    ├── single_point_regression_gradient.ipynb   # optimization
    ├── batch_regression_gradient.ipynb          # optimization
    └── regression_in_pytorch.ipynb              # optimization
```

## Notebooks

### Linear algebra

| Notebook | What it covers |
| --- | --- |
| [trace_operation.ipynb](notebooks/trace_operation.ipynb) | The trace of a matrix in PyTorch, and using it to show that the Frobenius norm equals $\sqrt{\mathrm{Tr}(AA^T)}$. |
| [matrix_operation_for_ml.ipynb](notebooks/matrix_operation_for_ml.ipynb) | Singular Value Decomposition (SVD) with NumPy and PyTorch: rebuilding a matrix from $UDV^T$ and linking singular values to the eigenvalues of $AA^T$ and $A^TA$. |
| [moore_penrose_pseudoinverse.ipynb](notebooks/moore_penrose_pseudoinverse.ipynb) | Deriving the pseudoinverse $A^+ = VD^+U^T$ from SVD, checking it against `pinv()`, and using it to solve an overdetermined system of linear equations. |
| [principal_component_analysis.ipynb](notebooks/principal_component_analysis.ipynb) | PCA with scikit-learn: reducing the Iris dataset to two principal components and plotting the result. |

### Calculus & optimization

| Notebook | What it covers |
| --- | --- |
| [delta_method.ipynb](notebooks/delta_method.ipynb) | Approximating the slope of a curve at a point by taking secant lines through ever-closer points (the delta method behind derivatives). |
| [partial_derivatives.ipynb](notebooks/partial_derivatives.ipynb) | Partial derivatives computed with PyTorch autograd, e.g. for $x^2 + y^2$, $x^2 - y^2$, and the volume of a cylinder. |
| [single_point_regression_gradient.ipynb](notebooks/single_point_regression_gradient.ipynb) | Gradient of the squared-error cost of a linear regression with respect to slope $m$ and intercept $b$, for a single data point. |
| [batch_regression_gradient.ipynb](notebooks/batch_regression_gradient.ipynb) | Gradient of the mean squared error over a whole batch, checked by hand, then used to fit $m$ and $b$ with SGD. |
| [regression_in_pytorch.ipynb](notebooks/regression_in_pytorch.ipynb) | End-to-end linear regression in PyTorch on a toy "clinical trial" dataset: autograd, a training loop, and plots of the fitted line. |
