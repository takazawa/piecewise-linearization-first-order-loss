# Piecewise Linearization for First-Order Loss Functions

## Requirements

- `Python>=3.8`
- `scipy>=1.8.1`

## Install

```shell
pip install git+https://github.com/takazawa/piecewise-linearization-first-order-loss
```

## Usage


#### Parameters:

- `dist`: The probability distribution you want to approximate. This should be an instance of the probability distribution classes from `scipy.stats` (often referred to as the `Rv` type in the code).
- `a_min`: Minimum value of the range.
- `b_max`: Maximum value of the range.
- `epsilon`: Accuracy of the approximation.
- `bound_func` (optional): The bounding function. Defaults to `calc_exact_delta`.

#### Returns:

- An approximated discrete random variable, which is an instance of the probability distribution classes from `scipy.stats`
- A list indicating the partitions of (a_min, b_max].

#### Example:

```python
import scipy
from partition import make_approx_dist

dist = scipy.stats.norm()  # Define your probability distribution here using scipy.stats
a_min = 0
b_max = 10
epsilon = 0.01

new_dist, partition_list = make_approx_dist(dist, a_min, b_max, epsilon)

```

By using the above sample, `new_dist` will contain the approximated discrete random variable and `partition_list` will contain the partitions of the distribution.

---

Note: When using the `Rv` type in the code, it refers to the probability distribution classes from `scipy.stats`. Make sure to define your distributions using these classes for compatibility.



