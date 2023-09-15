from __future__ import annotations

import math
from typing import Union

from scipy import optimize
from scipy.integrate import quad
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

RvContinuous = Union[rv_continuous, rv_continuous_frozen]
RvDiscrete = Union[rv_discrete, rv_discrete_frozen]
Rv = Union[RvContinuous, RvDiscrete]
MAX_INT = 10000


def is_continuous(dist: Rv) -> bool:
    # check if dist is continuous or discrete
    if isinstance(dist, rv_continuous) or isinstance(dist, rv_continuous_frozen):
        return True
    elif isinstance(dist, rv_discrete) or isinstance(dist, rv_discrete_frozen):
        return False
    else:
        assert "Type Error"


def get_inf(dist: Rv) -> int | float:
    # get a and b for (-inf, inf)
    if is_continuous(dist):
        return float("inf")
    else:
        return MAX_INT


def calc_bound_of_breakpoints(
    dist: Rv, epsilon: float, a: int | float, b: int | float, continuous_flag: bool, approx_flag: bool = False
) -> int:
    param_rv = 1 if continuous_flag else 2
    param_alg = 4 if not approx_flag else 8
    prob = dist.cdf(b) - dist.cdf(a)
    coefficient = ((1 + prob) * param_rv) / (2 * math.sqrt(param_alg))
    return int(coefficient * math.sqrt((b - a) / epsilon) + 1)


def _calc_bound_of_breakpoints(
    dist: Rv, epsilon: float, a: int | float, b: int | float, continuous_flag: bool, approx_flag: bool = False
) -> float:
    param_rv = 1 if continuous_flag else 2
    param_approx = 4 if not approx_flag else 8
    # prob = dist.cdf(b) - dist.cdf(a)
    prob = 1
    coefficient = ((1 + prob) * param_rv) / (2 * math.sqrt(param_approx))
    return coefficient * math.sqrt((b - a) / epsilon)


def calc_delta_bound(dist: Rv, a: int | float, b: int | float, param: int = 4) -> float:
    # calculate delta bound in (a, b]
    prob = dist.cdf(b) - dist.cdf(a)
    return prob * (b - a) / param


def calc_exact_delta(dist: Rv, a: int | float, b: int | float) -> float:
    if dist.cdf(b) - dist.cdf(a) == 0:
        return 0
    # calculate exact delta in (a, b]
    mu = conditional_expectation(dist, a, b)  # E[X|X in (a, b]]
    bar_mu = conditional_expectation(dist, a, mu)  # E[X|X in (a, mu]]
    delta = (dist.cdf(mu) - dist.cdf(a)) * (mu - bar_mu)  # delta in (a, b]
    return delta


def calc_exact_delta_simple(dist: Rv, a: int | float, b: int | float) -> float:
    if dist.cdf(b) - dist.cdf(a) == 0:
        return 0
    # calculate exact delta in (a, b]
    mu = conditional_expectation(dist, a, b)  # E[X|X in (a, b]]
    if mu == a or mu == b:
        return 0
    return partial_expectation(dist, a, mu, lambda x: (mu - x))


def partial_expectation(dist: Rv, a: int | float, b: int | float, func: callable = lambda x: x) -> float:
    # calculate partial expectation X in (a, b]
    if is_continuous(dist):
        val = quad(lambda x: func(x) * dist.pdf(x), a, b, epsabs=10 ** (-8))[0]
    else:
        if isinstance(dist, rv_discrete_frozen):
            if a < -MAX_INT:
                a = -MAX_INT
            if b > MAX_INT:
                b = MAX_INT
            val = sum([func(x) * dist.pmf(x) for x in range(int(a) + 1, int(b) + 1)])
        else:
            xk = dist.xk
            val = sum([func(x) * dist.pmf(x) for x in xk if a < x <= b])
    return val


def conditional_expectation(dist: Rv, a: int | float, b: int | float, func: callable = lambda x: x) -> float:
    # calculate E[X|X in (a, b]]
    prob = dist.cdf(b) - dist.cdf(a)
    if not is_continuous(dist) and b - a == 1:
        return b
    if prob == 0:
        assert False, f"the probability of ({a}, {b}] is 0"
    return partial_expectation(dist, a, b, func) / prob


def _calc_next_b_continuous(
    dist: RvContinuous, epsilon: float, a: int | float, b_max: int | float, delta_func: callable
) -> float:
    # calculate maximum b such that delta(a, b) <= epsilon
    def _calc_next_b_error(b):
        if b - a < epsilon:
            return -epsilon
        return delta_func(dist, a, b) - epsilon

    result = optimize.root_scalar(_calc_next_b_error, bracket=[a, b_max], method="bisect", xtol=0.1**8)
    return result.root


def _calc_next_b_discrete(
    dist: RvDiscrete, epsilon: float, a: int | float, b_max: int | float, delta_func: callable
) -> float:
    # calculate maximum b such that delta(a, b) <= epsilon
    for k in range(a + 1, b_max + 1):
        if delta_func(dist, a, k) > epsilon:
            if k == a + 1:
                return a + 1
            else:
                return k - 1
    assert "Error"


def calc_next_b(dist: Rv, epsilon: float, a: int | float, b_max: int | float, delta_func: callable) -> float | int:
    # calculate maximum b such that delta(a, b) <= epsilon
    if is_continuous(dist):
        next_b = _calc_next_b_continuous(dist, epsilon, a, b_max, delta_func)
    else:
        next_b = _calc_next_b_discrete(dist, epsilon, a, b_max, delta_func)
    return next_b


def partition_algorithm(
    dist: Rv, epsilon: float, a_min: int | float, b_max: int | float, delta_func: callable
) -> list[tuple[float | int, float | int]]:
    # calculate partition of dist in (a_min, b_max]
    partition = []
    a = a_min

    while delta_func(dist, a, b_max) > epsilon:
        b = calc_next_b(dist, epsilon, a=a, b_max=b_max, delta_func=delta_func)
        partition.append((a, b))
        a = b

    if a != b_max:
        partition.append((a, b_max))
    print("I=", partition)
    return partition


def uniform_partition_algorithm(
    a_min: int | float, b_max: int | float, n: int
) -> list[tuple[float | int, float | int]]:
    # calculate partition of dist in (a_min, b_max]
    partition = []
    a = a_min
    w = (b_max - a_min) / n
    while abs(a - b_max) >= 0.0001:
        b = a + w
        partition.append((a, b))
        a = b
    return partition


def make_loss_func(dist: Rv) -> callable:
    # make loss function of dist
    def loss_func(s):
        inf = get_inf(dist)
        return partial_expectation(dist, a=-inf, b=s) + s * (1 - dist.cdf(s))

    return loss_func


def _create_tilde_X(dist: Rv, partition: list[tuple[float | int, float | int]]) -> tuple[list, list]:
    # create random variable from partition
    partition = sorted(partition, key=lambda x: x[0])

    mu_k = []  # list of the values realized by the random variable
    p_k = []  # list of the probabilities of the values realized by the random variable

    inf = get_inf(dist)

    # calculate mu_0 and p_0
    a_min = partition[0][0]
    if dist.cdf(a_min) >= 0.0001:
        mu_0 = conditional_expectation(dist, a=-inf, b=a_min)
        p_0 = dist.cdf(a_min)
        mu_k.append(mu_0)
        p_k.append(p_0)

    # calculate mu_k and p_k for k = 1, ..., n
    for a, b in partition:
        if dist.cdf(b) - dist.cdf(a) >= 0.0001:
            mu_k.append(conditional_expectation(dist, a, b))
            p_k.append(dist.cdf(b) - dist.cdf(a))

    # calculate mu_n+1 and p_n+1
    b_max = partition[-1][1]
    if 1 - dist.cdf(b_max) >= 0.0001:
        mu_n_1 = conditional_expectation(dist, a=b_max, b=inf)
        p_n_1 = 1 - dist.cdf(b_max)
        mu_k.append(mu_n_1)
        p_k.append(p_n_1)

    sum_p = sum(p_k)
    if abs(sum_p - 1) > 0.001:
        assert False, f"sum of probabilities (={sum_p}) is not 1"
    elif 0 < abs(sum_p - 1) <= 0.001:
        p_k = [p / sum_p for p in p_k]

    return mu_k, p_k


def create_tilde_X(dist: Rv, partition: list[tuple[float | int, float | int]]) -> rv_discrete:
    # create random variable from partition
    mu_k, p_k = _create_tilde_X(dist, partition)
    return rv_discrete(name="tilde X", values=(mu_k, p_k))


def get_bound_func(func_type: str, param=4) -> callable:
    # get bound function
    if func_type == "approx":
        return lambda dist, a, b: calc_delta_bound(dist, a, b, param)
    elif func_type == "exact":
        return calc_exact_delta_simple
    else:
        assert False, "func_type must be 'approx' or 'exact'"


def make_approx_dist(
    dist: Rv, a_min, b_max, epsilon: float, bound_func: callable = calc_exact_delta
) -> tuple[rv_discrete, list[tuple[float | int, float | int]]]:
    # make approximated discrete random variable from dist with partition algorithm
    p = partition_algorithm(dist, epsilon, a_min, b_max, bound_func)
    new_dist = create_tilde_X(dist, p)
    return new_dist, p


def make_uniform_approx_dist(
    dist: Rv, a_min, b_max, n: int
) -> tuple[rv_discrete, list[tuple[float | int, float | int]]]:
    # make approximated discrete random variable from dist with partition algorithm
    p = uniform_partition_algorithm(a_min, b_max, n)
    new_dist = create_tilde_X(dist, p)
    return new_dist, p


def calc_approximation_error(dist: Rv, partition: list[tuple[float | int, float | int]]) -> float:
    # calculate approximation error
    return max([calc_exact_delta_simple(dist, a, b) for a, b in partition])
