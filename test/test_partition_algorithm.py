import pytest
from scipy import stats
from scipy.optimize import minimize_scalar

from src.partition import algorithm as partition


def assert_close(a, b, tol=1e-6):
    return a - b < tol and b - a < tol


TOL = 1e-3


def test_is_continuous():
    # continue distribution
    dist = stats.norm(scale=5)
    assert partition.is_continuous(dist)

    # discrete distribution
    dist = stats.poisson(mu=5)
    assert not partition.is_continuous(dist)

    # custom discrete distribution
    xk = (1, 2, 3, 4)
    pk = (0.1, 0.2, 0.3, 0.4)
    custom_dist = stats.rv_discrete(name="custom", values=(xk, pk))
    assert not partition.is_continuous(custom_dist)


def test_partial_expectation_normal():
    # check partial expectation of normal distribution on (-inf, inf) is 0
    dist = stats.norm()
    lb = -float("inf")
    ub = float("inf")
    assert partition.partial_expectation(dist, lb, ub) == 0


def test_partial_expectation_poisson_1():
    # check partial expectation of poisson distribution on (-inf, 5) + (5, inf) is 5
    dist = stats.poisson(mu=5)
    x = partition.partial_expectation(dist, -partition.MAX_INT, 5)
    y = partition.partial_expectation(dist, 5, partition.MAX_INT)
    assert x + y == pytest.approx(dist.mean())


def test_conditional_expectation_normal():
    dist = stats.norm()
    lb = -float("inf")
    assert partition.conditional_expectation(dist, lb, 0) == pytest.approx(-0.7979, TOL)


def test_create_discrete_partition_poisson():
    dist = stats.poisson(mu=5)
    p = [(1, 10), (10, 15)]
    mu_k, x_k = partition._create_tilde_X(dist, p)
    approx_dist = partition.create_tilde_X(dist, p)

    # check sum of probabilities is 1
    assert sum(x_k) == pytest.approx(1, TOL)
    # check mean is same
    assert approx_dist.mean() == pytest.approx(dist.mean(), TOL)


def test_create_discrete_partition_normal():
    dist = stats.norm()
    p = [(-1, 1), (1, 2)]
    mu_k, x_k = partition._create_tilde_X(dist, p)
    approx_dist = partition.create_tilde_X(dist, p)
    # check sum of probabilities is 1
    assert sum(x_k) == pytest.approx(1, TOL)
    # check mean is same
    assert approx_dist.mean() == pytest.approx(dist.mean(), TOL)


def test_loss_function_normal():
    # check loss function of normal distribution is correct (the actual value is calculated by Wolfram Alpha)
    dist = stats.norm()
    loss_func = partition.make_loss_func(dist)
    assert loss_func(0) == pytest.approx(-0.3989, TOL)


def test_bound():
    # check bound of breakpoints is correct (the actual value is calculated by Wolfram Alpha)
    a, b = -3, 3
    epsilon = 0.1
    dist = stats.norm()
    assert partition.calc_bound_of_breakpoints(dist, epsilon, a, b, continuous_flag=True) == 4


@pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001, 0.0001])
def test_approx_loss_function_normal(epsilon):
    dist = stats.norm()
    a_min, b_max = -3, 3
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    # check upper bound of the number of breakpoints is correct
    assert len(approx_dist.xk) - 2 <= partition.calc_bound_of_breakpoints(
        dist, epsilon, a_min, b_max, continuous_flag=True
    )

    # the maximum error is less than epsilon
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert max_error < epsilon


@pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001, 0.0001])
def test_approx_loss_function_normal_exact(epsilon):
    dist = stats.norm()
    a_min, b_max = -3, 3
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("exact"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    # check upper bound of the number of breakpoints is correct
    assert len(approx_dist.xk) - 2 <= partition.calc_bound_of_breakpoints(
        dist, epsilon, a_min, b_max, continuous_flag=True
    )

    # the maximum error is less than epsilon
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert max_error < epsilon


@pytest.mark.parametrize("epsilon", [0.1, 0.01])
def test_approx_loss_function_poisson(epsilon):
    dist = stats.poisson(mu=2)
    a_min, b_max = 1, 5
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    # check upper bound of the number of breakpoints is correct
    assert len(approx_dist.xk) - 2 <= partition.calc_bound_of_breakpoints(
        dist, epsilon, a_min, b_max, continuous_flag=False
    )

    # the maximum error is less than epsilon
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert max_error < epsilon


@pytest.mark.parametrize("epsilon", [0.1, 0.01])
def test_approx_loss_function_poisson_exact(epsilon):
    dist = stats.poisson(mu=2)
    a_min, b_max = 1, 5
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("exact"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    # check upper bound of the number of breakpoints is correct
    assert len(approx_dist.xk) - 2 <= partition.calc_bound_of_breakpoints(
        dist, epsilon, a_min, b_max, continuous_flag=False
    )

    # the maximum error is less than epsilon
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert max_error < epsilon


def test_calc_approximation_error_with_max_error():
    # check the approximation error is correct (the estimated value is calculated by scipy.optimize.minimize_scalar)
    dist = stats.norm()
    a_min, b_max = -2, 2
    epsilon = 0.1
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)

    max_error_by_delta = partition.calc_approximation_error(dist, partition=p)
    # TODO: check the error is less than 0.1
    # assert max_error_by_delta == pytest.approx(max_error, 0.1)
    assert abs(max_error - max_error_by_delta) < 0.1


def test_calc_approximation_error_with_value_of_breakpoint():
    # check the approximation error is correct (the estimated value is calculated by the value of breakpoint)
    dist = stats.norm()
    a_min, b_max = -2, 2
    epsilon = 0.01
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)
    max_error_by_delta = partition.calc_approximation_error(dist, partition=p)
    conditional_expectations = [partition.conditional_expectation(dist, a, b) for (a, b) in p]
    max_error_by_ce = max([abs(actual_loss_func(x) - approx_loss_func(x)) for x in conditional_expectations])
    assert abs(max_error_by_delta - max_error_by_ce) < 0.00001


def test_calc_exact_delta_wolfram_with_norm():
    dist = stats.norm()
    a, b = 1, 3
    mu = 1.5100495132439835
    p = 0.157305
    bar_p = 0.5920959151032898
    bar_mu = 1.2282315449192902
    delta = partition.calc_exact_delta(dist, a, b)
    delta_simple = partition.calc_exact_delta_simple(dist, a, b)
    actual_delta = p * bar_p * (mu - bar_mu)
    assert partition.conditional_expectation(dist, a, b) == pytest.approx(mu, TOL)
    assert dist.cdf(b) - dist.cdf(a) == pytest.approx(p, TOL)
    assert partition.conditional_expectation(dist, a, mu) == pytest.approx(bar_mu, TOL)
    # assert dist.cdf(mu) - dist.cdf(a) == pytest.approx(bar_p, TOL)

    assert delta == pytest.approx(actual_delta, TOL)
    assert delta_simple == pytest.approx(actual_delta, TOL)


def test_calc_exact_delta_wolfram_with_poisson():
    a, b = 0, 3
    dist = stats.poisson(mu=1)
    mu = 1.5
    p = 0.61313240
    bar_p = 3 / 5
    bar_mu = 1
    assert partition.conditional_expectation(dist, a, b) == pytest.approx(mu)
    assert dist.cdf(b) - dist.cdf(a) == pytest.approx(p)
    assert partition.conditional_expectation(dist, a, mu) == pytest.approx(bar_mu, TOL)
    delta = partition.calc_exact_delta(dist, a, b)
    actual_delta = p * bar_p * (mu - bar_mu)
    assert delta == pytest.approx(actual_delta, TOL)
    simple_delta = partition.calc_exact_delta_simple(dist, a, b)
    assert simple_delta == pytest.approx(actual_delta, TOL)


def test_calc_exact_delta_opt_with_poisson():
    a, b = 0, 3
    dist = stats.poisson(mu=1)
    approx_dist = partition.create_tilde_X(dist, [(a, b)])
    delta = partition.calc_exact_delta(dist, a, b)
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a, b), method="bounded")
    max_error = -objective_function(result.x)
    assert delta == pytest.approx(max_error, 0.001)


def test_calc_exact_delta_simple_opt_with_poisson():
    a, b = 0, 5
    dist = stats.poisson(mu=1)
    approx_dist = partition.create_tilde_X(dist, [(a, b)])
    delta = partition.calc_exact_delta_simple(dist, a, b)
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)
    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a, b), method="bounded")
    max_error = -objective_function(result.x)
    assert delta == pytest.approx(max_error, 0.001)


def test_calc_exact_delta_with_poisson_when_delta_0():
    # check delta is 0 when delta is 0 (when poisson distribution)
    poisson_dist = stats.poisson(mu=5)
    dist = poisson_dist
    partition.calc_exact_delta(dist, 0, 1)
    # assert_close(delta, 0)
    a_min = 0
    b_max = 10
    epsilon = 0.01
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert partition.calc_approximation_error(dist, partition=p) <= epsilon
    assert max_error <= 0.00001


def _test_clac_exact_delta_simple():
    a_min, b_max = 20, 30
    dist = stats.poisson(mu=30)
    delta_simple = partition.calc_exact_delta_simple(dist, a_min, b_max)
    delta = partition.calc_exact_delta(dist, a_min, b_max)
    assert delta_simple == delta


def _test_calc_exact_delta_3():
    a_min, b_max = 0, 100
    epsilon = 0.01
    dist = stats.poisson(mu=30)
    approx_dist, p = partition.make_approx_dist(dist, a_min, b_max, epsilon, partition.get_bound_func("approx"))
    actual_loss_func = partition.make_loss_func(dist)
    approx_loss_func = partition.make_loss_func(approx_dist)

    objective_function = lambda x: -abs(actual_loss_func(x) - approx_loss_func(x))
    result = minimize_scalar(objective_function, bounds=(a_min, b_max), method="bounded")
    max_error = -objective_function(result.x)
    assert partition.calc_approximation_error(dist, partition=p) <= epsilon
    assert max_error <= epsilon


def _test_calc_delta_2_method():
    dist = stats.poisson(mu=30)
    a_min, b_max = 0, 15
    assert partition.calc_exact_delta(dist, a_min, b_max) == partition.calc_exact_delta_simple(dist, a_min, b_max)
