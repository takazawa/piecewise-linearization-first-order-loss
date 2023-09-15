from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from scipy import stats

from src.partition import algorithm as pa


# 列挙型の定義
class AlgType(Enum):
    EXACT = "exact"
    APPROX4 = "approx4"
    APPROX8 = "approx8"


@dataclass()
class AlgorithmOutput:
    error: float
    num_breakpoints: int


@dataclass()
class Result:
    epsilon: float
    ub_breakpoints: dict[AlgType, float]
    alg_outputs: dict[AlgType, AlgorithmOutput]


@dataclass()
class Experiment:
    dist_name: str
    case_name: str
    dist_param: str
    rv_type: str
    interval: tuple[float | int, float | int]
    results: list[Result]

    def __init__(self, dist_name: str, dist: pa.Rv, support: tuple[float, float], dist_param: str, case_name: str):
        self.dist_name = dist_name
        self.case_name = case_name
        self.dist_param = dist_param
        self.rv_type = "continuous" if pa.is_continuous(dist) else "discrete"
        mean = dist.mean()

        a = max(support[0], mean - 3 * dist.std())
        b = min(support[1], mean + 3 * dist.std())
        if not pa.is_continuous(dist):
            a, b = int(a), int(b)
        self.interval = (a, b)
        self.results = []
        self.__dist = dist
        self.__is_continuous = pa.is_continuous(dist)

        print("dist_name=", dist_name, "a=", a, "b=", b, "mean=", mean, "std=", dist.std())

    def run(self, epsilons=(0.1, 0.05, 0.01)):
        for epsilon in epsilons:
            print("*****epsilon=", epsilon, "*****")
            ub_breakpoints = {}
            alg_outputs = {}
            for alg_type in AlgType:
                print("alg_type=", alg_type)
                approx_flag = True if alg_type == AlgType.APPROX8 else False
                if alg_type != AlgType.EXACT:
                    ub_breakpoints[alg_type] = pa.calc_bound_of_breakpoints(
                        self.__dist,
                        epsilon=epsilon,
                        a=self.interval[0],
                        b=self.interval[1],
                        continuous_flag=self.__is_continuous,
                        approx_flag=approx_flag,
                    )

                alg_outputs[alg_type] = do_one_experiment(
                    self.__dist, self.interval[0], self.interval[1], epsilon, alg_type
                )
            self.results.append(Result(epsilon, ub_breakpoints, alg_outputs))


@dataclass()
class Row:
    name: str
    epsilon: float
    ub_breakpoints: int
    b_exact: int
    b_4: int
    b_8: int
    e_exact: float
    e_4: float
    e_8: float


def do_one_experiment(dist: pa.Rv, a_min, b_max, epsilon: float, alg_type: AlgType) -> AlgorithmOutput:
    bound_func_dict = {
        AlgType.EXACT: pa.get_bound_func("exact"),
        AlgType.APPROX4: pa.get_bound_func("approx", param=4),
        AlgType.APPROX8: pa.get_bound_func("approx", param=8),
    }

    # run experiment
    approx_dist, partition = pa.make_approx_dist(dist, a_min, b_max, epsilon, bound_func_dict[alg_type])
    error = pa.calc_approximation_error(dist, partition)
    return AlgorithmOutput(error=error / epsilon, num_breakpoints=len(partition))


dists = [
    ("Normal", stats.norm(loc=0, scale=1), (-float("inf"), float("inf")), r"\mu=0, \sigma=1", "C-N1"),
    ("Normal", stats.norm(loc=0, scale=5), (-float("inf"), float("inf")), r"\mu=0, \sigma=5", "C-N2"),
    ("Exponential", stats.expon(scale=1), (0, float("inf")), r"\lambda=1", "C-Exp"),
    ("Uniform", stats.uniform(loc=0, scale=1), (0, 1), r"a=0, b=1", "C-Uni"),
    ("Beta", stats.beta(a=2, b=5), (0, 1), r"\alpha=2, \beta=5", "C-Bet"),
    ("Gamma", stats.gamma(a=2), (0, float("inf")), r"k=2, \theta=1", "C-Gam"),
    ("Chi-Squared", stats.chi2(df=3), (0, float("inf")), r"k=3", "C-Chi"),
    ("Student's t", stats.t(df=10), (-float("inf"), float("inf")), r"\nu=10", "C-Stu"),
    ("Logistic", stats.logistic(loc=0, scale=1), (-float("inf"), float("inf")), r"\mu=0, s=1", "C-Log"),
    ("Lognormal", stats.lognorm(s=1), (0, float("inf")), r"\mu=0, \sigma=1", "C-Lgn"),
    ("Binomial", stats.binom(n=200, p=0.5), (0, float("inf")), r"n=200, p=0.5", "D-Bin"),
    ("Poisson", stats.poisson(mu=100), (0, float("inf")), r"\lambda=100", "D-Poi"),
    ("Geometric", stats.geom(p=0.01), (1, float("inf")), r"p=0.01", "D-Geo"),
    ("Hypergeometric", stats.hypergeom(M=500, n=200, N=100), (0, float("inf")), r"M=500, n=200, N=100", "D-Hyp"),
    ("Negative Binomial", stats.nbinom(n=100, p=0.5), (0, float("inf")), r"r=100, p=0.5", "D-Neg"),
]


def modify_latex_table(latex_code: str) -> str:
    lines = latex_code.strip().split("\n")
    modified_lines = []
    previous_no = None
    for line in lines:
        if "breakpoints" in line:
            line = line.replace("r}{breakpoints}", "c}{breakpoints}")
        if "error" in line:
            line = line.replace("r}{error}", "c}{error}")
        if "C-" in line or "D-" in line:
            parts = line.split("&")
            current_no = parts[0].strip()
            if previous_no is not None and current_no == previous_no:
                parts[0] = " " * len(current_no)
            else:
                if previous_no is not None:
                    modified_lines.append("\\hline")
                previous_no = current_no
            line = "&".join(parts)
        modified_lines.append(line)
    return "\n".join(modified_lines)


def main():
    exp_results: list[Experiment] = []
    for dist_name, dist, support, dist_param, dataset_name in dists:
        # run experiment for each distribution
        exp = Experiment(dist_name, dist, support, dist_param, dataset_name)
        exp.run()
        exp_results.append(exp)

    rows = []
    for exp in exp_results:
        # create rows for csv
        a, b = exp.interval
        rv_type = exp.rv_type
        for result in exp.results:
            epsilon = result.epsilon
            ub_b_4 = result.ub_breakpoints[AlgType.APPROX4]
            ub_b_8 = result.ub_breakpoints[AlgType.APPROX8]
            b_exact = result.alg_outputs[AlgType.EXACT].num_breakpoints
            b_4 = result.alg_outputs[AlgType.APPROX4].num_breakpoints
            b_8 = result.alg_outputs[AlgType.APPROX8].num_breakpoints
            e_exact = result.alg_outputs[AlgType.EXACT].error
            e_4 = result.alg_outputs[AlgType.APPROX4].error
            e_8 = result.alg_outputs[AlgType.APPROX8].error
            interval_digit = 2
            if exp.rv_type == "discrete":
                a, b = int(a), int(b)
            elif exp.rv_type == "continuous":
                a, b = round(a, interval_digit), round(b, interval_digit)
            row = [
                exp.case_name,
                exp.dist_name,
                f"${exp.dist_param}$",
                a,
                b,
                rv_type,
                epsilon,
                b_exact,
                b_8,
                b_4,
                ub_b_8,
                ub_b_4,
                round(e_exact, 4),
                round(e_4, 4),
                round(e_8, 4),
            ]
            rows.append(row)
    cols = pd.MultiIndex.from_tuples(
        [
            ("No.", ""),
            ("distribution", ""),
            ("parameter", ""),
            ("$a$", ""),
            ("$b$", ""),
            ("rv-type", ""),
            ("$\epsilon$", ""),
            ("breakpoints", r"$B_{\text{exact}}$"),
            ("breakpoints", r"$B_{1/8}$"),
            ("breakpoints", r"$B_{1/4}$"),
            ("breakpoints", r"$UB_{1/8}$"),
            ("breakpoints", r"$UB_{1/4}$"),
            ("error", r"$B_{\text{exact}}$"),
            ("error", r"$B_{1/4}$"),
            ("error", r"$B_{1/8}$"),
        ]
    )

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("result.csv", index=False)
    latex_code = df[["No.", "$\epsilon$", "breakpoints", "error"]].to_latex(
        index=False, float_format="%.3f", column_format="lr|rrrrr|rrr"
    )
    modified_latex_code = modify_latex_table(latex_code)
    with open("result.tex", "w") as file:
        file.write(modified_latex_code)

    # for latex of dataset
    int_format = lambda x: f"{x:.0f}"
    float_format = lambda x: f"{x:.2f}"

    # Applying format for each column
    df_dataset = df[["No.", "distribution", "parameter", "$a$", "$b$"]].drop_duplicates()
    df_dataset.to_csv("dataset.csv", index=False)
    df_dataset.to_latex("dataset.tex", index=False, float_format="%.1f")


if __name__ == "__main__":
    main()
