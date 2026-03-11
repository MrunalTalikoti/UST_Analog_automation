#!/usr/bin/env python3

import numpy as np
import pickle
import optuna
from pathlib import Path
import argparse
import sys

optuna.logging.set_verbosity(optuna.logging.WARNING)


class InverseDesignOptimizer:

    def __init__(self, base_path=None):

        if base_path is None:
            base_path = Path(__file__).parent
        else:
            base_path = Path(base_path)

        self.models_path = base_path / "trained_models"
        self.netlist_template_path = base_path / "opamp180nm"

        print("Loading parameter bounds...")

        with open(self.models_path / "param_bounds.pkl", "rb") as f:
            self.bounds = pickle.load(f)

        self.feature_names = list(self.bounds.keys())

        self.feature_config = {
            "a": {"log": True},
            "b": {"log": False},
            "c": {"log": True},
            "d": {"log": True},
        }

        print("Loading trained models...")

        with open(self.models_path / "model_gain_xgboost.pkl", "rb") as f:
            self.model_gain = pickle.load(f)

        with open(self.models_path / "model_pm_xgboost.pkl", "rb") as f:
            self.model_pm = pickle.load(f)

        with open(self.models_path / "model_ugf_xgboost.pkl", "rb") as f:
            self.model_ugf = pickle.load(f)

        print("✓ Models loaded successfully\n")

        self.target_gain = None
        self.target_pm = None
        self.target_ugf = None

        self.best_params = None
        self.best_loss = float("inf")
        self.trials_since_improvement = 0


    def inverse_loss(self, x):

        gain = self.model_gain.predict(x)[0]
        pm = self.model_pm.predict(x)[0]
        ugf = self.model_ugf.predict(x)[0]

        # percentage-based loss (better scaling)
        loss = (
            abs(gain - self.target_gain) / abs(self.target_gain) +
            abs(pm - self.target_pm) / abs(self.target_pm) +
            abs(ugf - self.target_ugf) / abs(self.target_ugf)
        )

        return loss, gain, pm, ugf


    def objective(self, trial):

        x = []

        for f in self.feature_names:

            low, high = self.bounds[f]

            if self.feature_config[f]["log"]:
                val = trial.suggest_float(f, low, high, log=True)
            else:
                val = trial.suggest_float(f, low, high)

            x.append(val)

        x = np.array(x).reshape(1, -1)

        loss, pred_gain, pred_pm, pred_ugf = self.inverse_loss(x)

        error_gain = abs(pred_gain - self.target_gain) / abs(self.target_gain) * 100
        error_pm = abs(pred_pm - self.target_pm) / abs(self.target_pm) * 100
        error_ugf = abs(pred_ugf - self.target_ugf) / abs(self.target_ugf) * 100

        max_error = max(error_gain, error_pm, error_ugf)

        self.trials_since_improvement += 1

        if loss < self.best_loss:

            self.best_loss = loss
            self.trials_since_improvement = 0

            self.best_params = {
                "a": x[0][0],
                "b": x[0][1],
                "c": x[0][2],
                "d": x[0][3],
                "pred_gain": pred_gain,
                "pred_pm": pred_pm,
                "pred_ugf": pred_ugf,
                "loss": loss,
                "error_gain": error_gain,
                "error_pm": error_pm,
                "error_ugf": error_ugf,
                "max_error": max_error
            }

        return loss


    def refine_solution(self):

        if not self.best_params:
            return

        print("\nRunning local refinement...")

        step = 0.05  # 5%

        for name in ["a", "b", "c", "d"]:

            base = self.best_params[name]

            for factor in [1 - step, 1 + step]:

                test_params = self.best_params.copy()
                test_params[name] = base * factor

                x = np.array([[test_params["a"],
                               test_params["b"],
                               test_params["c"],
                               test_params["d"]]])

                loss, gain, pm, ugf = self.inverse_loss(x)

                if loss < self.best_loss:

                    error_gain = abs(gain - self.target_gain) / abs(self.target_gain) * 100
                    error_pm = abs(pm - self.target_pm) / abs(self.target_pm) * 100
                    error_ugf = abs(ugf - self.target_ugf) / abs(self.target_ugf) * 100

                    max_error = max(error_gain, error_pm, error_ugf)

                    self.best_loss = loss

                    self.best_params.update({
                        name: test_params[name],
                        "pred_gain": gain,
                        "pred_pm": pm,
                        "pred_ugf": ugf,
                        "loss": loss,
                        "error_gain": error_gain,
                        "error_pm": error_pm,
                        "error_ugf": error_ugf,
                        "max_error": max_error
                    })


    def optimize(self, target_gain, target_pm, target_ugf,
                 convergence_threshold=1.0,
                 max_trials=20000,
                 stagnation_trials=500):

        self.target_gain = target_gain
        self.target_pm = target_pm
        self.target_ugf = target_ugf

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=500,
            multivariate=True
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        trial_count = 0
        batch_size = 50

        while trial_count < max_trials:

            study.optimize(self.objective, n_trials=batch_size)

            trial_count += batch_size

            if self.best_params:

                print(
                    f"Trial {trial_count} | "
                    f"Best error = {self.best_params['max_error']:.4f}%"
                )

                if self.best_params["max_error"] < convergence_threshold:
                    break

            if self.trials_since_improvement >= stagnation_trials:
                break

        # local refinement
        self.refine_solution()

        return self.best_params


    def display_results(self, results):

        print("\nOPTIMIZATION RESULTS")
        print("=" * 60)

        print("\nOptimal Parameters:")
        print(f"a = {results['a']:.10e}")
        print(f"b = {results['b']:.10e}")
        print(f"c = {results['c']:.10e}")
        print(f"d = {results['d']:.10e}")

        print("\nPredicted vs Target")

        print(
            f"Gain: {results['pred_gain']:.6f} "
            f"(target {self.target_gain})"
        )

        print(
            f"PM: {results['pred_pm']:.6f} "
            f"(target {self.target_pm})"
        )

        print(
            f"UGF: {results['pred_ugf']:.6f} "
            f"(target {self.target_ugf})"
        )

        print(f"\nMax Error: {results['max_error']:.4f}%")
        print("=" * 60)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--pm", type=float, required=True)
    parser.add_argument("--ugf", type=float, required=True)

    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--max-trials", type=int, default=20000)
    parser.add_argument("--stagnation", type=int, default=500)

    args = parser.parse_args()

    optimizer = InverseDesignOptimizer()

    results = optimizer.optimize(
        args.gain,
        args.pm,
        args.ugf,
        args.threshold,
        args.max_trials,
        args.stagnation
    )

    optimizer.display_results(results)


if __name__ == "__main__":
    main()