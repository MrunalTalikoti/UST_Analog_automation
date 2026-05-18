#!/usr/bin/env python3
"""
Inverse Design Optimizer v4 – Improved
=======================================
Key improvements over v3:
  1. UGF uses XGB + GBR ensemble in log-space (lower MAPE)
  2. Loss is normalised per output range → no UGF domination
  3. Two-phase optimisation: Optuna-TPE  +  Nelder-Mead local refinement
  4. Stagnation detection retained
"""

import numpy as np
import pickle
import json
import optuna
from pathlib import Path
from scipy.optimize import minimize
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

        print("Loading parameter bounds and models...")
        with open(self.models_path / "param_bounds.pkl", "rb") as f:
            self.bounds = pickle.load(f)
        self.feature_names = list(self.bounds.keys())  # ['a','b','c','d']

        # Feature log-sampling flags
        self.feature_config = {
            "a": {"log": True},
            "b": {"log": False},
            "c": {"log": True},
            "d": {"log": True},
        }

        # Load GAIN and PM models
        with open(self.models_path / "model_gain_xgboost.pkl", "rb") as f:
            self.model_gain = pickle.load(f)
        with open(self.models_path / "model_pm_xgboost.pkl", "rb") as f:
            self.model_pm = pickle.load(f)

        # Load UGF ensemble models
        with open(self.models_path / "model_ugf_xgboost.pkl", "rb") as f:
            self.model_ugf_xgb = pickle.load(f)

        ugf_gbr_path = self.models_path / "model_ugf_gbr.pkl"
        if ugf_gbr_path.exists():
            with open(ugf_gbr_path, "rb") as f:
                self.model_ugf_gbr = pickle.load(f)
            blend_path = self.models_path / "ugf_blend_config.json"
            if blend_path.exists():
                with open(blend_path) as f:
                    blend_cfg = json.load(f)
                self.ugf_blend_w = blend_cfg.get("ugf_blend_xgb_weight", 0.65)
                self.ugf_log = blend_cfg.get("ugf_log_transform", True)
            else:
                self.ugf_blend_w = 0.65
                self.ugf_log = True
            self.use_ugf_ensemble = True
            print("  ✓ UGF ensemble loaded (XGB + GBR blend)")
        else:
            # Fallback: single XGBoost model
            self.model_ugf_gbr = None
            self.use_ugf_ensemble = False
            self.ugf_log = False
            print("  ⚠  UGF GBR model not found – using single XGB")

        # Derive output normalisation ranges from bounds
        # (approximate; will be refined if data is available)
        self.GAIN_RANGE = 37.0   # dB
        self.PM_RANGE   = 40.0   # degrees
        self.UGF_RANGE  = 125.0  # MHz

        # Targets
        self.target_gain = None
        self.target_pm   = None
        self.target_ugf  = None

        # Best solution tracking
        self.best_params = None
        self.best_loss   = float("inf")
        self.trials_since_improvement = 0

        print("✓ Optimizer ready\n")

    # ── Prediction helpers ────────────────────────────────────────────────────
    def predict_ugf(self, X):
        if self.use_ugf_ensemble:
            log_xgb = self.model_ugf_xgb.predict(X)
            log_gbr = self.model_ugf_gbr.predict(X)
            return np.exp(self.ugf_blend_w * log_xgb + (1 - self.ugf_blend_w) * log_gbr)
        else:
            raw = self.model_ugf_xgb.predict(X)
            return np.exp(raw) if self.ugf_log else raw

    def forward(self, x_1d):
        """x_1d: shape (4,)  → returns (gain, pm, ugf)"""
        X = x_1d.reshape(1, -1)
        gain = self.model_gain.predict(X)[0]
        pm   = self.model_pm.predict(X)[0]
        ugf  = self.predict_ugf(X)[0]
        return gain, pm, ugf

    # ── Normalised loss ───────────────────────────────────────────────────────
    def normalised_loss(self, x_1d):
        gain, pm, ugf = self.forward(x_1d)
        loss = (
            abs(gain - self.target_gain) / self.GAIN_RANGE +
            abs(pm   - self.target_pm)   / self.PM_RANGE   +
            abs(ugf  - self.target_ugf)  / self.UGF_RANGE
        )
        return loss, gain, pm, ugf

    # ── Optuna objective ─────────────────────────────────────────────────────
    def objective(self, trial):
        x = []
        for f in self.feature_names:
            lo, hi = self.bounds[f]
            if self.feature_config[f]["log"]:
                val = trial.suggest_float(f, lo, hi, log=True)
            else:
                val = trial.suggest_float(f, lo, hi)
            x.append(val)

        x = np.array(x)
        loss, gain, pm, ugf = self.normalised_loss(x)

        eg = abs(gain - self.target_gain) / abs(self.target_gain) * 100
        ep = abs(pm   - self.target_pm)   / abs(self.target_pm)   * 100
        eu = abs(ugf  - self.target_ugf)  / abs(self.target_ugf)  * 100
        max_error = max(eg, ep, eu)

        self.trials_since_improvement += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.trials_since_improvement = 0
            self.best_params = {
                "a": x[0], "b": x[1], "c": x[2], "d": x[3],
                "pred_gain": gain, "pred_pm": pm, "pred_ugf": ugf,
                "loss": loss,
                "error_gain": eg, "error_pm": ep, "error_ugf": eu,
                "max_error": max_error,
            }
        return loss

    # ── Main optimise ─────────────────────────────────────────────────────────
    def optimize(self, target_gain, target_pm, target_ugf,
                 convergence_threshold=1.0,
                 max_trials=10000,
                 stagnation_trials=600):

        self.target_gain = target_gain
        self.target_pm   = target_pm
        self.target_ugf  = target_ugf
        self.best_loss   = float("inf")
        self.best_params = None
        self.trials_since_improvement = 0

        lo = np.array([self.bounds[k][0] for k in self.feature_names])
        hi = np.array([self.bounds[k][1] for k in self.feature_names])

        print("=" * 70)
        print("INVERSE DESIGN OPTIMISER v4 (Normalised Loss + Nelder-Mead)")
        print("=" * 70)
        print(f"Target Gain: {target_gain}")
        print(f"Target PM:   {target_pm}")
        print(f"Target UGF:  {target_ugf}")
        print(f"Convergence: {convergence_threshold}% max error")
        print("=" * 70)

        # ── Phase 1: Optuna TPE ───────────────────────────────────────────────
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
        )

        trial_count = 0
        batch_size  = 50

        while trial_count < max_trials:
            study.optimize(self.objective, n_trials=batch_size, show_progress_bar=False)
            trial_count += batch_size

            if self.best_params:
                print(f"Trial {trial_count:5d}: max_err = {self.best_params['max_error']:.4f}%  "
                      f"loss = {self.best_params['loss']:.6f}  "
                      f"stagnation = {self.trials_since_improvement}")

            if self.best_params and self.best_params["max_error"] < convergence_threshold:
                print(f"\n✅ CONVERGED at trial {trial_count}")
                break

            if self.trials_since_improvement >= stagnation_trials:
                print(f"\n⚠️  Stagnation after {stagnation_trials} trials — proceeding to Nelder-Mead")
                break

        # ── Phase 2: Nelder-Mead local refinement ────────────────────────────
        if self.best_params:
            x0 = np.array([self.best_params[k] for k in self.feature_names])
            print("\nRunning Nelder-Mead local refinement...")

            def scipy_obj(x):
                x = np.clip(x, lo, hi)
                loss, _, _, _ = self.normalised_loss(x)
                return loss

            res = minimize(scipy_obj, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-12})
            x_refined = np.clip(res.x, lo, hi)
            _, gain_r, pm_r, ugf_r = self.normalised_loss(x_refined)
            eg_r = abs(gain_r - target_gain) / abs(target_gain) * 100
            ep_r = abs(pm_r   - target_pm)   / abs(target_pm)   * 100
            eu_r = abs(ugf_r  - target_ugf)  / abs(target_ugf)  * 100
            max_r = max(eg_r, ep_r, eu_r)

            if max_r < self.best_params["max_error"]:
                print(f"  Nelder-Mead improved max_error: "
                      f"{self.best_params['max_error']:.4f}% → {max_r:.4f}%")
                self.best_params = {
                    "a": x_refined[0], "b": x_refined[1],
                    "c": x_refined[2], "d": x_refined[3],
                    "pred_gain": gain_r, "pred_pm": pm_r, "pred_ugf": ugf_r,
                    "loss": res.fun,
                    "error_gain": eg_r, "error_pm": ep_r, "error_ugf": eu_r,
                    "max_error": max_r,
                }
            else:
                print(f"  Nelder-Mead did not improve (max_err stayed at "
                      f"{self.best_params['max_error']:.4f}%)")

        print("=" * 70)
        return self.best_params

    # ── Display ───────────────────────────────────────────────────────────────
    def display_results(self, results):
        print("\nOPTIMISATION RESULTS")
        print("=" * 70)
        print("\n📊 OPTIMAL PARAMETERS:")
        for k in self.feature_names:
            print(f"  {k} = {results[k]:.10e} m")
        print("\n🎯 PREDICTED vs TARGET:")
        print(f"  Gain : {results['pred_gain']:.6f}  (Target: {self.target_gain:.6f}, Error: {results['error_gain']:.4f}%)")
        print(f"  PM   : {results['pred_pm']:.6f}  (Target: {self.target_pm:.6f}, Error: {results['error_pm']:.4f}%)")
        print(f"  UGF  : {results['pred_ugf']:.6f}  (Target: {self.target_ugf:.6f}, Error: {results['error_ugf']:.4f}%)")
        status = "✅" if results["max_error"] < 1.0 else "⚠️ "
        print(f"\n{status}  Maximum Error: {results['max_error']:.4f}%")
        print("=" * 70)

    # ── Netlist generation ────────────────────────────────────────────────────
    def generate_netlist(self, results, output_path=None):
        with open(self.netlist_template_path, "r") as f:
            template = f.read()

        a_s = f"{results['a']:.6e}"; b_s = f"{results['b']:.6e}"
        c_s = f"{results['c']:.6e}"; d_s = f"{results['d']:.6e}"

        netlist = template.replace("W=a ", f"W={a_s} ")
        netlist = netlist.replace("W=b ", f"W={b_s} ")
        netlist = netlist.replace("W=c ", f"W={c_s} ")
        netlist = netlist.replace("W=d ", f"W={d_s} ")

        header = f"""* GENERATED BY INVERSE DESIGN OPTIMIZER v4
* Target: Gain={self.target_gain:.4f}  PM={self.target_pm:.4f}  UGF={self.target_ugf:.4f}
* Optimised: Gain={results['pred_gain']:.4f}({results['error_gain']:.3f}%)  PM={results['pred_pm']:.4f}({results['error_pm']:.3f}%)  UGF={results['pred_ugf']:.4f}({results['error_ugf']:.3f}%)
* Max Error: {results['max_error']:.4f}%
*
"""
        netlist = header + netlist
        out_file = Path(output_path) if output_path else Path("optimized_opamp180nm.cir")
        with open(out_file, "w") as f:
            f.write(netlist)
        print(f"\n💾 Netlist saved to: {out_file.absolute()}")
        return netlist


def main():
    parser = argparse.ArgumentParser(description="Inverse Design Optimiser v4")
    parser.add_argument("--gain",       type=float, required=True)
    parser.add_argument("--pm",         type=float, required=True)
    parser.add_argument("--ugf",        type=float, required=True)
    parser.add_argument("--threshold",  type=float, default=1.0)
    parser.add_argument("--stagnation", type=int,   default=200)
    parser.add_argument("--max-trials", type=int,   default=10000)
    parser.add_argument("--output",     type=str,   default=None)
    parser.add_argument("--base-path",  type=str,   default=None)
    args = parser.parse_args()

    try:
        opt = InverseDesignOptimizer(base_path=args.base_path)
        results = opt.optimize(
            target_gain=args.gain, target_pm=args.pm, target_ugf=args.ugf,
            convergence_threshold=args.threshold,
            max_trials=args.max_trials,
            stagnation_trials=args.stagnation,
        )
        if results is None:
            print("❌ No solution found"); sys.exit(1)
        opt.display_results(results)
        opt.generate_netlist(results, output_path=args.output)
        if results["max_error"] < args.threshold:
            print("\n✅ SUCCESS")
        else:
            print(f"\n⚠️  PARTIAL – best error {results['max_error']:.4f}%  (threshold {args.threshold}%)")
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
