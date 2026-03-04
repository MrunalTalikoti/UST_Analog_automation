#!/usr/bin/env python3
"""
Inverse Design Optimizer for Analog Circuit (v3)
Stops when no improvement for 200 trials (avoids infinite loops)
"""

import numpy as np
import pickle
import optuna
from pathlib import Path
import argparse
import sys

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class InverseDesignOptimizer:
    def __init__(self, base_path=None):
        """Initialize optimizer with models and configuration"""
        if base_path is None:
            # Auto-detect: assume script is in new_data_exp folder
            base_path = Path(__file__).parent
        else:
            base_path = Path(base_path)
        
        self.models_path = base_path / "trained_models"
        self.netlist_template_path = base_path / "opamp180nm"
        
        # Load parameter bounds
        print("Loading parameter bounds...")
        with open(self.models_path / "param_bounds.pkl", "rb") as f:
            self.bounds = pickle.load(f)
        
        self.feature_names = list(self.bounds.keys())  # ['a','b','c','d']
        
        # Log sampling config (matching inverse.ipynb)
        self.feature_config = {
            "a": {"log": True},
            "b": {"log": False},  # includes 0, so no log
            "c": {"log": True},
            "d": {"log": True},
        }
        
        # Load trained forward models
        print("Loading trained models...")
        with open(self.models_path / "model_gain_xgboost.pkl", "rb") as f:
            self.model_gain = pickle.load(f)
        with open(self.models_path / "model_pm_xgboost.pkl", "rb") as f:
            self.model_pm = pickle.load(f)
        with open(self.models_path / "model_ugf_xgboost.pkl", "rb") as f:
            self.model_ugf = pickle.load(f)
        
        print("✓ Models and bounds loaded successfully\n")
        
        # Targets
        self.target_gain = None
        self.target_pm = None
        self.target_ugf = None
        
        # Best solution tracking
        self.best_params = None
        self.best_loss = float('inf')
        self.trials_since_improvement = 0
        
    def inverse_loss(self, x):
        """
        Calculate inverse design loss
        x shape: (1, 4)
        """
        gain = self.model_gain.predict(x)[0]
        pm = self.model_pm.predict(x)[0]
        ugf = self.model_ugf.predict(x)[0]
        
        # Sum of absolute errors (matching inverse.ipynb)
        loss = (
            abs(gain - self.target_gain) +
            abs(pm - self.target_pm) +
            abs(ugf - self.target_ugf)
        )
        return loss, gain, pm, ugf
    
    def objective(self, trial):
        """Optuna objective function"""
        x = []
        
        # Suggest parameters with appropriate sampling
        for f in self.feature_names:
            low, high = self.bounds[f]
            
            if self.feature_config[f]["log"]:
                val = trial.suggest_float(f, low, high, log=True)
            else:
                val = trial.suggest_float(f, low, high)
            
            x.append(val)
        
        x = np.array(x).reshape(1, -1)
        loss, pred_gain, pred_pm, pred_ugf = self.inverse_loss(x)
        
        # Calculate percentage errors
        error_gain = abs(pred_gain - self.target_gain) / abs(self.target_gain) * 100
        error_pm = abs(pred_pm - self.target_pm) / abs(self.target_pm) * 100
        error_ugf = abs(pred_ugf - self.target_ugf) / abs(self.target_ugf) * 100
        max_error = max(error_gain, error_pm, error_ugf)
        
        # Track improvements
        self.trials_since_improvement += 1
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.trials_since_improvement = 0  # Reset counter on improvement
            self.best_params = {
                'a': x[0][0],
                'b': x[0][1],
                'c': x[0][2],
                'd': x[0][3],
                'pred_gain': pred_gain,
                'pred_pm': pred_pm,
                'pred_ugf': pred_ugf,
                'loss': loss,
                'error_gain': error_gain,
                'error_pm': error_pm,
                'error_ugf': error_ugf,
                'max_error': max_error
            }
        
        return loss
    
    def optimize(self, target_gain, target_pm, target_ugf, 
                 convergence_threshold=1.0, 
                 max_trials=10000,
                 stagnation_trials=200):
        """
        Run optimization with stagnation detection
        
        Args:
            target_gain: Target gain value
            target_pm: Target phase margin
            target_ugf: Target unity gain frequency
            convergence_threshold: Stop when max error < this % (default: 1.0)
            max_trials: Maximum number of trials (default: 10000)
            stagnation_trials: Stop if no improvement for this many trials (default: 200)
        """
        self.target_gain = target_gain
        self.target_pm = target_pm
        self.target_ugf = target_ugf
        self.best_loss = float('inf')
        self.best_params = None
        self.trials_since_improvement = 0
        
        print("="*70)
        print("INVERSE DESIGN OPTIMIZATION v3 (Anti-Stagnation)")
        print("="*70)
        print(f"Target Gain: {target_gain}")
        print(f"Target PM:   {target_pm}")
        print(f"Target UGF:  {target_ugf}")
        print(f"Convergence Threshold: {convergence_threshold}% (max error)")
        print(f"Stagnation Detection: Stop if no improvement for {stagnation_trials} trials")
        print("="*70)
        print(f"\nOptimizing...\n")
        
        # Create study with TPE sampler
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        
        # Optimize with stagnation detection
        trial_count = 0
        batch_size = 50
        
        while trial_count < max_trials:
            study.optimize(self.objective, n_trials=batch_size, show_progress_bar=False)
            trial_count += batch_size
            
            # Print progress
            if self.best_params:
                print(f"Trial {trial_count}: Best max error = {self.best_params['max_error']:.4f}% | "
                      f"Loss = {self.best_params['loss']:.6f} | "
                      f"No improvement: {self.trials_since_improvement} trials")
            
            # Check convergence (SUCCESS!)
            if self.best_params and self.best_params['max_error'] < convergence_threshold:
                print(f"\n✅ CONVERGED after {trial_count} trials!")
                print(f"   Max error ({self.best_params['max_error']:.4f}%) < threshold ({convergence_threshold}%)")
                break
            
            # Check stagnation (STUCK - STOP)
            if self.trials_since_improvement >= stagnation_trials:
                print(f"\n⚠️  STOPPED: No improvement for {stagnation_trials} trials (likely stuck in local minimum)")
                print(f"   Current best max error: {self.best_params['max_error']:.4f}%")
                print(f"   Total trials: {trial_count}")
                break
        
        # Check if hit max trials
        if trial_count >= max_trials:
            print(f"\n⚠️  STOPPED: Reached maximum trials ({max_trials})")
            if self.best_params:
                print(f"   Best max error found: {self.best_params['max_error']:.4f}%")
        
        print("="*70)
        return self.best_params
    
    def display_results(self, results):
        """Display optimization results"""
        print("\nOPTIMIZATION RESULTS")
        print("="*70)
        print("\n📊 OPTIMAL PARAMETERS:")
        print(f"  a = {results['a']:.10e} m")
        print(f"  b = {results['b']:.10e} m")
        print(f"  c = {results['c']:.10e} m")
        print(f"  d = {results['d']:.10e} m")
        
        print("\n🎯 PREDICTED vs TARGET OUTPUTS:")
        print(f"  Gain: {results['pred_gain']:.6f} (Target: {self.target_gain:.6f}, Error: {results['error_gain']:.4f}%)")
        print(f"  PM:   {results['pred_pm']:.6f} (Target: {self.target_pm:.6f}, Error: {results['error_pm']:.4f}%)")
        print(f"  UGF:  {results['pred_ugf']:.6f} (Target: {self.target_ugf:.6f}, Error: {results['error_ugf']:.4f}%)")
        
        print(f"\n{'✅' if results['max_error'] < 1.0 else '⚠️'}  Maximum Error: {results['max_error']:.4f}%")
        print(f"   Total Loss: {results['loss']:.6f}")
        print("="*70)
    
    def generate_netlist(self, results, output_path=None):
        """Generate SPICE netlist with optimized parameters"""
        # Read template
        with open(self.netlist_template_path, 'r') as f:
            template = f.read()
        
        # Format values in scientific notation
        a_str = f"{results['a']:.6e}"
        b_str = f"{results['b']:.6e}"
        c_str = f"{results['c']:.6e}"
        d_str = f"{results['d']:.6e}"
        
        # Replace placeholders (W=a, W=b, W=c, W=d)
        netlist = template.replace('W=a ', f'W={a_str} ')
        netlist = netlist.replace('W=b ', f'W={b_str} ')
        netlist = netlist.replace('W=c ', f'W={c_str} ')
        netlist = netlist.replace('W=d ', f'W={d_str} ')
        
        # Add header with optimization info
        header = f"""* ============================================================
* GENERATED BY INVERSE DESIGN OPTIMIZER v3
* ============================================================
* Target Specifications:
*   Gain = {self.target_gain:.6f}
*   PM   = {self.target_pm:.6f}
*   UGF  = {self.target_ugf:.6f}
*
* Optimized Parameters:
*   a = {a_str} m (Width of differential input pair)
*   b = {b_str} m (Width of PMOS active load)
*   c = {c_str} m (Width of NMOS current mirror)
*   d = {d_str} m (Width of PMOS output stage)
*
* Predicted Performance:
*   Gain = {results['pred_gain']:.6f} (Error: {results['error_gain']:.4f}%)
*   PM   = {results['pred_pm']:.6f} (Error: {results['error_pm']:.4f}%)
*   UGF  = {results['pred_ugf']:.6f} (Error: {results['error_ugf']:.4f}%)
*
* Maximum Error: {results['max_error']:.4f}%
* Total Loss: {results['loss']:.6f}
* ============================================================
*
"""
        netlist = header + netlist
        
        # Display netlist (first 30 lines only to avoid clutter)
        print("\n📄 GENERATED NETLIST (preview):")
        print("="*70)
        lines = netlist.split('\n')
        print('\n'.join(lines[:30]))
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
        print("="*70)
        
        # Save to file
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = Path("optimized_opamp180nm.cir")
        
        with open(output_file, 'w') as f:
            f.write(netlist)
        print(f"\n💾 Full netlist saved to: {output_file.absolute()}")
        
        return netlist


def main():
    parser = argparse.ArgumentParser(
        description='Inverse Design Optimizer v3 - With stagnation detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (stops automatically if stuck)
  python inverse_optimizer_v3.py --gain 18.742711 --pm 87.258927 --ugf 12.859450
  
  # Custom stagnation threshold (stop after 500 trials with no improvement)
  python inverse_optimizer_v3.py --gain 20.0 --pm 85.0 --ugf 15.0 --stagnation 500
  
  # Tighter convergence requirement
  python inverse_optimizer_v3.py --gain 20.0 --pm 85.0 --ugf 15.0 --threshold 0.5
  
  # All options
  python inverse_optimizer_v3.py --gain 20.0 --pm 85.0 --ugf 15.0 --threshold 1.0 --stagnation 200 --max-trials 20000
        """
    )
    
    parser.add_argument('--gain', type=float, required=True, help='Target gain value')
    parser.add_argument('--pm', type=float, required=True, help='Target phase margin')
    parser.add_argument('--ugf', type=float, required=True, help='Target unity gain frequency')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Convergence threshold (%% max error, default: 1.0)')
    parser.add_argument('--stagnation', type=int, default=200,
                       help='Stop if no improvement for this many trials (default: 200)')
    parser.add_argument('--max-trials', type=int, default=10000,
                       help='Maximum number of trials (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output netlist filename (default: optimized_opamp180nm.cir)')
    parser.add_argument('--base-path', type=str, default=None,
                       help='Base directory containing trained_models/ and opamp180nm')
    
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = InverseDesignOptimizer(base_path=args.base_path)
        
        # Run optimization
        results = optimizer.optimize(
            target_gain=args.gain,
            target_pm=args.pm,
            target_ugf=args.ugf,
            convergence_threshold=args.threshold,
            max_trials=args.max_trials,
            stagnation_trials=args.stagnation
        )
        
        if results is None:
            print("❌ Optimization failed to find any solution")
            sys.exit(1)
        
        # Display results
        optimizer.display_results(results)
        
        # Generate netlist
        optimizer.generate_netlist(results, output_path=args.output)
        
        # Final summary
        if results['max_error'] < args.threshold:
            print("\n✅ SUCCESS: Target specifications achieved!")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Best error {results['max_error']:.4f}% (target was {args.threshold}%)")
            print("   Consider:")
            print("   - Relaxing the convergence threshold: --threshold 2.0")
            print("   - Checking if target specs are physically achievable")
            print("   - Increasing stagnation trials: --stagnation 500")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files")
        print(f"   {e}")
        print("\nMake sure you're running from the correct directory:")
        print("   cd \"D:\\UST Project\\UST_Analog_automation\\notebooks\\new_data_exp\"")
        print("Or use --base-path to specify the location")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        if hasattr(optimizer, 'best_params') and optimizer.best_params:
            print("\nDisplaying best solution found so far...")
            optimizer.display_results(optimizer.best_params)
            optimizer.generate_netlist(optimizer.best_params, output_path=args.output)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()