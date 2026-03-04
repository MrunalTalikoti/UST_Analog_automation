#!/usr/bin/env python3
"""
Inverse Design Optimizer for Analog Circuit
Takes target gain, pm, ugf and finds optimal a,b,c,d parameters
Generates SPICE netlist with optimized values
"""

import pickle
import numpy as np
import optuna
from pathlib import Path
import argparse
import sys

# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class InverseDesignOptimizer:
    def __init__(self, models_path, param_bounds_path, netlist_template_path):
        """Initialize optimizer with models and configuration"""
        self.models_path = Path(models_path)
        self.netlist_template_path = Path(netlist_template_path)
        
        # Load trained models
        print("Loading trained models...")
        with open(self.models_path / "model_gain_xgboost.pkl", "rb") as f:
            self.model_gain = pickle.load(f)
        with open(self.models_path / "model_pm_xgboost.pkl", "rb") as f:
            self.model_pm = pickle.load(f)
        with open(self.models_path / "model_ugf_xgboost.pkl", "rb") as f:
            self.model_ugf = pickle.load(f)
        
        # Load parameter bounds
        print("Loading parameter bounds...")
        with open(param_bounds_path, "rb") as f:
            self.param_bounds = pickle.load(f)
        
        print("✓ Models and bounds loaded successfully\n")
        
        # Target values (will be set during optimization)
        self.target_gain = None
        self.target_pm = None
        self.target_ugf = None
        
        # Best solution found
        self.best_params = None
        self.best_error = float('inf')
        
    def objective(self, trial):
        """Optuna objective function - minimize maximum percentage error"""
        # Suggest values for a, b, c, d within bounds
        a = trial.suggest_float('a', self.param_bounds['a'][0], self.param_bounds['a'][1])
        b = trial.suggest_float('b', self.param_bounds['b'][0], self.param_bounds['b'][1])
        c = trial.suggest_float('c', self.param_bounds['c'][0], self.param_bounds['c'][1])
        d = trial.suggest_float('d', self.param_bounds['d'][0], self.param_bounds['d'][1])
        
        # Create input array
        X = np.array([[a, b, c, d]])
        
        # Predict outputs using the three models
        pred_gain = self.model_gain.predict(X)[0]
        pred_pm = self.model_pm.predict(X)[0]
        pred_ugf = self.model_ugf.predict(X)[0]
        
        # Calculate percentage errors
        error_gain = abs(pred_gain - self.target_gain) / abs(self.target_gain) * 100
        error_pm = abs(pred_pm - self.target_pm) / abs(self.target_pm) * 100
        error_ugf = abs(pred_ugf - self.target_ugf) / abs(self.target_ugf) * 100
        
        # Use maximum error as objective (minimax optimization)
        max_error = max(error_gain, error_pm, error_ugf)
        
        # Update best solution if this is better
        if max_error < self.best_error:
            self.best_error = max_error
            self.best_params = {
                'a': a, 'b': b, 'c': c, 'd': d,
                'pred_gain': pred_gain,
                'pred_pm': pred_pm,
                'pred_ugf': pred_ugf,
                'error_gain': error_gain,
                'error_pm': error_pm,
                'error_ugf': error_ugf,
                'max_error': max_error
            }
        
        return max_error
    
    def optimize(self, target_gain, target_pm, target_ugf, convergence_threshold=1.0):
        """
        Run optimization to find a,b,c,d that produce target outputs
        
        Args:
            target_gain: Target gain value
            target_pm: Target phase margin value
            target_ugf: Target unity gain frequency value
            convergence_threshold: Stop when max error < this percentage (default 1%)
        """
        self.target_gain = target_gain
        self.target_pm = target_pm
        self.target_ugf = target_ugf
        self.best_error = float('inf')
        self.best_params = None
        
        print("="*70)
        print("INVERSE DESIGN OPTIMIZATION")
        print("="*70)
        print(f"Target Gain: {target_gain}")
        print(f"Target PM:   {target_pm}")
        print(f"Target UGF:  {target_ugf}")
        print(f"Convergence Threshold: {convergence_threshold}%")
        print("="*70)
        print("\nOptimizing... (will stop when error < {:.2f}%)".format(convergence_threshold))
        
        # Create study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        
        # Optimize until convergence
        trial_count = 0
        while self.best_error > convergence_threshold:
            study.optimize(self.objective, n_trials=50, show_progress_bar=False)
            trial_count += len(study.trials)
            
            # Print progress every 50 trials
            if trial_count % 50 == 0:
                print(f"Trial {trial_count}: Best error = {self.best_error:.4f}%")
            
            # Safety check to prevent infinite loop
            if trial_count > 10000:
                print(f"\n⚠ Warning: Reached 10000 trials. Current best error: {self.best_error:.4f}%")
                break
        
        print(f"\n✓ Converged after {trial_count} trials!")
        print("="*70)
        
        return self.best_params
    
    def display_results(self, results):
        """Display optimization results in a formatted way"""
        print("\nOPTIMIZATION RESULTS")
        print("="*70)
        print("\n📊 OPTIMAL PARAMETERS:")
        print(f"  a = {results['a']:.6e}")
        print(f"  b = {results['b']:.6e}")
        print(f"  c = {results['c']:.6e}")
        print(f"  d = {results['d']:.6e}")
        
        print("\n🎯 PREDICTED OUTPUTS:")
        print(f"  Gain = {results['pred_gain']:.4f}  (Target: {self.target_gain:.4f}, Error: {results['error_gain']:.4f}%)")
        print(f"  PM   = {results['pred_pm']:.4f}  (Target: {self.target_pm:.4f}, Error: {results['error_pm']:.4f}%)")
        print(f"  UGF  = {results['pred_ugf']:.4e}  (Target: {self.target_ugf:.4e}, Error: {results['error_ugf']:.4f}%)")
        
        print(f"\n✅ Maximum Error: {results['max_error']:.4f}%")
        print("="*70)
    
    def generate_netlist(self, results, output_path=None):
        """Generate SPICE netlist with optimized parameters"""
        # Read template
        with open(self.netlist_template_path, 'r') as f:
            template = f.read()
        
        # Format optimized values for SPICE netlist
        # Use scientific notation for width values
        a_str = f"{results['a']:.6e}"
        b_str = f"{results['b']:.6e}"
        c_str = f"{results['c']:.6e}"
        d_str = f"{results['d']:.6e}"
        
        # Replace placeholders with optimized values
        # Replace W=a, W=b, W=c, W=d with actual values
        netlist = template.replace('W=a ', f'W={a_str} ')  # Space after to avoid partial matches
        netlist = netlist.replace('W=b ', f'W={b_str} ')
        netlist = netlist.replace('W=c ', f'W={c_str} ')
        netlist = netlist.replace('W=d ', f'W={d_str} ')
        
        # Add header comment with optimization info
        header = f"""* Generated by Inverse Design Optimizer
* Target: Gain={self.target_gain}, PM={self.target_pm}, UGF={self.target_ugf}
* Optimized Parameters: a={a_str}, b={b_str}, c={c_str}, d={d_str}
* Predicted: Gain={results['pred_gain']:.4f}, PM={results['pred_pm']:.4f}, UGF={results['pred_ugf']:.4e}
* Max Error: {results['max_error']:.4f}%
*
"""
        netlist = header + netlist
        
        # Display netlist
        print("\n📄 GENERATED NETLIST:")
        print("="*70)
        print(netlist)
        print("="*70)
        
        # Save to file if output path provided
        if output_path:
            output_file = Path(output_path)
            with open(output_file, 'w') as f:
                f.write(netlist)
            print(f"\n💾 Netlist saved to: {output_file.absolute()}")
        
        return netlist


def main():
    parser = argparse.ArgumentParser(
        description='Inverse Design Optimizer - Find optimal circuit parameters for target specs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inverse_design_optimizer.py --gain 45.5 --pm 65.0 --ugf 1e6
  python inverse_design_optimizer.py --gain 45.5 --pm 65.0 --ugf 1e6 --output optimized_netlist.cir
  python inverse_design_optimizer.py --gain 45.5 --pm 65.0 --ugf 1e6 --threshold 0.5
        """
    )
    
    parser.add_argument('--gain', type=float, required=True, help='Target gain value')
    parser.add_argument('--pm', type=float, required=True, help='Target phase margin value')
    parser.add_argument('--ugf', type=float, required=True, help='Target unity gain frequency')
    parser.add_argument('--threshold', type=float, default=1.0, 
                       help='Convergence threshold in percentage (default: 1.0%%)')
    parser.add_argument('--output', type=str, default='optimized_opamp180nm.cir',
                       help='Output netlist filename (default: optimized_opamp180nm.cir)')
    parser.add_argument('--models-path', type=str, 
                       default='notebooks/new_data_exp/trained_models',
                       help='Path to trained models directory')
    parser.add_argument('--param-bounds', type=str,
                       default='notebooks/new_data_exp/trained_models/param_bounds.pkl',
                       help='Path to parameter bounds file')
    parser.add_argument('--netlist-template', type=str,
                       default='notebooks/new_data_exp/opamp180nm',
                       help='Path to netlist template file')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    try:
        optimizer = InverseDesignOptimizer(
            models_path=args.models_path,
            param_bounds_path=args.param_bounds,
            netlist_template_path=args.netlist_template
        )
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files: {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  - {args.models_path}/model_gain_xgboost.pkl")
        print(f"  - {args.models_path}/model_pm_xgboost.pkl")
        print(f"  - {args.models_path}/model_ugf_xgboost.pkl")
        print(f"  - {args.param_bounds}")
        print(f"  - {args.netlist_template}")
        sys.exit(1)
    
    # Run optimization
    try:
        results = optimizer.optimize(
            target_gain=args.gain,
            target_pm=args.pm,
            target_ugf=args.ugf,
            convergence_threshold=args.threshold
        )
        
        # Display results
        optimizer.display_results(results)
        
        # Generate and save netlist
        optimizer.generate_netlist(results, output_path=args.output)
        
        print("\n✅ Optimization complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Optimization interrupted by user")
        if optimizer.best_params:
            print("Displaying best solution found so far...")
            optimizer.display_results(optimizer.best_params)
            optimizer.generate_netlist(optimizer.best_params, output_path=args.output)
        sys.exit(0)


if __name__ == "__main__":
    main()