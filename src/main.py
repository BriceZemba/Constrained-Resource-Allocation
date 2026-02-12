"""
Main Execution Module for Resource Allocation Research Project

This module orchestrates the complete experimental workflow:
1. Data generation
2. Model building and solving
3. Sensitivity analysis
4. Visualization and reporting

Usage:
    python main.py --config config.json --output-dir ../results

Author: Research Engineer
Date: 2026-02-12
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Import project modules
from data_generation import DataGenerator, generate_multiple_datasets
from model import ResourceAllocationModel
from solve import ResourceAllocationSolver, BatchSolver, check_solver_availability
from analysis import SensitivityAnalyzer, ResultVisualizer, StatisticalAnalyzer


def setup_directories(base_dir: str = '..') -> Dict[str, str]:
    """
    Create necessary directory structure.
    
    Parameters:
    -----------
    base_dir : str
        Base project directory
        
    Returns:
    --------
    Dict[str, str] : Directory paths
    """
    dirs = {
        'data': os.path.join(base_dir, 'data'),
        'results': os.path.join(base_dir, 'results'),
        'figures': os.path.join(base_dir, 'figures')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def generate_data(config: Dict, dirs: Dict[str, str], seed: int = 42) -> Dict:
    """
    Generate synthetic dataset.
    
    Parameters:
    -----------
    config : Dict
        Generation configuration
    dirs : Dict[str, str]
        Directory paths
    seed : int
        Random seed
        
    Returns:
    --------
    Dict : Generated dataset
    """
    print("=" * 60)
    print("STEP 1: DATA GENERATION")
    print("=" * 60)
    
    generator = DataGenerator(seed=seed, config=config)
    dataset = generator.generate_dataset(output_dir=dirs['data'])
    
    print(f"\nGenerated dataset:")
    print(f"  - Tasks: {len(dataset['tasks'])}")
    print(f"  - VM Types: {len(dataset['vm_types'])}")
    print(f"  - Time Horizon: {dataset['system_config']['time_horizon']} hours")
    print(f"  - Budget: ${dataset['system_config']['budget']:,.2f}")
    print(f"  - Data saved to: {dirs['data']}")
    
    return dataset


def run_base_optimization(data: Dict, solver_name: str) -> Dict:
    """
    Run base LP and MILP optimizations.
    
    Parameters:
    -----------
    data : Dict
        Problem data
    solver_name : str
        Solver to use
        
    Returns:
    --------
    Dict : Results for both LP and MILP
    """
    print("\n" + "=" * 60)
    print("STEP 2: BASE OPTIMIZATION")
    print("=" * 60)
    
    results = {}
    
    # LP Model
    print("\n--- LP Relaxation ---")
    lp_model = ResourceAllocationModel(data, 'LP')
    m_lp = lp_model.build_model()
    lp_summary = lp_model.get_model_summary()
    
    print(f"Model statistics:")
    print(f"  - Variables: {lp_summary['num_variables']}")
    print(f"  - Constraints: {lp_summary['num_constraints']}")
    
    lp_solver = ResourceAllocationSolver(m_lp, 'LP', solver_name)
    lp_result = lp_solver.solve(tee=False)
    
    print(f"Solution:")
    print(f"  - Objective: ${lp_result['metrics']['objective_value']:,.2f}")
    print(f"  - Solve Time: {lp_result['metrics']['solve_time']:.4f}s")
    print(f"  - Status: {lp_result['metrics']['termination_condition']}")
    
    results['LP'] = lp_result
    
    # MILP Model
    print("\n--- MILP Model ---")
    milp_model = ResourceAllocationModel(data, 'MILP')
    m_milp = milp_model.build_model()
    milp_summary = milp_model.get_model_summary()
    
    print(f"Model statistics:")
    print(f"  - Variables: {milp_summary['num_variables']}")
    print(f"  - Constraints: {milp_summary['num_constraints']}")
    
    milp_solver = ResourceAllocationSolver(m_milp, 'MILP', solver_name)
    milp_result = milp_solver.solve(tee=False)
    
    print(f"Solution:")
    print(f"  - Objective: ${milp_result['metrics']['objective_value']:,.2f}")
    print(f"  - Solve Time: {milp_result['metrics']['solve_time']:.4f}s")
    print(f"  - Status: {milp_result['metrics']['termination_condition']}")
    
    results['MILP'] = milp_result
    
    # Calculate integrality gap
    gap = StatisticalAnalyzer.calculate_integrality_gap(
        lp_result['metrics']['objective_value'],
        milp_result['metrics']['objective_value']
    )
    
    print(f"\n--- Integrality Analysis ---")
    print(f"  - Absolute Gap: ${gap['absolute_gap']:,.2f}")
    print(f"  - Relative Gap: {gap['relative_gap_percent']:.2f}%")
    
    results['integrality_gap'] = gap
    
    return results


def run_sensitivity_analysis(data: Dict, solver_name: str, dirs: Dict[str, str]) -> Dict:
    """
    Run sensitivity analysis on budget and capacity constraints.
    
    Parameters:
    -----------
    data : Dict
        Problem data
    solver_name : str
        Solver to use
    dirs : Dict[str, str]
        Directory paths
        
    Returns:
    --------
    Dict : Sensitivity analysis results
    """
    print("\n" + "=" * 60)
    print("STEP 3: SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    analyzer = SensitivityAnalyzer(data, solver_name)
    visualizer = ResultVisualizer(dirs['figures'])
    results = {}
    
    # Budget sensitivity
    print("\n--- Budget Sensitivity ---")
    base_budget = data['system_config']['budget']
    budget_values = [base_budget * m for m in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
    
    budget_df = analyzer.analyze_budget_constraint(budget_values, 'LP')
    results['budget_sensitivity'] = budget_df
    
    print(f"Tested {len(budget_values)} budget levels:")
    for _, row in budget_df.iterrows():
        print(f"  Budget ${row['budget']:,.0f}: "
              f"Profit=${row['objective']:,.2f}, "
              f"Tasks={row['tasks_completed']}")
    
    # Plot budget sensitivity
    fig = visualizer.plot_budget_sensitivity(budget_df, save=True)
    plt.close(fig)
    
    # Capacity sensitivity
    print("\n--- Capacity Sensitivity ---")
    capacity_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    capacity_df = analyzer.analyze_capacity_constraint(capacity_multipliers, 'LP')
    results['capacity_sensitivity'] = capacity_df
    
    print(f"Tested {len(capacity_multipliers)} capacity levels:")
    for _, row in capacity_df.iterrows():
        print(f"  Capacity {row['capacity_multiplier']:.1f}x: "
              f"Profit=${row['objective']:,.2f}, "
              f"Tasks={row['tasks_completed']}")
    
    # Plot capacity sensitivity
    fig = visualizer.plot_capacity_sensitivity(capacity_df, save=True)
    plt.close(fig)
    
    return results


def create_visualizations(base_results: Dict, data: Dict, dirs: Dict[str, str]):
    """
    Create additional visualizations.
    
    Parameters:
    -----------
    base_results : Dict
        Base optimization results
    data : Dict
        Problem data
    dirs : Dict[str, str]
        Directory paths
    """
    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)
    
    visualizer = ResultVisualizer(dirs['figures'])
    
    # LP vs MILP comparison
    print("\nCreating LP vs MILP comparison plot...")
    fig = visualizer.plot_lp_vs_milp_comparison(
        base_results['LP'], 
        base_results['MILP'],
        save=True
    )
    plt.close(fig)
    
    # Allocation heatmap for MILP
    print("Creating allocation heatmap...")
    fig = visualizer.plot_allocation_heatmap(
        base_results['MILP']['solution'],
        data['vm_types'],
        save=True
    )
    plt.close(fig)
    
    print(f"All figures saved to: {dirs['figures']}")


def save_results(all_results: Dict, dirs: Dict[str, str]):
    """
    Save all results to files.
    
    Parameters:
    -----------
    all_results : Dict
        All experimental results
    dirs : Dict[str, str]
        Directory paths
    """
    print("\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS")
    print("=" * 60)
    
    results_dir = dirs['results']
    
    # Save base results
    with open(os.path.join(results_dir, 'base_results.json'), 'w') as f:
        json.dump({
            'LP': {
                'metrics': all_results['base']['LP']['metrics'],
                'summary': all_results['base']['LP']['solution']['summary']
            },
            'MILP': {
                'metrics': all_results['base']['MILP']['metrics'],
                'summary': all_results['base']['MILP']['solution']['summary']
            },
            'integrality_gap': all_results['base']['integrality_gap']
        }, f, indent=2)
    
    # Save sensitivity results
    if 'sensitivity' in all_results:
        all_results['sensitivity']['budget_sensitivity'].to_csv(
            os.path.join(results_dir, 'budget_sensitivity.csv'), index=False
        )
        all_results['sensitivity']['capacity_sensitivity'].to_csv(
            os.path.join(results_dir, 'capacity_sensitivity.csv'), index=False
        )
    
    print(f"Results saved to: {results_dir}")


def create_summary_report(all_results: Dict, data: Dict, dirs: Dict[str, str]):
    """
    Create comprehensive summary report.
    
    Parameters:
    -----------
    all_results : Dict
        All experimental results
    data : Dict
        Problem data
    dirs : Dict[str, str]
        Directory paths
    """
    print("\n" + "=" * 60)
    print("STEP 6: GENERATING REPORT")
    print("=" * 60)
    
    visualizer = ResultVisualizer(dirs['figures'])
    
    report_data = {
        'lp_vs_milp': all_results['base'],
        'budget_sensitivity': all_results.get('sensitivity', {}).get('budget_sensitivity'),
        'capacity_sensitivity': all_results.get('sensitivity', {}).get('capacity_sensitivity')
    }
    
    report_path = os.path.join(dirs['results'], 'analysis_report.md')
    visualizer.create_summary_report(report_data, report_path)
    
    # Also create a JSON summary
    summary = {
        'experiment_date': pd.Timestamp.now().isoformat(),
        'problem_size': {
            'num_tasks': len(data['tasks']),
            'num_vm_types': len(data['vm_types']),
            'time_horizon': data['system_config']['time_horizon']
        },
        'base_results': {
            'lp_objective': all_results['base']['LP']['metrics']['objective_value'],
            'milp_objective': all_results['base']['MILP']['metrics']['objective_value'],
            'integrality_gap_percent': all_results['base']['integrality_gap']['relative_gap_percent'],
            'lp_solve_time': all_results['base']['LP']['metrics']['solve_time'],
            'milp_solve_time': all_results['base']['MILP']['metrics']['solve_time']
        }
    }
    
    with open(os.path.join(dirs['results'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Report saved to: {report_path}")


def run_full_experiment(config: Dict = None, solver: str = 'glpk', 
                        seed: int = 42, base_dir: str = '..'):
    """
    Run the complete experimental workflow.
    
    Parameters:
    -----------
    config : Dict
        Data generation configuration
    solver : str
        Solver name
    seed : int
        Random seed
    base_dir : str
        Base project directory
    """
    start_time = time.time()
    
    # Setup
    dirs = setup_directories(base_dir)
    
    # Check solver availability
    print("Checking solver availability...")
    avail = check_solver_availability()
    for s, a in avail.items():
        status = "✓" if a else "✗"
        print(f"  {status} {s}")
    
    if not avail.get(solver, False):
        print(f"Warning: Solver '{solver}' not available. Trying alternatives...")
        for alt in ['glpk', 'cbc', 'ipopt']:
            if avail.get(alt, False):
                solver = alt
                print(f"Using alternative solver: {solver}")
                break
    
    # Default config if not provided
    if config is None:
        config = {
            'num_tasks': 50,
            'num_vm_types': 5,
            'time_horizon': 24,
            'budget': 5000.0
        }
    
    all_results = {}
    
    try:
        # Step 1: Generate data
        data = generate_data(config, dirs, seed)
        
        # Step 2: Base optimization
        base_results = run_base_optimization(data, solver)
        all_results['base'] = base_results
        
        # Step 3: Sensitivity analysis
        sens_results = run_sensitivity_analysis(data, solver, dirs)
        all_results['sensitivity'] = sens_results
        
        # Step 4: Visualizations
        create_visualizations(base_results, data, dirs)
        
        # Step 5: Save results
        save_results(all_results, dirs)
        
        # Step 6: Generate report
        create_summary_report(all_results, data, dirs)
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"\nKey Findings:")
        print(f"  - LP Objective: ${base_results['LP']['metrics']['objective_value']:,.2f}")
        print(f"  - MILP Objective: ${base_results['MILP']['metrics']['objective_value']:,.2f}")
        print(f"  - Integrality Gap: {base_results['integrality_gap']['relative_gap_percent']:.2f}%")
        print(f"\nOutput directories:")
        print(f"  - Data: {dirs['data']}")
        print(f"  - Results: {dirs['results']}")
        print(f"  - Figures: {dirs['figures']}")
        
        return all_results
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Resource Allocation Optimization Research Project'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--solver', 
        type=str, 
        default='glpk',
        choices=['glpk', 'cbc', 'gurobi', 'cplex', 'ipopt'],
        help='Optimization solver to use'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='..',
        help='Base output directory'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run experiment
    run_full_experiment(
        config=config,
        solver=args.solver,
        seed=args.seed,
        base_dir=args.output_dir
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
