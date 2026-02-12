"""
Analysis Module for Resource Allocation Problem

This module provides sensitivity analysis, visualization, and comparison
of optimization results for the resource allocation problem.

Author: Research Engineer
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on optimization model parameters.
    
    Parameters:
    -----------
    data : Dict
        Base problem data
    solver_name : str
        Solver to use for optimization
    """
    
    def __init__(self, data: Dict, solver_name: str = 'glpk'):
        self.data = data
        self.solver_name = solver_name
        self.results = {}
        
    def analyze_budget_constraint(self, budget_values: List[float],
                                   model_type: str = 'LP') -> pd.DataFrame:
        """
        Analyze sensitivity to budget constraint.
        
        Parameters:
        -----------
        budget_values : List[float]
            List of budget values to test
        model_type : str
            'LP' or 'MILP'
            
        Returns:
        --------
        pd.DataFrame : Sensitivity results
        """
        from model import ResourceAllocationModel
        from solve import ResourceAllocationSolver
        
        results = []
        
        for budget in budget_values:
            # Create modified data
            data_mod = self._copy_data()
            data_mod['system_config']['budget'] = budget
            
            # Build and solve
            model = ResourceAllocationModel(data_mod, model_type)
            m = model.build_model()
            
            solver = ResourceAllocationSolver(m, model_type, self.solver_name)
            solution = solver.solve(tee=False)
            
            # Extract metrics
            metrics = solution['metrics']
            sol_data = solution['solution']
            
            results.append({
                'budget': budget,
                'objective': metrics['objective_value'],
                'solve_time': metrics['solve_time'],
                'status': metrics['termination_condition'],
                'total_revenue': sol_data['summary'].get('total_revenue', 0),
                'total_cost': sol_data['summary'].get('total_cost', 0),
                'profit': sol_data['summary'].get('profit', 0),
                'tasks_completed': sol_data['summary'].get('num_tasks_completed', 0)
            })
        
        df = pd.DataFrame(results)
        self.results['budget'] = df
        return df
    
    def analyze_capacity_constraint(self, capacity_multipliers: List[float],
                                     model_type: str = 'LP') -> pd.DataFrame:
        """
        Analyze sensitivity to VM capacity constraint.
        
        Parameters:
        -----------
        capacity_multipliers : List[float]
            Multipliers for max_instances (1.0 = base capacity)
        model_type : str
            'LP' or 'MILP'
            
        Returns:
        --------
        pd.DataFrame : Sensitivity results
        """
        from model import ResourceAllocationModel
        from solve import ResourceAllocationSolver
        
        results = []
        base_max = self.data['vm_types'][0]['max_instances']
        
        for mult in capacity_multipliers:
            # Create modified data
            data_mod = self._copy_data()
            for vm in data_mod['vm_types']:
                vm['max_instances'] = int(vm['max_instances'] * mult)
            
            # Build and solve
            model = ResourceAllocationModel(data_mod, model_type)
            m = model.build_model()
            
            solver = ResourceAllocationSolver(m, model_type, self.solver_name)
            solution = solver.solve(tee=False)
            
            metrics = solution['metrics']
            sol_data = solution['solution']
            
            results.append({
                'capacity_multiplier': mult,
                'effective_capacity': int(base_max * mult),
                'objective': metrics['objective_value'],
                'solve_time': metrics['solve_time'],
                'status': metrics['termination_condition'],
                'total_revenue': sol_data['summary'].get('total_revenue', 0),
                'total_cost': sol_data['summary'].get('total_cost', 0),
                'profit': sol_data['summary'].get('profit', 0),
                'tasks_completed': sol_data['summary'].get('num_tasks_completed', 0)
            })
        
        df = pd.DataFrame(results)
        self.results['capacity'] = df
        return df
    
    def analyze_priority_distribution(self, priority_scenarios: List[Dict],
                                       model_type: str = 'LP') -> pd.DataFrame:
        """
        Analyze sensitivity to task priority distribution.
        
        Parameters:
        -----------
        priority_scenarios : List[Dict]
            List of priority distribution configurations
        model_type : str
            'LP' or 'MILP'
            
        Returns:
        --------
        pd.DataFrame : Sensitivity results
        """
        from model import ResourceAllocationModel
        from solve import ResourceAllocationSolver
        
        results = []
        
        for scenario in priority_scenarios:
            data_mod = self._copy_data()
            
            # Modify priorities
            for i, task in enumerate(data_mod['tasks']):
                task['priority'] = scenario['distribution'][i % len(scenario['distribution'])]
            
            model = ResourceAllocationModel(data_mod, model_type)
            m = model.build_model()
            
            solver = ResourceAllocationSolver(m, model_type, self.solver_name)
            solution = solver.solve(tee=False)
            
            results.append({
                'scenario_name': scenario['name'],
                'objective': solution['metrics']['objective_value'],
                'solve_time': solution['metrics']['solve_time'],
                'avg_priority': np.mean([t['priority'] for t in data_mod['tasks']])
            })
        
        return pd.DataFrame(results)
    
    def _copy_data(self) -> Dict:
        """Create a deep copy of the data."""
        import copy
        return copy.deepcopy(self.data)
    
    def get_shadow_prices(self, model: 'pyo.ConcreteModel') -> Dict:
        """
        Extract shadow prices (dual values) from LP solution.
        
        Parameters:
        -----------
        model : pyo.ConcreteModel
            Solved Pyomo model
            
        Returns:
        --------
        Dict : Shadow prices for each constraint
        """
        shadow_prices = {}
        
        # Get dual values if available
        if hasattr(model, 'dual'):
            # Budget constraint shadow price
            if hasattr(model, 'budget_constr'):
                shadow_prices['budget'] = model.dual.get(model.budget_constr, 0)
            
            # Capacity constraints
            if hasattr(model, 'cpu_cap_constr'):
                cpu_prices = [model.dual.get(model.cpu_cap_constr[j,k], 0) 
                             for j in model.J for k in model.K]
                shadow_prices['cpu_capacity_avg'] = np.mean(cpu_prices)
            
            if hasattr(model, 'mem_cap_constr'):
                mem_prices = [model.dual.get(model.mem_cap_constr[j,k], 0) 
                             for j in model.J for k in model.K]
                shadow_prices['mem_capacity_avg'] = np.mean(mem_prices)
        
        return shadow_prices


class ResultVisualizer:
    """
    Visualization tools for optimization results.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save figures
    """
    
    def __init__(self, output_dir: str = '../figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_budget_sensitivity(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot budget sensitivity analysis results.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Results from budget sensitivity analysis
        save : bool
            Save figure to file
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Objective vs Budget
        ax = axes[0, 0]
        ax.plot(df['budget'], df['objective'], 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Budget (USD)')
        ax.set_ylabel('Objective Value (Profit)')
        ax.set_title('Profit vs Budget Constraint')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Revenue and Cost breakdown
        ax = axes[0, 1]
        width = (df['budget'].iloc[1] - df['budget'].iloc[0]) * 0.35 if len(df) > 1 else 100
        x = np.arange(len(df))
        ax.bar(x - 0.2, df['total_revenue'], width=0.4, label='Revenue', alpha=0.8)
        ax.bar(x + 0.2, df['total_cost'], width=0.4, label='Cost', alpha=0.8)
        ax.set_xlabel('Budget Scenario')
        ax.set_ylabel('Amount (USD)')
        ax.set_title('Revenue vs Cost by Budget Level')
        ax.set_xticks(x)
        ax.set_xticklabels([f'${b:,.0f}' for b in df['budget']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Tasks completed
        ax = axes[1, 0]
        ax.bar(x, df['tasks_completed'], color='green', alpha=0.7)
        ax.set_xlabel('Budget Scenario')
        ax.set_ylabel('Tasks Completed')
        ax.set_title('Number of Tasks Completed vs Budget')
        ax.set_xticks(x)
        ax.set_xticklabels([f'${b:,.0f}' for b in df['budget']], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Solve time
        ax = axes[1, 1]
        ax.plot(df['budget'], df['solve_time'], 'r-s', linewidth=2, markersize=8)
        ax.set_xlabel('Budget (USD)')
        ax.set_ylabel('Solve Time (seconds)')
        ax.set_title('Computational Time vs Budget')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'budget_sensitivity.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_capacity_sensitivity(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Plot capacity sensitivity analysis results.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Results from capacity sensitivity analysis
        save : bool
            Save figure to file
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Objective vs Capacity
        ax = axes[0, 0]
        ax.plot(df['capacity_multiplier'], df['objective'], 'g-o', linewidth=2, markersize=8)
        ax.set_xlabel('Capacity Multiplier')
        ax.set_ylabel('Objective Value (Profit)')
        ax.set_title('Profit vs VM Capacity')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Resource utilization efficiency
        ax = axes[0, 1]
        efficiency = df['profit'] / (df['effective_capacity'] + 1)
        ax.bar(range(len(df)), efficiency, color='purple', alpha=0.7)
        ax.set_xlabel('Capacity Scenario')
        ax.set_ylabel('Profit per Unit Capacity')
        ax.set_title('Resource Efficiency vs Capacity')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f'{m:.1f}x' for m in df['capacity_multiplier']], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cost breakdown
        ax = axes[1, 0]
        ax.fill_between(df['capacity_multiplier'], 0, df['total_cost'], 
                        alpha=0.5, label='Cost', color='red')
        ax.fill_between(df['capacity_multiplier'], df['total_cost'], 
                        df['total_revenue'], alpha=0.5, label='Profit', color='green')
        ax.set_xlabel('Capacity Multiplier')
        ax.set_ylabel('Amount (USD)')
        ax.set_title('Cost Structure vs Capacity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Tasks completed
        ax = axes[1, 1]
        ax.plot(df['capacity_multiplier'], df['tasks_completed'], 'm-^', 
                linewidth=2, markersize=10)
        ax.set_xlabel('Capacity Multiplier')
        ax.set_ylabel('Tasks Completed')
        ax.set_title('Task Completion vs Capacity')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'capacity_sensitivity.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_lp_vs_milp_comparison(self, lp_results: Dict, milp_results: Dict,
                                    save: bool = True) -> plt.Figure:
        """
        Visualize comparison between LP and MILP solutions.
        
        Parameters:
        -----------
        lp_results : Dict
            LP solution results
        milp_results : Dict
            MILP solution results
        save : bool
            Save figure to file
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract data
        lp_obj = lp_results['metrics']['objective_value']
        milp_obj = milp_results['metrics']['objective_value']
        lp_time = lp_results['metrics']['solve_time']
        milp_time = milp_results['metrics']['solve_time']
        
        lp_summary = lp_results['solution']['summary']
        milp_summary = milp_results['solution']['summary']
        
        # Plot 1: Objective comparison
        ax = axes[0]
        models = ['LP', 'MILP']
        objectives = [lp_obj, milp_obj]
        colors = ['skyblue', 'lightcoral']
        bars = ax.bar(models, objectives, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Objective Value (Profit)')
        ax.set_title('LP vs MILP: Objective Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, objectives):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Solve time comparison
        ax = axes[1]
        times = [lp_time, milp_time]
        bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Solve Time (seconds)')
        ax.set_title('LP vs MILP: Computational Time')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Revenue/Cost breakdown
        ax = axes[2]
        metrics = ['Revenue', 'Cost', 'Profit']
        lp_values = [lp_summary['total_revenue'], lp_summary['total_cost'], 
                     lp_summary['profit']]
        milp_values = [milp_summary['total_revenue'], milp_summary['total_cost'],
                       milp_summary['profit']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, lp_values, width, label='LP', color='skyblue', 
               alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, milp_values, width, label='MILP', color='lightcoral',
               alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Amount (USD)')
        ax.set_title('Financial Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'lp_vs_milp_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_allocation_heatmap(self, solution: Dict, vm_types: List[Dict],
                                 save: bool = True) -> plt.Figure:
        """
        Create heatmap of VM usage over time.
        
        Parameters:
        -----------
        solution : Dict
            Solution dictionary with vm_usage
        vm_types : List[Dict]
            VM type definitions
        save : bool
            Save figure to file
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        vm_usage = solution['vm_usage']
        
        # Determine dimensions
        max_time = max(u['time_period'] for u in vm_usage) + 1
        num_vm_types = len(vm_types)
        
        # Create usage matrix
        usage_matrix = np.zeros((num_vm_types, max_time))
        for u in vm_usage:
            usage_matrix[u['vm_type'], u['time_period']] = u['count']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        vm_names = [v['vm_name'] for v in vm_types]
        sns.heatmap(usage_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=range(max_time), yticklabels=vm_names,
                   cbar_kws={'label': 'Number of VMs'}, ax=ax)
        
        ax.set_xlabel('Time Period (hours)')
        ax.set_ylabel('VM Type')
        ax.set_title('VM Usage Over Time')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'allocation_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def create_summary_report(self, results: Dict, output_path: str):
        """
        Create a summary report of all analyses.
        
        Parameters:
        -----------
        results : Dict
            Dictionary containing all analysis results
        output_path : str
            Path to save report
        """
        with open(output_path, 'w') as f:
            f.write("# Resource Allocation Optimization: Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # LP vs MILP comparison
            if 'lp_vs_milp' in results:
                f.write("## LP vs MILP Comparison\n\n")
                comp = results['lp_vs_milp']
                f.write(f"- Integrality Gap: {comp.get('integrality_gap_percent', 'N/A')}%\n")
                f.write(f"- LP Objective: ${comp['LP']['metrics']['objective_value']:,.2f}\n")
                f.write(f"- MILP Objective: ${comp['MILP']['metrics']['objective_value']:,.2f}\n")
                f.write(f"- LP Solve Time: {comp['LP']['metrics']['solve_time']:.4f}s\n")
                f.write(f"- MILP Solve Time: {comp['MILP']['metrics']['solve_time']:.4f}s\n\n")
            
            # Budget sensitivity
            if 'budget_sensitivity' in results:
                f.write("## Budget Sensitivity Analysis\n\n")
                df = results['budget_sensitivity']
                f.write(f"Tested budget range: ${df['budget'].min():,.0f} - ${df['budget'].max():,.0f}\n")
                f.write(f"Objective range: ${df['objective'].min():,.2f} - ${df['objective'].max():,.2f}\n")
                f.write(f"Average solve time: {df['solve_time'].mean():.4f}s\n\n")
            
            # Capacity sensitivity
            if 'capacity_sensitivity' in results:
                f.write("## Capacity Sensitivity Analysis\n\n")
                df = results['capacity_sensitivity']
                f.write(f"Tested capacity multipliers: {df['capacity_multiplier'].tolist()}\n")
                f.write(f"Objective range: ${df['objective'].min():,.2f} - ${df['objective'].max():,.2f}\n\n")
        
        print(f"Report saved to {output_path}")


class StatisticalAnalyzer:
    """
    Statistical analysis of optimization results.
    """
    
    @staticmethod
    def calculate_integrality_gap(lp_obj: float, milp_obj: float) -> Dict:
        """
        Calculate integrality gap metrics.
        
        Parameters:
        -----------
        lp_obj : float
            LP objective value
        milp_obj : float
            MILP objective value
            
        Returns:
        --------
        Dict : Gap metrics
        """
        absolute_gap = abs(lp_obj - milp_obj)
        relative_gap = absolute_gap / abs(lp_obj) if abs(lp_obj) > 1e-10 else 0
        
        return {
            'absolute_gap': absolute_gap,
            'relative_gap': relative_gap,
            'relative_gap_percent': relative_gap * 100,
            'lp_objective': lp_obj,
            'milp_objective': milp_obj
        }
    
    @staticmethod
    def analyze_constraint_tightness(solution: Dict, data: Dict) -> pd.DataFrame:
        """
        Analyze which constraints are binding in the solution.
        
        Parameters:
        -----------
        solution : Dict
            Solution dictionary
        data : Dict
            Problem data
            
        Returns:
        --------
        pd.DataFrame : Constraint analysis
        """
        summary = solution['summary']
        config = data['system_config']
        
        analysis = []
        
        # Budget utilization
        budget_util = summary['total_cost'] / config['budget'] * 100
        analysis.append({
            'constraint': 'Budget',
            'limit': config['budget'],
            'used': summary['total_cost'],
            'utilization_percent': budget_util,
            'status': 'Binding' if budget_util > 95 else 'Non-binding'
        })
        
        # VM capacity utilization
        for vm in data['vm_types']:
            vm_usage = [u for u in solution['vm_usage'] if u['vm_type'] == vm['vm_id']]
            max_usage = max([u['count'] for u in vm_usage]) if vm_usage else 0
            util = max_usage / vm['max_instances'] * 100
            
            analysis.append({
                'constraint': f"VM {vm['vm_name']} Capacity",
                'limit': vm['max_instances'],
                'used': max_usage,
                'utilization_percent': util,
                'status': 'Binding' if util > 95 else 'Non-binding'
            })
        
        return pd.DataFrame(analysis)


if __name__ == '__main__':
    print("Analysis module loaded successfully.")
    print("Available classes:")
    print("  - SensitivityAnalyzer: Perform sensitivity analysis")
    print("  - ResultVisualizer: Create visualizations")
    print("  - StatisticalAnalyzer: Statistical analysis tools")
