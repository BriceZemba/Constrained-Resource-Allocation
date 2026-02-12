"""
Solver Module for Resource Allocation Problem

This module handles the solving of LP and MILP optimization models,
including result extraction and performance metrics.

Author: Research Engineer
Date: 2026-02-12
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import pandas as pd
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SolutionMetrics:
    """Metrics for a solved optimization problem."""
    model_type: str
    objective_value: float
    solve_time: float
    termination_condition: str
    solver_status: str
    num_iterations: Optional[int] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AllocationResult:
    """Detailed allocation result for a task-VM-time assignment."""
    task_id: int
    vm_type: int
    time_period: int
    allocation_value: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ResourceAllocationSolver:
    """
    Solver for resource allocation optimization models.
    
    Parameters:
    -----------
    model : pyo.ConcreteModel
        Pyomo model instance
    model_type : str
        'LP' or 'MILP'
    solver_name : str
        Name of solver to use ('glpk', 'cbc', 'gurobi', 'cplex')
    """
    
    def __init__(self, model: pyo.ConcreteModel, model_type: str, 
                 solver_name: str = 'glpk'):
        self.model = model
        self.model_type = model_type.upper()
        self.solver_name = solver_name
        self.results = None
        self.metrics = None
        self.solution_data = None
        
    def solve(self, tee: bool = False, time_limit: int = None,
              mip_gap: float = None) -> Dict:
        """
        Solve the optimization model.
        
        Parameters:
        -----------
        tee : bool
            Print solver output
        time_limit : int
            Time limit in seconds
        mip_gap : float
            Relative MIP gap tolerance (for MILP)
            
        Returns:
        --------
        Dict : Solution results and metrics
        """
        # Configure solver
        solver = SolverFactory(self.solver_name)
        
        if solver is None:
            raise ValueError(f"Solver '{self.solver_name}' not available. "
                           f"Please install it or use a different solver.")
        
        # Set solver options
        options = {}
        if time_limit:
            options['timelimit'] = time_limit
        if mip_gap and self.model_type == 'MILP':
            if self.solver_name in ['glpk']:
                options['mipgap'] = mip_gap
            elif self.solver_name in ['cbc', 'gurobi', 'cplex']:
                options['mipgap'] = mip_gap
        
        # Solve
        start_time = time.time()
        self.results = solver.solve(self.model, tee=tee, options=options)
        solve_time = time.time() - start_time
        
        # Check solver status
        status = self.results.solver.status
        termination = self.results.solver.termination_condition
        
        if status != SolverStatus.ok:
            print(f"Warning: Solver status is {status}")
        
        if termination not in [TerminationCondition.optimal, 
                               TerminationCondition.feasible]:
            print(f"Warning: Termination condition is {termination}")
        
        # Extract metrics
        self.metrics = self._extract_metrics(solve_time)
        
        # Extract solution
        if status == SolverStatus.ok and termination in [
            TerminationCondition.optimal, 
            TerminationCondition.feasible
        ]:
            self.solution_data = self._extract_solution()
        
        return {
            'metrics': self.metrics.to_dict(),
            'solution': self.solution_data,
            'status': str(termination)
        }
    
    def _extract_metrics(self, solve_time: float) -> SolutionMetrics:
        """Extract solver performance metrics."""
        results = self.results
        
        # Get iteration count if available
        num_iter = None
        if hasattr(results.solver, 'iterations'):
            num_iter = results.solver.iterations
        
        # Get bounds for MILP
        lower_bound = None
        upper_bound = None
        gap = None
        
        if self.model_type == 'MILP':
            if hasattr(results.problem, 'lower_bound'):
                lower_bound = results.problem.lower_bound
            if hasattr(results.problem, 'upper_bound'):
                upper_bound = results.problem.upper_bound
            
            # Calculate MIP gap
            obj_value = pyo.value(self.model.OBJ)
            if lower_bound is not None and upper_bound is not None:
                if abs(upper_bound) > 1e-10:
                    gap = abs(upper_bound - lower_bound) / abs(upper_bound)
        
        return SolutionMetrics(
            model_type=self.model_type,
            objective_value=pyo.value(self.model.OBJ),
            solve_time=solve_time,
            termination_condition=str(results.solver.termination_condition),
            solver_status=str(results.solver.status),
            num_iterations=num_iter,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mip_gap=gap
        )
    
    def _extract_solution(self) -> Dict:
        """Extract detailed solution data."""
        m = self.model
        solution = {
            'allocations': [],
            'vm_usage': [],
            'task_completion': [],
            'summary': {}
        }
        
        # Extract allocations
        if self.model_type == 'LP':
            # Continuous variables
            for i in m.I:
                for j in m.J:
                    for k in m.K:
                        val = pyo.value(m.x[i,j,k])
                        if val > 1e-6:  # Only non-zero
                            solution['allocations'].append({
                                'task_id': i,
                                'vm_type': j,
                                'time_period': k,
                                'allocation_value': round(val, 4)
                            })
            
            # VM usage
            for j in m.J:
                for k in m.K:
                    val = pyo.value(m.u[j,k])
                    if val > 1e-6:
                        solution['vm_usage'].append({
                            'vm_type': j,
                            'time_period': k,
                            'count': round(val, 2)
                        })
        
        else:  # MILP
            # Binary variables
            for i in m.I:
                for j in m.J:
                    for k in m.K:
                        val = pyo.value(m.y[i,j,k])
                        if val > 0.5:
                            solution['allocations'].append({
                                'task_id': i,
                                'vm_type': j,
                                'time_period': k,
                                'allocation_value': 1
                            })
            
            # VM usage (integer)
            for j in m.J:
                for k in m.K:
                    val = pyo.value(m.n[j,k])
                    if val > 0:
                        solution['vm_usage'].append({
                            'vm_type': j,
                            'time_period': k,
                            'count': int(round(val))
                        })
            
            # Task completion
            for i in m.I:
                completed = pyo.value(m.z[i])
                solution['task_completion'].append({
                    'task_id': i,
                    'completed': bool(round(completed))
                })
        
        # Calculate summary statistics
        total_revenue = 0
        total_cost = 0
        
        if self.model_type == 'LP':
            for alloc in solution['allocations']:
                i = alloc['task_id']
                total_revenue += (pyo.value(m.priority[i]) * 
                                 pyo.value(m.revenue[i]) * 
                                 alloc['allocation_value'])
            for vm in solution['vm_usage']:
                j = vm['vm_type']
                total_cost += (pyo.value(m.vm_cost[j]) * vm['count'])
        else:
            for tc in solution['task_completion']:
                if tc['completed']:
                    i = tc['task_id']
                    total_revenue += pyo.value(m.priority[i]) * pyo.value(m.revenue[i])
            for vm in solution['vm_usage']:
                j = vm['vm_type']
                total_cost += pyo.value(m.vm_cost[j]) * vm['count']
        
        solution['summary'] = {
            'total_revenue': round(total_revenue, 2),
            'total_cost': round(total_cost, 2),
            'profit': round(total_revenue - total_cost, 2),
            'num_allocations': len(solution['allocations']),
            'num_tasks_completed': sum(1 for tc in solution['task_completion'] 
                                      if tc.get('completed', False))
        }
        
        return solution
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        results = {
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'solution': self.solution_data,
            'solver_name': self.solver_name,
            'model_type': self.model_type
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


class BatchSolver:
    """
    Batch solver for running multiple experiments.
    
    Parameters:
    -----------
    solver_name : str
        Default solver to use
    """
    
    def __init__(self, solver_name: str = 'glpk'):
        self.solver_name = solver_name
        self.results = []
    
    def solve_comparison(self, data: Dict, solvers: List[str] = None) -> pd.DataFrame:
        """
        Solve same problem with different solvers and compare.
        
        Parameters:
        -----------
        data : Dict
            Problem data
        solvers : List[str]
            List of solver names to compare
            
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        if solvers is None:
            solvers = ['glpk', 'cbc'] if self.solver_name == 'glpk' else [self.solver_name]
        
        from model import ResourceAllocationModel
        
        results = []
        for solver in solvers:
            for model_type in ['LP', 'MILP']:
                try:
                    model = ResourceAllocationModel(data, model_type)
                    m = model.build_model()
                    
                    solver_obj = ResourceAllocationSolver(m, model_type, solver)
                    result = solver_obj.solve(tee=False)
                    
                    results.append({
                        'solver': solver,
                        'model_type': model_type,
                        **result['metrics']
                    })
                except Exception as e:
                    print(f"Error with {solver}/{model_type}: {e}")
                    results.append({
                        'solver': solver,
                        'model_type': model_type,
                        'error': str(e)
                    })
        
        return pd.DataFrame(results)
    
    def solve_lp_vs_milp(self, data: Dict) -> Dict:
        """
        Solve both LP and MILP versions and compare results.
        
        Parameters:
        -----------
        data : Dict
            Problem data
            
        Returns:
        --------
        Dict : Comparison of LP and MILP solutions
        """
        from model import ResourceAllocationModel
        
        comparison = {}
        
        # Solve LP
        print("Solving LP relaxation...")
        lp_model = ResourceAllocationModel(data, 'LP')
        m_lp = lp_model.build_model()
        lp_solver = ResourceAllocationSolver(m_lp, 'LP', self.solver_name)
        lp_result = lp_solver.solve(tee=False)
        comparison['LP'] = lp_result
        
        # Solve MILP
        print("Solving MILP...")
        milp_model = ResourceAllocationModel(data, 'MILP')
        m_milp = milp_model.build_model()
        milp_solver = ResourceAllocationSolver(m_milp, 'MILP', self.solver_name)
        milp_result = milp_solver.solve(tee=False)
        comparison['MILP'] = milp_result
        
        # Calculate integrality gap
        lp_obj = lp_result['metrics']['objective_value']
        milp_obj = milp_result['metrics']['objective_value']
        gap = abs(lp_obj - milp_obj) / abs(lp_obj) * 100 if abs(lp_obj) > 1e-10 else 0
        
        comparison['integrality_gap'] = gap
        comparison['integrality_gap_percent'] = round(gap, 2)
        
        return comparison


def check_solver_availability() -> Dict[str, bool]:
    """
    Check which solvers are available on the system.
    
    Returns:
    --------
    Dict[str, bool] : Solver availability
    """
    solvers = ['glpk', 'cbc', 'ipopt', 'gurobi', 'cplex']
    availability = {}
    
    for solver in solvers:
        opt = SolverFactory(solver)
        availability[solver] = opt.available()
    
    return availability


if __name__ == '__main__':
    # Check available solvers
    print("Checking solver availability...")
    avail = check_solver_availability()
    for solver, available in avail.items():
        print(f"  {solver}: {'Available' if available else 'Not Available'}")
    
    # Example usage
    import json
    from model import ResourceAllocationModel
    
    # Load data
    try:
        with open('../data/dataset.json', 'r') as f:
            data = json.load(f)
        
        # Solve LP
        print("\nSolving LP model...")
        lp_model = ResourceAllocationModel(data, 'LP')
        m = lp_model.build_model()
        solver = ResourceAllocationSolver(m, 'LP', 'glpk')
        result = solver.solve(tee=False)
        
        print(f"Objective: {result['metrics']['objective_value']:.2f}")
        print(f"Solve time: {result['metrics']['solve_time']:.3f}s")
        
    except FileNotFoundError:
        print("Dataset not found. Run data_generation.py first.")
