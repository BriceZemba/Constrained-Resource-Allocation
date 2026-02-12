"""
Optimization Model Module for Resource Allocation Problem

This module implements both LP (Linear Programming) and MILP (Mixed-Integer Linear Programming)
formulations of the cloud computing resource allocation problem using Pyomo.

Author: Research Engineer
Date: 2026-02-12
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ResourceAllocationModel:
    """
    Base class for resource allocation optimization models.
    
    Parameters:
    -----------
    data : Dict
        Dictionary containing tasks, vm_types, and system_config
    model_type : str
        Either 'LP' for linear programming or 'MILP' for mixed-integer
    """
    
    def __init__(self, data: Dict, model_type: str = 'LP'):
        self.data = data
        self.model_type = model_type.upper()
        self.model = None
        self.results = None
        self.solution = None
        
        # Extract data
        self.tasks = data['tasks']
        self.vm_types = data['vm_types']
        self.config = data['system_config']
        
        # Create index sets
        self.I = range(len(self.tasks))  # Tasks
        self.J = range(len(self.vm_types))  # VM types
        self.K = range(self.config['time_horizon'])  # Time periods
        
        # Big-M constant
        self.M = len(self.tasks) * 2
        
    def build_model(self) -> pyo.ConcreteModel:
        """
        Build the optimization model based on model_type.
        
        Returns:
        --------
        pyo.ConcreteModel : Pyomo model instance
        """
        if self.model_type == 'LP':
            return self._build_lp_model()
        elif self.model_type == 'MILP':
            return self._build_milp_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_lp_model(self) -> pyo.ConcreteModel:
        """
        Build the LP relaxation model (continuous variables).
        
        Model Structure:
        - Variables: x[i,j,k] (task assignment fraction), u[j,k] (VM count)
        - Objective: Maximize profit (revenue - cost)
        - Constraints: Budget, capacity, deadlines, service levels
        """
        m = pyo.ConcreteModel()
        
        # Sets
        m.I = pyo.Set(initialize=self.I)
        m.J = pyo.Set(initialize=self.J)
        m.K = pyo.Set(initialize=self.K)
        
        # Parameters
        def revenue_init(m, i):
            return self.tasks[i]['revenue']
        m.revenue = pyo.Param(m.I, initialize=revenue_init)
        
        def priority_init(m, i):
            return self.tasks[i]['priority']
        m.priority = pyo.Param(m.I, initialize=priority_init)
        
        def deadline_init(m, i):
            return self.tasks[i]['deadline']
        m.deadline = pyo.Param(m.I, initialize=deadline_init)
        
        def min_service_init(m, i):
            return self.tasks[i]['min_service_level']
        m.min_service = pyo.Param(m.I, initialize=min_service_init)
        
        def cpu_req_init(m, i):
            return self.tasks[i]['cpu_req']
        m.cpu_req = pyo.Param(m.I, initialize=cpu_req_init)
        
        def mem_req_init(m, i):
            return self.tasks[i]['mem_req']
        m.mem_req = pyo.Param(m.I, initialize=mem_req_init)
        
        def duration_init(m, i):
            return self.tasks[i]['duration']
        m.duration = pyo.Param(m.I, initialize=duration_init)
        
        def vm_cost_init(m, j):
            return self.vm_types[j]['hourly_cost']
        m.vm_cost = pyo.Param(m.J, initialize=vm_cost_init)
        
        def cpu_cap_init(m, j):
            return self.vm_types[j]['cpu_cap']
        m.cpu_cap = pyo.Param(m.J, initialize=cpu_cap_init)
        
        def mem_cap_init(m, j):
            return self.vm_types[j]['mem_cap']
        m.mem_cap = pyo.Param(m.J, initialize=mem_cap_init)
        
        def max_vm_init(m, j):
            return self.vm_types[j]['max_instances']
        m.max_vm = pyo.Param(m.J, initialize=max_vm_init)
        
        m.budget = pyo.Param(initialize=self.config['budget'])
        
        # Decision Variables
        # x[i,j,k]: fraction of task i assigned to VM j at time k
        m.x = pyo.Var(m.I, m.J, m.K, domain=pyo.NonNegativeReals, bounds=(0, 1))
        
        # u[j,k]: number of VM type j active at time k
        m.u = pyo.Var(m.J, m.K, domain=pyo.NonNegativeReals)
        
        # Objective: Maximize profit
        def obj_rule(m):
            revenue = sum(m.priority[i] * m.revenue[i] * m.x[i,j,k] 
                         for i in m.I for j in m.J for k in m.K)
            cost = sum(m.vm_cost[j] * m.u[j,k] for j in m.J for k in m.K)
            return revenue - cost
        m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        # Constraints
        
        # Budget constraint
        def budget_rule(m):
            return sum(m.vm_cost[j] * m.u[j,k] for j in m.J for k in m.K) <= m.budget
        m.budget_constr = pyo.Constraint(rule=budget_rule)
        
        # CPU capacity constraint
        def cpu_cap_rule(m, j, k):
            return sum(m.cpu_req[i] * m.x[i,j,k] for i in m.I) <= m.cpu_cap[j] * m.u[j,k]
        m.cpu_cap_constr = pyo.Constraint(m.J, m.K, rule=cpu_cap_rule)
        
        # Memory capacity constraint
        def mem_cap_rule(m, j, k):
            return sum(m.mem_req[i] * m.x[i,j,k] for i in m.I) <= m.mem_cap[j] * m.u[j,k]
        m.mem_cap_constr = pyo.Constraint(m.J, m.K, rule=mem_cap_rule)
        
        # Assignment limit: each task at most once per period
        def assign_limit_rule(m, i, k):
            return sum(m.x[i,j,k] for j in m.J) <= 1
        m.assign_limit_constr = pyo.Constraint(m.I, m.K, rule=assign_limit_rule)
        
        # Deadline constraint: no assignment after deadline
        def deadline_rule(m, i, j, k):
            if k > self.tasks[i]['deadline'] - self.tasks[i]['duration']:
                return m.x[i,j,k] == 0
            return pyo.Constraint.Skip
        m.deadline_constr = pyo.Constraint(m.I, m.J, m.K, rule=deadline_rule)
        
        # Minimum service level constraint
        def min_service_rule(m, i):
            max_time = min(self.tasks[i]['deadline'], len(self.K))
            return sum(m.x[i,j,k] for j in m.J for k in range(max_time)) >= m.min_service[i]
        m.min_service_constr = pyo.Constraint(m.I, rule=min_service_rule)
        
        # VM availability constraint
        def vm_avail_rule(m, j, k):
            return m.u[j,k] <= m.max_vm[j]
        m.vm_avail_constr = pyo.Constraint(m.J, m.K, rule=vm_avail_rule)
        
        self.model = m
        return m
    
    def _build_milp_model(self) -> pyo.ConcreteModel:
        """
        Build the MILP model (binary and integer variables).
        
        Model Structure:
        - Variables: y[i,j,k] (binary assignment), n[j,k] (integer VM count), z[i] (completion)
        - Objective: Maximize profit with binary completion
        - Constraints: Budget, capacity, completion logic, deadlines
        """
        m = pyo.ConcreteModel()
        
        # Sets
        m.I = pyo.Set(initialize=self.I)
        m.J = pyo.Set(initialize=self.J)
        m.K = pyo.Set(initialize=self.K)
        
        # Parameters (same as LP)
        def revenue_init(m, i):
            return self.tasks[i]['revenue']
        m.revenue = pyo.Param(m.I, initialize=revenue_init)
        
        def priority_init(m, i):
            return self.tasks[i]['priority']
        m.priority = pyo.Param(m.I, initialize=priority_init)
        
        def deadline_init(m, i):
            return self.tasks[i]['deadline']
        m.deadline = pyo.Param(m.I, initialize=deadline_init)
        
        def min_service_init(m, i):
            return self.tasks[i]['min_service_level']
        m.min_service = pyo.Param(m.I, initialize=min_service_init)
        
        def cpu_req_init(m, i):
            return self.tasks[i]['cpu_req']
        m.cpu_req = pyo.Param(m.I, initialize=cpu_req_init)
        
        def mem_req_init(m, i):
            return self.tasks[i]['mem_req']
        m.mem_req = pyo.Param(m.I, initialize=mem_req_init)
        
        def duration_init(m, i):
            return self.tasks[i]['duration']
        m.duration = pyo.Param(m.I, initialize=duration_init)
        
        def vm_cost_init(m, j):
            return self.vm_types[j]['hourly_cost']
        m.vm_cost = pyo.Param(m.J, initialize=vm_cost_init)
        
        def cpu_cap_init(m, j):
            return self.vm_types[j]['cpu_cap']
        m.cpu_cap = pyo.Param(m.J, initialize=cpu_cap_init)
        
        def mem_cap_init(m, j):
            return self.vm_types[j]['mem_cap']
        m.mem_cap = pyo.Param(m.J, initialize=mem_cap_init)
        
        def max_vm_init(m, j):
            return self.vm_types[j]['max_instances']
        m.max_vm = pyo.Param(m.J, initialize=max_vm_init)
        
        m.budget = pyo.Param(initialize=self.config['budget'])
        
        # Decision Variables
        # y[i,j,k]: 1 if task i is assigned to VM j at time k
        m.y = pyo.Var(m.I, m.J, m.K, domain=pyo.Binary)
        
        # n[j,k]: integer count of VM type j at time k
        m.n = pyo.Var(m.J, m.K, domain=pyo.NonNegativeIntegers, bounds=(0, 100))
        
        # z[i]: 1 if task i is fully completed
        m.z = pyo.Var(m.I, domain=pyo.Binary)
        
        # Objective: Maximize profit
        def obj_rule(m):
            revenue = sum(m.priority[i] * m.revenue[i] * m.z[i] for i in m.I)
            cost = sum(m.vm_cost[j] * m.n[j,k] for j in m.J for k in m.K)
            return revenue - cost
        m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        # Constraints
        
        # Budget constraint
        def budget_rule(m):
            return sum(m.vm_cost[j] * m.n[j,k] for j in m.J for k in m.K) <= m.budget
        m.budget_constr = pyo.Constraint(rule=budget_rule)
        
        # CPU capacity constraint
        def cpu_cap_rule(m, j, k):
            return sum(m.cpu_req[i] * m.y[i,j,k] for i in m.I) <= m.cpu_cap[j] * m.n[j,k]
        m.cpu_cap_constr = pyo.Constraint(m.J, m.K, rule=cpu_cap_rule)
        
        # Memory capacity constraint
        def mem_cap_rule(m, j, k):
            return sum(m.mem_req[i] * m.y[i,j,k] for i in m.I) <= m.mem_cap[j] * m.n[j,k]
        m.mem_cap_constr = pyo.Constraint(m.J, m.K, rule=mem_cap_rule)
        
        # Single assignment: each task at most one VM per period
        def single_assign_rule(m, i, k):
            return sum(m.y[i,j,k] for j in m.J) <= 1
        m.single_assign_constr = pyo.Constraint(m.I, m.K, rule=single_assign_rule)
        
        # Deadline constraint
        def deadline_rule(m, i, j, k):
            if k > self.tasks[i]['deadline'] - self.tasks[i]['duration']:
                return m.y[i,j,k] == 0
            return pyo.Constraint.Skip
        m.deadline_constr = pyo.Constraint(m.I, m.J, m.K, rule=deadline_rule)
        
        # Completion definition: z[i] <= sum of assignments
        def completion_def_rule(m, i):
            return m.z[i] <= sum(m.y[i,j,k] for j in m.J for k in m.K)
        m.completion_def_constr = pyo.Constraint(m.I, rule=completion_def_rule)
        
        # Completion big-M: sum of assignments <= M * z[i]
        def completion_big_m_rule(m, i):
            return sum(m.y[i,j,k] for j in m.J for k in m.K) <= self.M * m.z[i]
        m.completion_big_m_constr = pyo.Constraint(m.I, rule=completion_big_m_rule)
        
        # VM availability
        def vm_avail_rule(m, j, k):
            return m.n[j,k] <= m.max_vm[j]
        m.vm_avail_constr = pyo.Constraint(m.J, m.K, rule=vm_avail_rule)
        
        self.model = m
        return m
    
    def get_model_summary(self) -> Dict:
        """
        Get summary statistics of the model.
        
        Returns:
        --------
        Dict : Model statistics
        """
        if self.model is None:
            self.build_model()
            
        m = self.model
        num_vars = sum(1 for _ in m.component_objects(pyo.Var, active=True))
        num_constrs = sum(1 for _ in m.component_objects(pyo.Constraint, active=True))
        
        # Count individual variables
        var_counts = {}
        for v in m.component_objects(pyo.Var, active=True):
            var_counts[v.name] = v.__len__()
        
        return {
            'model_type': self.model_type,
            'num_variables': num_vars,
            'num_constraints': num_constrs,
            'variable_counts': var_counts,
            'num_tasks': len(self.tasks),
            'num_vm_types': len(self.vm_types),
            'time_horizon': self.config['time_horizon']
        }


class SensitivityModel:
    """
    Model wrapper for sensitivity analysis on specific constraints.
    
    Parameters:
    -----------
    base_model : ResourceAllocationModel
        Base model instance
    """
    
    def __init__(self, base_model: ResourceAllocationModel):
        self.base_model = base_model
        self.results = []
    
    def analyze_budget_sensitivity(self, budget_range: List[float], 
                                   solver_name: str = 'glpk') -> pd.DataFrame:
        """
        Perform sensitivity analysis on budget constraint.
        
        Parameters:
        -----------
        budget_range : List[float]
            List of budget values to test
        solver_name : str
            Solver to use
            
        Returns:
        --------
        pd.DataFrame : Results for each budget value
        """
        results = []
        
        for budget in budget_range:
            # Create new model with modified budget
            data_copy = self.base_model.data.copy()
            data_copy['system_config'] = self.base_model.config.copy()
            data_copy['system_config']['budget'] = budget
            
            model = ResourceAllocationModel(data_copy, self.base_model.model_type)
            m = model.build_model()
            
            # Solve
            solver = SolverFactory(solver_name)
            result = solver.solve(m, tee=False)
            
            # Extract results
            if result.solver.status == SolverStatus.ok:
                obj_value = pyo.value(m.OBJ)
                total_cost = sum(pyo.value(m.vm_cost[j] * 
                               (m.u[j,k] if hasattr(m, 'u') else m.n[j,k])) 
                               for j in m.J for k in m.K)
                
                results.append({
                    'budget': budget,
                    'objective': obj_value,
                    'total_cost': total_cost,
                    'profit': obj_value,
                    'status': str(result.solver.termination_condition)
                })
        
        return pd.DataFrame(results)
    
    def analyze_capacity_sensitivity(self, capacity_multipliers: List[float],
                                     solver_name: str = 'glpk') -> pd.DataFrame:
        """
        Perform sensitivity analysis on VM capacity.
        
        Parameters:
        -----------
        capacity_multipliers : List[float]
            List of multipliers for max_instances
        solver_name : str
            Solver to use
            
        Returns:
        --------
        pd.DataFrame : Results for each capacity level
        """
        results = []
        
        for mult in capacity_multipliers:
            # Modify VM capacities
            data_copy = self.base_model.data.copy()
            data_copy['vm_types'] = [vm.copy() for vm in self.base_model.vm_types]
            
            for vm in data_copy['vm_types']:
                vm['max_instances'] = int(vm['max_instances'] * mult)
            
            model = ResourceAllocationModel(data_copy, self.base_model.model_type)
            m = model.build_model()
            
            # Solve
            solver = SolverFactory(solver_name)
            result = solver.solve(m, tee=False)
            
            if result.solver.status == SolverStatus.ok:
                obj_value = pyo.value(m.OBJ)
                results.append({
                    'capacity_multiplier': mult,
                    'objective': obj_value,
                    'status': str(result.solver.termination_condition)
                })
        
        return pd.DataFrame(results)


if __name__ == '__main__':
    # Example usage
    import json
    
    # Load sample data
    with open('../data/dataset.json', 'r') as f:
        data = json.load(f)
    
    # Build LP model
    print("Building LP model...")
    lp_model = ResourceAllocationModel(data, 'LP')
    m_lp = lp_model.build_model()
    summary = lp_model.get_model_summary()
    print(f"LP Model: {summary}")
    
    # Build MILP model
    print("\nBuilding MILP model...")
    milp_model = ResourceAllocationModel(data, 'MILP')
    m_milp = milp_model.build_model()
    summary = milp_model.get_model_summary()
    print(f"MILP Model: {summary}")
