"""
Data Generation Module for Resource Allocation Problem

This module generates synthetic datasets for the cloud computing resource allocation
optimization problem. It creates realistic scenarios with tasks, VM types, and system parameters.

Author: Research Engineer
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import random


@dataclass
class Task:
    """Represents a computational task with resource requirements."""
    task_id: int
    revenue: float
    deadline: int
    priority: float
    min_service_level: float
    cpu_req: int
    mem_req: int
    duration: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VMType:
    """Represents a virtual machine type with capacity and cost."""
    vm_id: int
    hourly_cost: float
    cpu_cap: int
    mem_cap: int
    max_instances: int
    vm_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SystemConfig:
    """System-wide configuration parameters."""
    budget: float
    time_horizon: int
    num_tasks: int
    num_vm_types: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DataGenerator:
    """
    Generates synthetic data for the resource allocation problem.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    config : Dict
        Configuration dictionary with generation parameters
    """
    
    def __init__(self, seed: int = 42, config: Dict = None):
        self.seed = seed
        self.config = config or self._default_config()
        self._set_seed()
        
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def _default_config(self) -> Dict:
        """Default configuration for data generation."""
        return {
            'num_tasks': 50,
            'num_vm_types': 5,
            'time_horizon': 24,  # 24 hours
            'budget': 5000.0,
            'task_revenue_range': (50, 500),
            'task_cpu_range': (1, 8),
            'task_mem_range': (2, 32),
            'task_duration_range': (1, 6),
            'vm_cost_range': (0.5, 4.0),
            'vm_cpu_range': [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
            'vm_mem_range': [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)],
            'vm_max_instances': 20,
        }
    
    def generate_tasks(self) -> List[Task]:
        """
        Generate synthetic tasks with realistic characteristics.
        
        Returns:
        --------
        List[Task] : List of task objects
        """
        tasks = []
        cfg = self.config
        
        for i in range(cfg['num_tasks']):
            # Revenue correlates with resource requirements
            cpu_req = np.random.randint(*cfg['task_cpu_range'])
            mem_req = np.random.randint(*cfg['task_mem_range'])
            duration = np.random.randint(*cfg['task_duration_range'])
            
            # Revenue scales with resource intensity
            base_revenue = np.random.uniform(*cfg['task_revenue_range'])
            resource_factor = (cpu_req / 8 + mem_req / 32) / 2
            revenue = base_revenue * (0.5 + 0.5 * resource_factor)
            
            # Deadline based on duration (must be >= duration)
            min_deadline = duration
            max_deadline = cfg['time_horizon']
            deadline = np.random.randint(min_deadline, max_deadline + 1)
            
            # Priority: some tasks are more important
            priority = np.random.choice([1.0, 1.5, 2.0], p=[0.6, 0.3, 0.1])
            
            # Minimum service level (fraction of task that must be completed)
            min_service = np.random.uniform(0.0, 0.5)
            
            task = Task(
                task_id=i,
                revenue=round(revenue, 2),
                deadline=deadline,
                priority=priority,
                min_service_level=round(min_service, 2),
                cpu_req=cpu_req,
                mem_req=mem_req,
                duration=duration
            )
            tasks.append(task)
            
        return tasks
    
    def generate_vm_types(self) -> List[VMType]:
        """
        Generate VM types with different capacity tiers.
        
        Returns:
        --------
        List[VMType] : List of VM type objects
        """
        vm_types = []
        cfg = self.config
        
        vm_names = ['Small', 'Medium', 'Large', 'XLarge', '2XLarge']
        
        for j in range(cfg['num_vm_types']):
            cpu_min, cpu_max = cfg['vm_cpu_range'][j]
            mem_min, mem_max = cfg['vm_mem_range'][j]
            
            cpu_cap = np.random.randint(cpu_min, cpu_max + 1)
            mem_cap = np.random.randint(mem_min, mem_max + 1)
            
            # Cost scales with capacity
            base_cost = cfg['vm_cost_range'][0]
            max_cost = cfg['vm_cost_range'][1]
            capacity_factor = j / (cfg['num_vm_types'] - 1)
            hourly_cost = base_cost + (max_cost - base_cost) * capacity_factor
            hourly_cost *= (0.9 + 0.2 * np.random.random())  # Add noise
            
            max_instances = cfg['vm_max_instances']
            
            vm = VMType(
                vm_id=j,
                hourly_cost=round(hourly_cost, 2),
                cpu_cap=cpu_cap,
                mem_cap=mem_cap,
                max_instances=max_instances,
                vm_name=vm_names[j]
            )
            vm_types.append(vm)
            
        return vm_types
    
    def generate_system_config(self) -> SystemConfig:
        """
        Generate system configuration.
        
        Returns:
        --------
        SystemConfig : System configuration object
        """
        cfg = self.config
        return SystemConfig(
            budget=cfg['budget'],
            time_horizon=cfg['time_horizon'],
            num_tasks=cfg['num_tasks'],
            num_vm_types=cfg['num_vm_types']
        )
    
    def generate_dataset(self, output_dir: str = None) -> Dict:
        """
        Generate complete dataset and optionally save to files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save generated files
            
        Returns:
        --------
        Dict : Dictionary containing all generated data
        """
        tasks = self.generate_tasks()
        vm_types = self.generate_vm_types()
        system_config = self.generate_system_config()
        
        dataset = {
            'tasks': [t.to_dict() for t in tasks],
            'vm_types': [v.to_dict() for v in vm_types],
            'system_config': system_config.to_dict(),
            'metadata': {
                'seed': self.seed,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSON
            with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
                json.dump(dataset, f, indent=2)
            
            # Save as CSV for easy inspection
            tasks_df = pd.DataFrame([t.to_dict() for t in tasks])
            vm_df = pd.DataFrame([v.to_dict() for v in vm_types])
            
            tasks_df.to_csv(os.path.join(output_dir, 'tasks.csv'), index=False)
            vm_df.to_csv(os.path.join(output_dir, 'vm_types.csv'), index=False)
            
            # Save config
            with open(os.path.join(output_dir, 'system_config.json'), 'w') as f:
                json.dump(system_config.to_dict(), f, indent=2)
                
        return dataset
    
    def generate_sensitivity_scenarios(self, base_budget: float, 
                                       budget_variations: List[float] = None,
                                       capacity_variations: List[float] = None) -> List[Dict]:
        """
        Generate multiple scenarios for sensitivity analysis.
        
        Parameters:
        -----------
        base_budget : float
            Base budget value
        budget_variations : List[float]
            List of budget multipliers (e.g., [0.5, 0.75, 1.0, 1.25, 1.5])
        capacity_variations : List[float]
            List of capacity multipliers
            
        Returns:
        --------
        List[Dict] : List of scenario configurations
        """
        if budget_variations is None:
            budget_variations = [0.5, 0.75, 1.0, 1.25, 1.5]
        if capacity_variations is None:
            capacity_variations = [0.6, 0.8, 1.0, 1.2, 1.4]
            
        scenarios = []
        
        # Budget sensitivity scenarios
        for mult in budget_variations:
            config = self.config.copy()
            config['budget'] = base_budget * mult
            config['scenario_type'] = 'budget'
            config['multiplier'] = mult
            scenarios.append(config)
            
        # Capacity sensitivity scenarios
        for mult in capacity_variations:
            config = self.config.copy()
            config['budget'] = base_budget
            config['vm_max_instances'] = int(self.config['vm_max_instances'] * mult)
            config['scenario_type'] = 'capacity'
            config['multiplier'] = mult
            scenarios.append(config)
            
        return scenarios


def generate_multiple_datasets(output_dir: str, num_datasets: int = 3, 
                                base_seed: int = 42):
    """
    Generate multiple datasets with different seeds for robustness testing.
    
    Parameters:
    -----------
    output_dir : str
        Base output directory
    num_datasets : int
        Number of datasets to generate
    base_seed : int
        Starting seed value
    """
    for i in range(num_datasets):
        seed = base_seed + i * 100
        dataset_dir = os.path.join(output_dir, f'dataset_{i+1}')
        
        generator = DataGenerator(seed=seed)
        generator.generate_dataset(output_dir=dataset_dir)
        
        print(f"Generated dataset {i+1} with seed {seed} in {dataset_dir}")


if __name__ == '__main__':
    # Example usage
    output_dir = '../data'
    
    # Generate main dataset
    generator = DataGenerator(seed=42)
    dataset = generator.generate_dataset(output_dir=output_dir)
    
    print("Dataset generated successfully!")
    print(f"Number of tasks: {len(dataset['tasks'])}")
    print(f"Number of VM types: {len(dataset['vm_types'])}")
    print(f"Budget: ${dataset['system_config']['budget']}")
    
    # Generate multiple datasets for robustness
    generate_multiple_datasets(output_dir, num_datasets=3)
