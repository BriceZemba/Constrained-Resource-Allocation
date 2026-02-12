# Project Summary: Constrained Resource Allocation using LP and MILP

## Project Overview

This research project provides a complete, reproducible implementation of constrained resource allocation optimization using Linear Programming (LP) and Mixed-Integer Linear Programming (MILP). The problem is motivated by cloud computing resource allocation where virtual machines must be assigned to computational tasks under budget, capacity, and deadline constraints.

## Deliverables

### 1. Mathematical Formulation (`formulation.tex`)
Complete LaTeX document containing:
- Problem description and motivation
- Sets, indices, and parameter definitions
- LP formulation with continuous variables
- MILP formulation with binary and integer variables
- Constraint descriptions (budget, capacity, deadlines, service levels)
- Problem complexity analysis

### 2. Python Implementation

#### `src/data_generation.py`
Synthetic dataset generation with:
- `Task` dataclass for task specifications (revenue, deadline, priority, resource requirements)
- `VMType` dataclass for VM specifications (cost, capacity)
- `DataGenerator` class with configurable parameters
- Support for multiple datasets with different seeds
- Sensitivity scenario generation

#### `src/model.py`
Optimization model implementations:
- `ResourceAllocationModel` class supporting both LP and MILP
- Pyomo-based model construction
- Proper constraint formulations
- Model statistics and summary methods
- `SensitivityModel` for constraint sensitivity analysis

#### `src/solve.py`
Solver interface and execution:
- `ResourceAllocationSolver` class
- Support for multiple solvers (GLPK, CBC, Gurobi, CPLEX)
- Solution extraction and metrics calculation
- `BatchSolver` for comparative experiments
- Solver availability checking

#### `src/analysis.py`
Analysis and visualization:
- `SensitivityAnalyzer` for budget and capacity sensitivity
- `ResultVisualizer` for publication-quality plots
- `StatisticalAnalyzer` for gap analysis and constraint tightness
- Support for heatmaps, comparison charts, and trend analysis

#### `src/main.py`
Experiment orchestration:
- Complete workflow from data generation to reporting
- Command-line interface with configurable options
- Automated result saving and visualization
- Reproducibility measures

### 3. Configuration Files

#### `config.json`
Experiment configuration:
- Data generation parameters
- Optimization settings
- Sensitivity analysis ranges
- Output preferences

#### `requirements.txt`
Python dependencies:
- pyomo (optimization framework)
- numpy, pandas (data manipulation)
- matplotlib, seaborn (visualization)

### 4. Documentation

#### `README.md`
Comprehensive documentation including:
- Project overview and problem description
- Installation instructions
- Usage examples
- Experimental results summary
- Reproducibility checklist
- Future work suggestions

#### `technical_report.md`
Full research report with:
- Introduction and motivation
- Detailed mathematical formulation
- Methodology and implementation details
- Experimental protocol
- Results and analysis
- Limitations and future work
- References

## Key Features

### 1. Modularity
- Clear separation between data, models, solvers, and analysis
- Each module can be used independently
- Easy to extend with new formulations or analyses

### 2. Reproducibility
- Fixed random seeds
- Documented parameters
- Automated workflow
- Version-controlled dependencies

### 3. Flexibility
- Configurable problem sizes
- Multiple solver support
- Adjustable sensitivity ranges
- Customizable visualizations

### 4. Completeness
- Both LP and MILP formulations
- Comprehensive sensitivity analysis
- Statistical analysis tools
- Publication-quality outputs

## Problem Formulation Summary

### Sets
- $I$: Tasks (jobs) to be scheduled
- $J$: VM types available
- $K$: Time periods in planning horizon

### Decision Variables (MILP)
- $y_{ijk} \in \{0,1\}$: Assignment of task $i$ to VM $j$ at time $k$
- $n_{jk} \in \mathbb{Z}_+$: Number of VM type $j$ active at time $k$
- $z_i \in \{0,1\}$: Task completion indicator

### Objective
$$\max Z = \sum_{i \in I} w_i r_i z_i - \sum_{j \in J} \sum_{k \in K} c_j^{vm} n_{jk}$$

### Key Constraints
1. **Budget**: Total VM costs ≤ Available budget
2. **CPU Capacity**: CPU usage ≤ Available CPU per VM
3. **Memory Capacity**: Memory usage ≤ Available memory per VM
4. **Deadlines**: Tasks completed before deadlines
5. **Service Levels**: Minimum allocation for each task

## Usage Instructions

### Quick Start
```bash
cd src
python main.py
```

### With Custom Configuration
```bash
python main.py --config ../config.json --solver cbc --seed 42
```

### Individual Components
```python
# Generate data
from data_generation import DataGenerator
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(output_dir='../data')

# Build and solve model
from model import ResourceAllocationModel
model = ResourceAllocationModel(dataset, 'MILP')
m = model.build_model()

# Solve
from solve import ResourceAllocationSolver
solver = ResourceAllocationSolver(m, 'MILP', 'cbc')
result = solver.solve()
```

## Expected Results

### Base Optimization
- **LP Objective**: Upper bound on optimal value
- **MILP Objective**: Integer optimal solution
- **Integrality Gap**: Typically 3-8% for this problem class
- **Solve Time**: LP ~0.1s, MILP ~1-10s for 50-task instances

### Sensitivity Analysis
- **Budget Constraint**: Linear improvement until saturation
- **Capacity Constraint**: Critical threshold effects
- **Shadow Prices**: Marginal value of constraint relaxation

### Visualizations
- Budget sensitivity curves
- Capacity sensitivity curves
- LP vs MILP comparison charts
- VM allocation heatmaps

## Project Structure
```
resource_allocation_project/
├── src/
│   ├── data_generation.py
│   ├── model.py
│   ├── solve.py
│   ├── analysis.py
│   └── main.py
├── data/                  # Generated datasets
├── results/               # Experimental results
├── figures/               # Generated plots
├── formulation.tex        # Mathematical formulation
├── config.json            # Experiment configuration
├── requirements.txt       # Python dependencies
├── README.md              # User documentation
├── technical_report.md    # Research report
└── PROJECT_SUMMARY.md     # This file
```

## Reproducibility Checklist

- [x] All code is modular and documented
- [x] Random seeds are fixed (default: 42)
- [x] Dependencies are specified with versions
- [x] Data generation is fully automated
- [x] Solver configurations are explicit
- [x] Experimental protocol is automated via main.py
- [x] Results are saved in structured format
- [x] Visualizations are generated programmatically
- [x] Mathematical formulation is documented in LaTeX
- [x] Limitations are clearly stated

## Future Extensions

1. **Stochastic Programming**: Incorporate uncertainty in task arrivals
2. **Dynamic Optimization**: Online decision-making with receding horizon
3. **Multi-Objective**: Pareto analysis for profit-fairness trade-offs
4. **Decomposition**: Benders/Dantzig-Wolfe for large-scale problems
5. **Real-World Data**: Validation with actual cloud workload traces

## Citation

```bibtex
@misc{resource_allocation_or2024,
  title={Constrained Resource Allocation using LP and MILP},
  author={Research Engineer},
  year={2024},
  note={Operations Research Research Project}
}
```

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: February 12, 2026
