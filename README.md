# solver_FLP

---

![License](https://img.shields.io/github/license/GeorgeKontsevik/solver_FLP?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style={0}&logo=tqdm&logoColor=black)

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

solver_FLP is a Python package for facility-location and service-placement optimization experiments. It is intended for researchers and developers working on optimization problems that combine exact solving with genetic search, with supporting plotting and example notebook workflows. Newcomers can use the example notebook to see the full pipeline in context, or start with Getting Started for runnable setup and usage steps.

---

## Core Features

- Solve facility-location placement problems with a combination of exact optimization and genetic search, giving developers a way to explore candidate service layouts and improve them iteratively.
- Support demand-aware placement decisions using service-radius coverage and optional existing-facility information, which helps preserve or prioritize current service points when needed.
- Compute facility capacities and selected service locations from the optimized solution, making it easier to translate a candidate layout into deployable outputs.
- Produce plots for fitness trends, facility connections, and service coverage, which helps developers inspect optimization results and present them clearly.
- Include a notebook-driven example workflow over geospatial demand data, which provides a practical starting point for experimenting with the package on real datasets.

---

## Installation

**Prerequisites:** requires Python >=3.7

Install solver_FLP using one of the following methods:

**Build from source:**

1. Clone the solver_FLP repository:
```sh
git clone https://github.com/GeorgeKontsevik/solver_FLP
```

2. Navigate to the project directory:
```sh
cd solver_FLP
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.7 or newer, as specified in `pyproject.toml`.
- The package dependencies listed in `pyproject.toml` (`numpy`, `pandas`, `geopandas`, `pulp`, `matplotlib`, `tqdm`, `pyarrow`).
- Example data files used by the notebook: `./data/data_getter_matrix.pickle` and `./data/df_with_demand.geojson`.

### Quick start

1. Install the package and its dependencies.
2. Open `examples/using_example.ipynb` in Jupyter.
3. Load the example inputs from `./data/`:

```python
accessibility_matrix = pd.read_pickle(os.path.join(example_data_path, "data_getter_matrix.pickle"))
df_with_demand = gpd.read_file(os.path.join(example_data_path, "df_with_demand.geojson")).to_crs(local_crs)
```

4. Run the notebook cells that prepare the uncovered-demand subset, build edge candidates with `choose_edges`, and execute `genetic_algorithm_main`.
5. Use `block_coverage` on the returned candidate to derive capacities and selected facility IDs.
6. If you want to run the package entry point directly, use:

```bash
python -m method
```

---

## Architecture

`solver_FLP` is a small Python package organized under `src/method` as a package-based monolith. The core modules are split by responsibility: `location_problem.py` defines the facility-location optimization model and helper functions for demand and existing-capacity data, `genetic_algorithm.py` provides a heuristic search layer, and `optimizer.py` combines those pieces into a higher-level placement routine.

The main control flow is straightforward: the optimizer can first use the genetic algorithm to search over candidate matrices, then passes the selected candidate to the location-problem solver to compute capacities and chosen facility IDs. The example notebook shows this package being used end-to-end with pandas/geopandas data, an accessibility matrix, and plotting utilities.

Plotting support lives in `plots.py`, and the package exposes a module entrypoint in `__main__.py`, although the provided code only prints a simple message when run as a module.

---

## API Reference

The public usage surface is concentrated in `src/method` and is mostly script-oriented. The key callable APIs visible in the repository are:

- `optimize_placement(...)` (`src/method/optimizer.py`) — high-level placement optimization wrapper that optionally runs the genetic search and then computes coverage/capacity results.
- `genetic_algorithm_main(...)` (`src/method/genetic_algorithm.py`) — runs the genetic search loop and returns the best candidate matrix plus fitness history.
- `choose_edges(...)` (`src/method/genetic_algorithm.py`) — builds candidate edges from a similarity/accessibility matrix and service radius.
- `calculate_fitness(...)` (`src/method/genetic_algorithm.py`) — evaluates a candidate matrix against demand and service-radius constraints.
- `solve_combined_problem(...)` (`src/method/location_problem.py`) — solves the combined facility-location/assignment problem used by the optimizer and fitness calculation.
- `block_coverage(...)` (`src/method/location_problem.py`) — used by the optimizer and example notebook to derive capacities and selected service IDs.
- `resolve_demand_series(...)` (`src/method/location_problem.py`) — resolves the demand series from input data.
- `resolve_existing_facility_mask(...)` (`src/method/location_problem.py`) — resolves an existing-facility mask from input data.
- `resolve_existing_capacity_series(...)` (`src/method/location_problem.py`) — resolves existing capacity values from input data.
- Plotting helpers in `src/method/plots.py`: `fitness_plot(...)`, `connect_blocks_plot(...)`, and `services_plot(...)` are imported by `__main__.py` and used in the example notebook.

The package also exposes a module entry point via `src/method/__main__.py`, whose `main()` currently just prints `"Running the module src.method"`.

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/GeorgeKontsevik/solver_FLP/tree/main/examples).

---

## Documentation

A detailed solver_FLP description is available [here](https://github.com/GeorgeKontsevik/solver_FLP/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/GeorgeKontsevik/solver_FLP/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/GeorgeKontsevik/solver_FLP/tree/main/CONTRIBUTING.md)**: To learn more about making a contribution to solver_FLP.

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/GeorgeKontsevik/solver_FLP/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    GeorgeKontsevik (2026). solver_FLP repository [Computer software]. https://github.com/GeorgeKontsevik/solver_FLP

### BibTeX format:

    @misc{solver_FLP,

        author = {GeorgeKontsevik},

        title = {solver_FLP repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/GeorgeKontsevik/solver_FLP}},

        url = {https://github.com/GeorgeKontsevik/solver_FLP}

    }

---