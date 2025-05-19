# Forest-Fire CA

A stochastic 2D forest-fire cellular automaton with rich diagnostics and interactive visualization.

## Requirements

- Python 3.11

- Install dependencies
```uv pip install -r pyproject.toml```

## Quick Start

1. **Main Simulation**

    Take a peek into _forest_ca.py_ and change the hyperparameters as you wish. Then run it and wait for the simulation to finish, visualize the plots and take a look at the .gif file generated in the directory
```python src/forest_ca.py```

2. **Interactive simulation**

    Run the _live_file_dash.py_ file and open **http://127.0.0.1:8050/**
```python src/live_fire_dash.py```

3. **Helper files**

    _cmi.py_ and _sweep.py_ are merely helper files that allowed us to make the testing phase a bit quicker and helped visualize information. (cmi.py calculates the conditional mutual information of the system and sweep.py helped automate tests and visualize a heatmap)