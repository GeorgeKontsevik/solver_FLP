# solver_flp

Facility location solver with exact and genetic modes.

## Scheme

```mermaid
flowchart LR
    A[Inputs] --> B[Run: examples/using_example.ipynb]
    B --> C[Checked outputs]
    C --> D[Paper / thesis use]
```

## Main Result

![Main result](docs/readme_result.svg)

## Run

Entrypoint: `examples/using_example.ipynb`

Human:

```bash
pip install -e . && jupyter notebook examples/using_example.ipynb
```

Agent:

Always state exact/non-genetic vs genetic mode and whether existing services can expand.

## Publication

No standalone publication tracked.

## Next Steps / Heuristics

Heuristic: prefer explicit flags over implicit solver defaults; track demand_left as practical unmet demand.
