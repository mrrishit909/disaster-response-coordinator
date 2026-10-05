# Performance

Docker stack (one uvicorn process, Postgres 16) after the demo, at hour 6 of the scenario: 140 neighbourhoods, 250 road segments,
78 incidents, 24 units. Reproduce with `make demo && make load-test`.

## Reads: map, incident search, isolation forecast, shelter allocation (in rotation), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 2091 | 209.0 req/s | 4.8 ms | 6.3 ms | 6.9 ms | 0.00% |
| 8 | 2560 | 255.2 req/s | 29.5 ms | 44.4 ms | 53.5 ms | 0.00% |
| 32 | 2074 | 205.3 req/s | 124.1 ms | 149.8 ms | 169.1 ms | 0.00% |

Target from the blueprint: p95 under 400 ms for reads: met at every level. Throughput does not rise past 8 clients because one
Python process serves everything; the shelter allocation solves a small LP on every call.

The isolation forecast (300 simulated storms, about 0.1 s) is memoised per scenario hour and hospital status, in process memory.

## Compute

| Work | Time |
|---|---|
| Seed: fit and score the report classifier | about 1 s |
| Advance two hours: 118 reports classified and merged, 66 road closures | about 0.1 s |
| Dispatch: 51 incidents x 24 units, two all-pairs shortest-path runs + exact assignment | about 0.02 s |

Not measured: a real county. This one has 140 nodes; a road network with 100,000 segments would need the shortest-path runs
restricted to the units' surroundings, and a Monte Carlo forecast run as a background job rather than inside a request.
