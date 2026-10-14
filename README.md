# Task Placement Optimisation

I wanted to see how much better a scheduler could do than putting each task on the first server that fits. Each task needs CPU and memory, and each server has its own limits and costs. I compared a greedy method, a mixed integer solver and a genetic algorithm on generated data.

## Try it

Requires Python 3.11 or later.

```bash
python3 -m pip install -r requirements.txt
python3 run_experiments.py --sizes 20x4 --seeds 11 --budget 1 --repetitions 1 --output local_results
```

The code is in `scheduler.py`. The saved experiments are in `results/` and `ga_tuning/`. The [report](report.pdf) explains the model and what I found.
