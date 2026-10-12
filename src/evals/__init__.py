"""The benchmark evaluators, one sub-package per benchmark:

    src.evals.tsexam    TimeSeriesExam MCQ (746 items)         python -m src.evals.tsexam.eval
    src.evals.tsrbench  TSRBench (12 tasks, per-task protocol)  python -m src.evals.tsrbench.eval
    src.evals.chatts    ChatTS alignment A/B (vendored scorer)  python -m src.evals.chatts.eval
    src.evals.common    model loading, provenance, letter parsing shared by all three

`python scripts/eval.py --kind tsexam|tsrb|chatts|all` drives them.
"""
