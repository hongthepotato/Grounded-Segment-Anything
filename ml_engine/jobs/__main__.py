"""
Entry point for running worker via python -m ml_engine.jobs

Usage:
    python -m ml_engine.jobs --gpu 0
    python -m ml_engine.jobs --redis-url redis://localhost:6379 --gpu 0
"""

from ml_engine.jobs.worker import main

if __name__ == "__main__":
    main()
