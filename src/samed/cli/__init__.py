"""Command-line entry points, one per pipeline stage.

Each stage is a separate process so that it can be a separate Slurm array job
with its own partition, walltime and shard count, and so that a failure is
contained to one stage of one shard.
"""
