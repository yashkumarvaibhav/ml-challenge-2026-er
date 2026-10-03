# Business entity resolution source

This directory contains the main pipeline from GitBoosters' final #27-19 submission, including the later assembly steps and the independent CPU matcher. The public leaderboard score recorded for that submission is 0.989275.

Start with the [repository README](../../README.md), [methodology](../../docs/methodology.md), and [reproduction guide](../../docs/reproduction.md). Run the main scripts from the repository root with `PYTHONPATH=code/business_entity_resolution/src` and `BER_ROOT` set to that root.

Unlike the original submission archive, this directory has no learned artifacts, weights, adapters, predictions, challenge data, or raw reports. `stage_artifacts.sh` and the historical `reproduce_final*.sh` entry points need separately held private inputs. The final entry point is `reproduce_final_2719.sh`; earlier scripts produce intermediate submissions, not the final output by themselves.
