# Independent CPU matcher

Pinaksh Kumar's CPU matcher supplies the second opinion used by the final US/India tie-breaker, along with France house-number veto and filler-rescue code. Its source is included under the repository's MIT license with contributor approval.

This public directory contains implementation and training scripts. Trained `models2/` and `models3/`, datasets, learned maps, predictions, and the original private packaging notes are excluded. A fresh clone therefore cannot score test records with frozen submission models.

Run CPU commands from this directory in a separate Python 3.12 environment. Its requirements file preserves the recorded CPU environment pins; this source release has not reinstalled or retrained that environment. Prepare your authorized private dataset as `student_resource/dataset/` under this directory, or link that directory locally to your dataset. Dataset contents must remain outside Git.

```bash
# From this directory, with a dedicated environment already installed:
export PY="$(command -v python)"
export BER_ROOT="$PWD" PYTHONPATH=src POLARS_MAX_THREADS=8
mkdir -p logs
bash run_prep.sh
bash run_all2.sh
bash run_all3.sh
bash run_s2b.sh
```

Scripts skip existing cached outputs. Review those conditions before a retraining run; stale caches are not evidence of a completed new run. Historical hardware was an 8-core laptop with 15 GB RAM.

To generate the auxiliary outputs for the final assembly, `run_sub11_plus.sh` requires the main pipeline's stage-2 predictions and both TSV files for each of the private #27-7 and #27-11 submissions. Pass `ER_TEAM_PREDS`, `ER_SUB7`, and `ER_SUB11` explicitly. Its output under `output/sub7_fr/` includes the France veto list, and `output/sub11_plus/` contains the independent blend variant. The final team pipeline uses these as inputs rather than substituting the entire CPU output.

See the [main reproduction guide](../../../../../docs/reproduction.md) for how these inputs relate to the final assembly. Learned artifacts and the resulting record identifiers remain private.
