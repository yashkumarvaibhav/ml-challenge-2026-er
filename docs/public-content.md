# Public content boundaries

This repository contains source code and newly written public documentation. The final submission package was used as the source of the algorithm code, including its final assembly and the teammate CPU pipeline. Runtime artifacts from that package are deliberately excluded.

Keep the following outside public Git history:

- Challenge datasets, samples, business records, entity IDs, labels, candidate lists, and predictions.
- Learned transliteration/alias dictionaries, token vocabularies, trained LightGBM files, encoder weights, LoRA adapters, and test-specific correction lists.
- Original organizer material, FAQ exports, screenshots, transcripts, participant/team identifiers, and correspondence.
- Agent handoffs, planning discussions, raw audit notes, submission trackers, local environment freezes, infrastructure details, and backup maps.
- Logs, caches, local environments, credentials, archives, and copies of the original private repository.

The code can generate private files while running. `.gitignore` excludes their usual locations, and `scripts/check_public_release.py` checks tracked paths, text content, file sizes, and reachable Git history against a narrow source/documentation policy. Run it before each push. It rejects private files even if they were added with `git add -f`.

The content check is a safeguard, not proof that every possible secret can be recognized. Review new source literals, examples, and documentation before adding them. Demonstrations should use invented examples rather than supplied challenge records. The examples in normalization docstrings use generic names.

Use a fresh clone of the public source for future work. Do not merge or push old private-history branches or submission tags into it. Keep any original history backup private and outside this repository.
