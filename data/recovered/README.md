# Recovered Modal records

Recovered on 17 October 2026 using the user's old Modal account. `modal/`
preserves the small remote records under their volume-relative paths. `sources/`
contains exact UTF-8 worker source strings extracted from those requests, named
by SHA-256. These are original executable sources, not recreated implementations.

The archive contains no original notebooks. The root notebook
`phase6-7-recovered-continuation.ipynb` is explicitly reconstructed for inspection
and continuation. Original local-only reviews and protocol files that were never
uploaded cannot be recovered from the volume. Published summaries preserve their
findings, but must not be passed off as those original files.

`data/recovery/restored-sources.json` records how published files were restored.
Every restored file matches the published SHA-256 exactly, including historical
Windows line endings where necessary. Full volume inventories are under
`data/recovery/`. Large tensors remain outside Git in `.archive/`; local download
ledgers record their sizes and SHA-256. Remote data is preserved unchanged.

Use `tools/download_modal_archive.py --old-account` to resume archive reads;
no credentials are saved in the repository. Uploads and computation select the
new account explicitly through `tools/modal_accounts.py`. The default
size cap recovers small artifacts. Explicitly select large checkpoint paths with
`--prefix` and `--max-file-gib`; historical checkpoints total hundreds of GiB.
Use separate `--ledger` filenames for concurrent disjoint selections.

`tools/upload_modal_archive.py` transfers verified local files to the new account.
`tools/materialize_model_cache.py` verifies original LFS/Git blob hashes and
reconstructs cache aliases locally without duplicating large files.
`tools/verify_new_account.py` checks destination files and recreates cache symlinks
using a CPU job in the new account. All continuation dependencies are now verified
locally, so the old credentials may be revoked; destination transfers and offline
replay use only the new account. The migration status file records this boundary.

Monitoring and verification, from the repository root:

```powershell
.venv/Scripts/python.exe tools/experiment_status.py   # sync progress of registered runs; schedules no model work
python tools/verify_recovered_tensors.py              # verify restored tensors against the ledgers
```

The recovered end of Phase 6 is read in the final section of `phase6.ipynb`; the recovered
Phase 7 history opens `phase7-adaptive-pruning.ipynb`. The reconstructed
`phase6-7-recovered-continuation.ipynb` was folded into those two notebooks on 18 October 2026.
