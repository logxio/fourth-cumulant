# Frozen checks of Tables 2b and 3

`PREREG.md`, `PREREG.sha256`, the saved logs and truth arrays retain the original evidence. The timestamp in `PREREG.sha256` is a recorded local timestamp; it is not an independent timestamping service.

The original driver hash in `PREREG.sha256` refers to the original checkout-specific script. The released `run.recorded.sh` changes only line 4 to resolve the repository root relative to the script. `PORTABLE.sha256` authenticates this relocated copy and the unchanged preregistration. The original driver hash is retained as historical evidence, rather than asserted to match the relocated file.

Use `python3 research/frozen_s16.py check` to check the 122 saved values, preregistration hash, released driver hashes, run timestamps, all completed steps, and the three criteria. Use `python3 research/frozen_s16.py rerun` for the portable implementation of the same commands; it writes new output under `outputs/frozen_s16/`. `bash report/data/frozen_s16/run.sh` calls this portable Python runner and forwards its options. The recorded shell recipe uses macOS `/usr/bin/time -l` and GNU `timeout`; it is retained as evidence.

The Python runner enforces a 295-second and 4-GiB limit per subprocess. A complete run takes about 40 minutes; choose `--steps e1_64_8_301 --truth fresh` for the two-method small cell. Research dependencies and the pinned ARC reference in `REPRODUCE.md` are required for reruns. Saved-evidence checks use the standard library.
