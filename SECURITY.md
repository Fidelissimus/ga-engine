# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainers (use the
repository's private security-advisory feature if it is enabled) instead of
opening a public issue. Include a minimal reproduction and the versions of Python,
NumPy and ga-engine.

## Things to know when using ga-engine

- **Checkpoints are data, not code.** `.json` checkpoints are parsed with the
  standard `json` module and `.npz` checkpoints are read with
  `allow_pickle=False`, so loading a checkpoint does not execute code. A
  malicious file can still contain misleading *values*; only load checkpoints you
  trust.
- **Your objective function runs as-is.** The library calls the function you give
  it, in this process or, with `n_workers > 1`, in worker processes started with
  the `spawn` method. Functions are pickled by reference to their module, so a
  worker imports your module. Treat objective functions like any code you run.
- **Parallel workers import your script.** Because of `spawn`, keep import-time
  side effects out of modules that define objective functions.
- **Atomic writes.** Checkpoints are written to a temporary file in the target
  directory and moved into place; the destination directory must be writable by
  you only if you do not want others to replace checkpoints.
