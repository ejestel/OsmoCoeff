# Environment setup

The environment is called `polymerist-osmocoeff` and is used for both simulation and analysis.
It targets 64-bit Linux. GPU runs need an NVIDIA GPU.

| File | Purpose |
|---|---|
| `environment.yml` | Readable list of requirements. A fresh solve may pick newer versions. |
| `lock/osmocoeff-linux-64.txt` | Exact conda packages |
| `lock/osmocoeff-pip.txt` | Exact pip package versions |
| `check_environment.py` | Checks water, ion and net charges, and CUDA |
| `check_gpu.sbatch` | Runs the check as a one-GPU Slurm job |

## Install

Run all commands from the repository root. Install into a **new** environment (see
[Why these versions](#why-these-versions)). `PYTHONNOUSERSITE=1` stops packages in `~/.local`
from leaking into the environment.

**Recommended: exact versions**

```bash
export PYTHONNOUSERSITE=1
conda create -n polymerist-osmocoeff --file Setup/lock/osmocoeff-linux-64.txt
conda activate polymerist-osmocoeff
python -m pip install --no-deps -r Setup/lock/osmocoeff-pip.txt
python -m pip check
```

On a cluster where environments live outside your home directory, replace
`-n polymerist-osmocoeff` with `--prefix /path/to/env`, and activate that path.

**Alternative: new solve from the requirements**

```bash
export PYTHONNOUSERSITE=1
CONDA_CHANNEL_PRIORITY=strict mamba env create -f Setup/environment.yml
conda activate polymerist-osmocoeff
python -m pip check
```

## Check

On a computer with a GPU:

```bash
python Setup/check_environment.py --require-gpu
```

Without a GPU, leave out `--require-gpu`; the CUDA test is then reported as skipped if it fails.

On a Slurm cluster, submit the check as a batch job:

```bash
sbatch Setup/check_gpu.sbatch
```

The `#SBATCH` lines and the default environment path are for CU Boulder's `blanca-shirts` nodes.
Change them for your cluster, or override them on the command line. To use another environment path:
`OSMOCOEFF_ENV=/path/to/env sbatch Setup/check_gpu.sbatch`.

Do not start simulations unless the output ends with `All required checks passed`.
The check covers a small test system. It does not validate your simulation boxes.

## OpenEye (optional)

Requires your own license. It is deliberately left out of the lock files. Add it after the base install:

```bash
mamba install -n polymerist-osmocoeff --freeze-installed -c openeye -c conda-forge openeye-toolkits
export OE_LICENSE=/full/path/to/oe_license.txt     # never commit the license file
python -c "from openeye import oechem; print(oechem.OEChemIsLicensed())"   # expect True
```

With a cluster path, use `-p /path/to/env` instead of `-n polymerist-osmocoeff`.
Review the install plan before accepting it, then run the check again.
The setup check has passed with OpenEye 2025.2.3 and 2026.1.0. Machines do not have to use the same
version. Record the version used for each set of runs (`conda list openeye`).

## Why these versions

- **polymerist 1.1.0.** Version 1.0.2, used for earlier runs, gave TIP3P oxygen a charge of
  -0.843 e instead of -0.834 e, so each water carried -0.009 e. Do not upgrade an old environment
  in place: polymerist creates `water_TIP3P.sdf` only when the file is missing, so an old copy
  could survive.
- **CUDA 11.8, CPU-only PyTorch and JAX.** A CUDA 12.9 build failed on a driver supporting CUDA 12.2
  (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). OpenMM still runs on the GPU. PyTorch and JAX do not.
- **conda first, then pip.** Installing polymerist with pip first left mixed pip and conda copies of
  OpenMM and other packages.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `CUDA_ERROR_NO_DEVICE` | Check that your job was given a GPU: `nvidia-smi -L` and `echo $SLURM_JOB_GPUS`. Being on a GPU node is not enough. Use the batch check. |
| `mamba` stops with no message | Run `echo $?` immediately afterwards. `137` means the process was killed, often for lack of memory. Retry in a job with more memory. |
| `new stuff` is printed | Harmless leftover print in polymerist 1.1.0. |
| `openff-forcefields not found in package metadata` | Affects only the version listing. |
| `Run 'mamba init'...` in a job log | Seen on CU Boulder in a job that passed. `mamba init` was not needed. |

## Updating the lock files

Only after the check passes, and with OpenEye filtered out:

```bash
conda list --explicit --md5 | grep -vi openeye > Setup/lock/osmocoeff-linux-64.txt
conda list | awk '$4=="pypi" && tolower($1) !~ /openeye/ {print $1"=="$2}' > Setup/lock/osmocoeff-pip.txt
```

If installing OpenEye pulled in other packages, export from an environment without OpenEye instead.

## Tested

2026-11-25: all checks, including CUDA, passed on:

- a local Linux workstation (installed from `environment.yml`; the lock files were exported from it);
- a CU Boulder cluster environment installed from those lock files, tested in batch jobs on an
  NVIDIA A40 node (`blanca-shirts`) and on one preemptable NVIDIA H100 node.

The local and cluster environments had identical non-OpenEye package lists. Other preemptable nodes
may have different GPUs or drivers, so run the check at the start of each job on them.

To use a preemptable node on the CU Boulder cluster:

```bash
sbatch --partition=blanca --qos=preemptable Setup/check_gpu.sbatch
```
