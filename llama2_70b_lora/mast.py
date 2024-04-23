"""
Kick off training runs on MAST.

Run this script through torchx from your xlformers checkout.

    $ cd ~/xlformers
    $ torchx run mast.py:train --sweep sweeps/xyz.yaml --nodes 32
"""

import copy
import getpass
import itertools
import json
import logging
import os
import socket
import stat
import textwrap
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional

import torchx.specs as specs
import torchx.components.fb.conda as conda


logger: logging.Logger = logging.getLogger(__name__)


_DEFAULT_ENV = {
    "NCCL_ASYNC_ERROR_HANDLING": "3",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC",
    "NCCL_IB_QPS_PER_CONNECTION": "16",
    "NCCL_IB_SPLIT_DATA_ON_QPS": "0",
    "NCCL_NET_OVERHEAD": "2750",
    "NCCL_SET_THREAD_NAME": "1",
    "NVTE_BIAS_GELU_NVFUSION": "0",
    "NVTE_DISABLE_NVRTC": "1",
    "NVTE_TORCH_COMPILE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:50",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    # Mount configuration
    "STORAGE": "oilfs",
    "FUSE_DST": "/mnt/wsfuse",
    "NFS_ROOT_DIR": "/mnt/gen_ai_input_data_nfs/aidev",
    "NFS_MOUNT_DIR": "/mnt/aidev",
}

_STORAGE_SRC = {
    #"oilfs": "oil://warmstorage_fuse_ai_pci_meta/warmstorage_fuse_ai_pci_data/genai_fair_llm",
    "oilfs":"ws://ws.ai.eag0genai/genai_fair_llm",
    "wsfs": "ws://ws.ai.eag0genai/genai_fair_llm",
}

_STORAGE_PACKAGES = {
    "oilfs": "oil.oilfs:stable",
    "wsfs": "warm_storage.fuse:prod",
}

#_RUN_SCRIPT = "run_nocont.sub"
#_RUN_SCRIPT = "scripts/torchx_mount_and_run.sh"
_RUN_SCRIPT = "scripts/interactive_setup.sh"


def train(
    *script_args: str,
    script: str = "train.py",
    sweep: Optional[str] = None,
    sweep_index: int = 0,
    preview: bool = False,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "xlformers",
    h: str = "grandteton",
    env: Optional[Dict[str, str]] = None,
    retry_policy: Optional[str] = None,
    run_as_root: bool = False,
    dump_dir_id: str = "${app_id}",
    conda_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
) -> specs.AppDef:
    """
    Kick off a training job on MAST.
    Sane defaults are specified in the .torchxconfig.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        sweep: name of yaml file for parameters
        sweep_index: in case there are mulitple possible runs, choose the run to kick off
        nodes: total hosts to use
        nproc_per_node: processes per node
        preview: see the available commands
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        retry_policy: as title
        run_as_root: run the job as root; should be set to true for mounting
        dump_dir_id: Explicitly specify an mast job to continue training (defaults to new job id)
        conda_dir: path to conda in NFS relative to your homedir
        workspace_dir: path to code in NFS relative to your homedir
    """

    # Set up the environment variables
    mast_env = dict(_DEFAULT_ENV)
    username = getpass.getuser()
    run_script = "${img_root}/" + _RUN_SCRIPT

    if conda_dir:
        mast_env["CONDA_DIR"] = os.path.join(
            mast_env["NFS_MOUNT_DIR"], username, conda_dir
        )

        # TODO: Remove once torchx ships (D48690192)
        mast_env["PYTHON_EXEC"] = os.path.join(mast_env["CONDA_DIR"], "/bin/python")

    if workspace_dir:
        mast_env["WORKSPACE_DIR"] = os.path.join(
            mast_env["NFS_MOUNT_DIR"], username, workspace_dir
        )

        # The mounting script is part of the code that is in NFS but hasn't been mounted yet
        # TODO: Should we move it to a shared NFS directory that isn't user editable?
        run_script = os.path.join(
            mast_env["NFS_ROOT_DIR"], username, workspace_dir, _RUN_SCRIPT
        )

    if env:
        mast_env.update(env)

    _ensure_fuse_src(mast_env)

    # Set up arguments for fb.conda.torchrun
    kwargs = {
        "name": name,
        "h": h,
        "env": mast_env,
        "retry_policy": retry_policy,
        "run_as_root": run_as_root,
    }
    for key in list(kwargs.keys()):
        if kwargs[key] is None:
            kwargs.pop(key)

    # Fetch sweep parameters
    if sweep:
        sweep_args = _get_all_params(sweep)
    else:
        sweep_args = [{}]

    # Ensure that a dump dir is available
    fuse_dst = Path(mast_env["FUSE_DST"])
    dump_dir = fuse_dst / "outputs" / dump_dir_id

    # Construct arguments per sweep
    full_args = [
        [
            "--tee",
            "3",
            "--nnodes",
            str(nodes),
            "--nproc-per-node",
            str(nproc_per_node),
            "--no-python",
            run_script,
            script,
            *_args_dict_to_args_list(
                {
                    **sweep,
                    "dump_dir": dump_dir,
                }
            ),
            *script_args,
        ]
        for sweep in sweep_args
    ]

    if preview:
        _print_preview(script, full_args, sweep, sweep_index)
        exit(1)

    job_spec = conda.torchrun(*full_args[sweep_index], **kwargs)
    job_spec.roles[0].image = ";".join(
        (
            job_spec.roles[0].image,
            _STORAGE_PACKAGES[mast_env["STORAGE"]],
        )
    )
    # Set a custom base image
    job_spec.roles[0].metadata["mast"] = {
        "HpcTaskGroupSpec": {
            "baseImage": {
                "baseImagePackage": {
                    "fbpkgIdentifier": "tupperware.image.sendstream.c9.podman",
                }
            }
        }
    }

    return job_spec


def train_interactive(
    *script_args: str,
    script: str = "train.py",
    sweep: Optional[str] = None,
    sweep_index: int = 0,
    preview: bool = False,
    nodes: int = 1,
    nproc_per_node: int = 8,
    name: str = "xlformers",
    h: str = "grandteton",
    env: Optional[Dict[str, str]] = None,
    retry_policy: Optional[str] = None,
    run_as_root: bool = False,
    conda_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    sleep_hrs: int = 4,
) -> specs.AppDef:
    """
    Minimal support for interactive workflows.

    Updates your trainer job to run on a single node, and then runs sleep on the job.

    This command will generate an `interactive.sh` file, available at
    $WORKSPACE_DIR which you can use to quickly run the command that
    would have been run automatically for you.

    It will also create a "sync.sh" file to copy over any changes you make back to your
    devserver.

    Must be run in the repo root to function correctly.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        sweep: name of yaml file for parameters
        sweep_index: in case there are mulitple possible runs, choose the run to kick off
        nodes: Ignore, provided for parity with train(); over-ridden to a single node
        nproc_per_node: processes per node
        preview: see the available commands
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        retry_policy: as title
        run_as_root: ignored, set to true for interactive jobs for easier debugging
        sleep_hrs: how long to hold the host in hours; maxes out at 8
        conda_dir: path to conda in NFS, if set there'll be no conda fbpkg attached
        workspace_dir: path to code in NFS, if set there'll be no workspace fbpkg attached
    """
    job_spec = train(
        *script_args,
        script=script,
        sweep=sweep,
        sweep_index=sweep_index,
        preview=preview,
        nodes=1,
        nproc_per_node=nproc_per_node,
        name=f"interactive-" + name if name else "interactive-conda-xlformers",
        h=h,
        env=env,
        retry_policy=retry_policy,
        run_as_root=True,
        conda_dir=conda_dir,
        workspace_dir=workspace_dir,
    )

    original_cmd = job_spec.roles[0].entrypoint
    original_args = job_spec.roles[0].args

    with open("interactive.sh", "w") as interactive_script:
        # Make the command more readable
        args_str = ""
        for arg in original_args:
            if not arg:
                continue

            if arg.startswith("--"):
                args_str += " \\\n "
            args_str += " " + arg

        # Fix the macros
        args_str = args_str.replace("${app_id}", "$MAST_HPC_JOB_NAME")
        args_str = args_str.replace("${img_root}", "$WORKSPACE_DIR")
        args_str = args_str.replace(
            "${torchx_torchrun_root}", "/packages/torchx_torchrun"
        )

        interactive_script.write(
            textwrap.dedent(
                f"""
                #!/bin/bash
                set -ex
                {original_cmd} \\
                """
            ).lstrip()
        )
        interactive_script.write(" " + args_str + "\n")
    os.chmod("interactive.sh", os.stat("interactive.sh").st_mode | 0o111)

    # Only generate a sync file if the workspace dir is not from NFS
    if not workspace_dir:
        with open("sync.sh", "w") as sync_script:
            sync_script.write(
                textwrap.dedent(
                    f"""
                    #!/bin/bash
                    echo "Syncing back changes to the repo (not attempting to sync changes to conda)"

                    set -ex
                    scp -r /tmp/workspace-upperdir/* {getpass.getuser()}@{socket.gethostname()}:{os.getcwd()}
                    """
                ).lstrip()
            )
        os.chmod("sync.sh", os.stat("sync.sh").st_mode | 0o111)

    commands = []

    if not workspace_dir:  # Workspace comes from an FBPkg
        commands.extend(_overlay_commands(label="workspace", lowerdir="$WORKSPACE_DIR"))

    if not conda_dir:  # Conda comes from an FBPkg
        commands.extend(_overlay_commands(label="conda", lowerdir="$CONDA_DIR"))

    # TODO Make this based on a file with expiry time
    commands.append(f"sleep {min(sleep_hrs * 3600, 8 * 3600)}")

    job_spec.roles[0].entrypoint = " && \\\n  ".join(commands)
    job_spec.roles[0].args = []

    return job_spec


def _print_preview(
    script: str, full_args: List[List[str]], sweep: Optional[str], sweep_index: int
) -> None:
    # TODO: Improve the output
    if sweep:
        print(
            f"`{sweep}` contained {len(full_args)} configurations; "
            f"this command will run configuration {sweep_index}."
        )

    for i in range(len(full_args)):
        selected = " SELECTED" if i == sweep_index else ""
        print(f"Configuration {i}:{selected}")
        print("torchx run fb.conda.torchrun \\")
        print(f"{full_args[i]}")

    print("You can run `torchx run --dryrun` to see the full generated app-def.")


def _ensure_fuse_src(mast_env: Dict[str, str]) -> None:
    """In case the user hasn't specified a source, choose based on $STORAGE"""
    if "FUSE_SRC" in mast_env:
        return

    storage = mast_env["STORAGE"]
    if storage not in _STORAGE_SRC:
        raise ValueError(
            f"Unknown storage type {storage}! Please set env STORAGE explicitly"
        )
    mast_env["FUSE_SRC"] = _STORAGE_SRC[storage]


def _overlay_commands(label: str, lowerdir: str) -> List[str]:
    workdir = f"/tmp/{label}-workdir"
    upperdir = f"/tmp/{label}-upperdir"

    return [
        f"mkdir {workdir} {upperdir}",
        (
            "mount -t overlay overlay "
            f"-o lowerdir={lowerdir},upperdir={upperdir},workdir={workdir} {lowerdir}"
        ),
    ]


# TODO: This is derived from stool.py
#       To share as a dep after checking with xlformers owners


def _get_all_params(file_name) -> List[Dict[str, Any]]:
    if not os.path.isfile(file_name):
        logger.exception(f"Error: sweep file {file_name} does not exist!")
        exit(1)

    if file_name.endswith(".yaml"):
        grid = _read_yaml(file_name)
    else:
        grid = json.load(file_name)

    if not all(isinstance(v, dict) for v in grid.values()):
        return _parse_grid(grid)

    all_params = []
    for v in grid.values():
        all_params.extend(_parse_grid(v))

    return all_params


def _read_yaml(file_name) -> Dict[str, Any]:
    with open(file_name) as f:
        grid = yaml.full_load(f.read())

    final_grid = {}
    for include in grid.pop("include", []):
        final_grid.update(_read_yaml(include))

    final_grid.update(grid)
    return final_grid


def _parse_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    subsweep_keys = [k for k in grid.keys() if k.startswith("SUBSWEEP")]
    all_swept_params = []

    if len(subsweep_keys) > 0:
        subsweep_grids = itertools.product(*[grid[k].values() for k in subsweep_keys])

        for k in subsweep_keys:
            del grid[k]

        for subsweep in subsweep_grids:
            new_grid = copy.deepcopy(grid)
            for v1 in subsweep:
                for k2, v2 in v1.items():
                    new_grid[k2] = v2
            all_swept_params.extend(_parse_grid(new_grid))
        return all_swept_params

    perms = list(itertools.product(*grid.values()))
    for p in perms:
        swept_params = {}
        for i, k in enumerate(grid.keys()):
            if p[i] is not None:  # to avoid setting optional parameters
                swept_params[k] = p[i]
        all_swept_params.append(swept_params)

    return all_swept_params


def _args_dict_to_args_list(swept_params: Dict[str, Any]) -> List[str]:
    args_list = []

    for key, value in swept_params.items():
        if value is None:  # to avoid setting optional parameters
            continue

        v = (
            json.dumps(value)
            if type(value) is list or type(value) is dict
            else str(value)
        )
        args_list.append(f"--{key}={v}")

    return args_list
