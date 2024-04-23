# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import getpass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import facebook.hpc_scheduler.hpcscheduler.types as mast
import torchx.components.fb.conda as conda
import torchx.specs as specs
from torchx.specs.fb.named_resources import MAST_NETWORK_AFFINITY

logger: logging.Logger = logging.getLogger(__name__)


_DEFAULT_ENV = {
    "NCCL_DEBUG": "INFO,WARN",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "STORAGE": "oilfs",  # TODO: Flip this for storage
    "FUSE_DST": "/mnt/wsfuse",
    "DISABLE_NFS": "1",
    "TORCH_ADDR2LINE_BINARY": "/packages/folly.symbolizer/folly-addr2line",
}

DEFAULT_ARGS = {
    "test.py": {
        # Required for alerting & getting metrics on a dashboard.
        "enable_ods": True,
    }
}

_STORAGE_PACKAGES = {
    "oilfs": "oil.oilfs:stable",
}

_MOUNT_SCRIPT = "$WORKSPACE_DIR/mount.sh"
_PY_SPY_SCRIPT = "py_spy_startup.sh"
_TEE_SCRIPT = "torchx_tee.sh"
_RUN_SCRIPT = "$WORKSPACE_DIR/run_mlperf_llama.sh"


def train(
    *script_args: str,
    script: str = "test.py",  # wenyin: not really used currently
    module: Optional[str] = None,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "cpu_nccl_init",
    h: str = "t1",
    env: Optional[Dict[str, str]] = None,
    unset_env: Optional[List[str]] = None,
    retry_policy: Optional[str] = None,
    run_as_root: bool = False,
    dump_dir_id: str = "${app_id}",
    conda_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    nfs_dump: bool = False,
    xzone: bool = False,
    dump_logs: bool = True,
    additional_libraries: Optional[List[str]] = None,
    additional_folders: Optional[List[str]] = None,
    additional_python_paths: Optional[List[str]] = None,
    py_spy_startup: bool = False,
    retries: int = 1,
    twtask_bootstrap_script: str = None,
) -> specs.AppDef:
    """
    Kick off a training job on MAST.
    Sane defaults are specified in the .torchxconfig.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        module: if provided, run Python module instead of script
        sweep: name of yaml file for parameters
        sweep_index: in case there are mulitple possible runs, choose the run to kick off
        nodes: total hosts to use
        nproc_per_node: processes per node
        preview: see the available commands
        preview_local: preview the sweep command for running locally
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        unset_env: environment parameters to unset/delete
        retry_policy: as title
        run_as_root: run the job as root; should be set to true for mounting
        dump_dir_id: Explicitly specify an mast job to continue training (defaults to new job id)
        conda_dir: path to conda in NFS relative to your homedir
        workspace_dir: path to code in NFS relative to your homedir
        nfs_dump: use NFS instead of fuse to write checkpoints and logs. DO NOT USE UNLESS FUSE IS BROKEN.
        xzone: enable cross zone jobs
        dump_logs: save logs to dump dir as well under "<dump dir>/logs"
        additional_libraries: copy these folders into xlformers_pretrain2 and add them to python path
        additional_folders: copy these folders into the fbpkg xlformers_pretrain2
        additional_python_paths: add these paths to $PYTHONPATH before executing
        py_spy_startup: trace script startup; see scripts/mast/py_spy_init.sh for configuration
        retries: number of times to retry the job before failing completely
        twtask_bootstrap_script: shell script that is run on each tw task which bootstraps the real training script
    """

    # Set up the environment variables
    mast_env = dict(_DEFAULT_ENV)
    username = getpass.getuser()
    run_script = (
        "${img_root}/" + twtask_bootstrap_script
        if twtask_bootstrap_script
        else _RUN_SCRIPT
    )
    tee_script = "$WORKSPACE_DIR/" + _TEE_SCRIPT
    py_spy_script = "$WORKSPACE_DIR/" + _PY_SPY_SCRIPT

    if conda_dir:
        mast_env["CONDA_DIR"] = os.path.join(
            mast_env["NFS_MOUNT_DIR"], username, conda_dir
        )

    if workspace_dir:
        mast_env["WORKSPACE_DIR"] = os.path.join(
            mast_env["NFS_MOUNT_DIR"], username, workspace_dir
        )
        tee_script = os.path.join(
            mast_env["NFS_ROOT_DIR"], username, workspace_dir, _TEE_SCRIPT
        )

    # Ensure that a dump dir is available
    dump_mount = Path(mast_env["FUSE_DST"])
    if nfs_dump:
        dump_mount = Path(mast_env["NFS_DIR"])
        mast_env["NFS_DUMP_READ"] = "1"
    dump_dir = dump_mount / "outputs/wenyinfu/torchtrain"

    # Make the dump dir available for shell scripts
    mast_env["DUMP_DIR"] = str(dump_dir)
    mast_env["JOB_ID"] = dump_dir_id

    # Dependencies libraries for picking up latest site package
    additional_python_paths = additional_python_paths or []
    additional_folders = additional_folders or []
    additional_libraries = additional_libraries or []
    additional_pkg = "mlperf_llama2_ft_dataset"

    if additional_libraries or additional_folders:
        additional_folders.extend(additional_libraries)
        result = _make_fbpkg(additional_folders)
        additional_pkg = result.identifier
        for folder in additional_libraries:
            additional_python_paths.append(
                f"/packages/{result.package}/{os.path.basename(folder.rstrip('/'))}"
            )

    mast_env["TORCHX_RUN_PYTHONPATH"] = ":".join(additional_python_paths)

    if env:
        mast_env.update(env)

    if unset_env:
        for env_var in unset_env:
            mast_env.pop(env_var, None)

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

    if nodes == 1:
        rdzv_args = ["--standalone"]
    else:
        rdzv_args = ["--rdzv_backend", "zeus"]

    # Construct arguments per sweep
    sweep_args_list = [
        _args_dict_to_args_list(
            {
                "dump_dir": dump_dir,
                **DEFAULT_ARGS.get(module or script, {}),
            }
        )
    ]
    full_args = [
        [
            "--rdzv_id",
            specs.macros.app_id,
            *rdzv_args,
            "--tee",
            "3",
            "--nnodes",
            str(nodes),
            "--nproc-per-node",
            str(nproc_per_node),
            "--role",
            "trainer",
            "--no-python",
            run_script,
            # f"-m{module}" if module else script,
            *script_args,
        ]
    ]

    if not dump_logs:
        tee_script = ""

    if not py_spy_startup:
        py_spy_script = ""

    job_spec = conda.torchrun(*full_args[0], **kwargs)

    inner_entrypoint = job_spec.roles[0].entrypoint
    conda_prep = ""
    # if supervisor:
    #     conda_prep = "&& . $CONDA_DIR/bin/activate && cd $WORKSPACE_DIR"
    #     inner_entrypoint = "$PYTHON_EXEC -m src.startup.supervise"

    entrypoint = f"{_MOUNT_SCRIPT} {conda_prep} && {tee_script} {py_spy_script} {inner_entrypoint}"

    print(f"wenyin: {entrypoint=}")

    job_spec.roles[0].entrypoint = entrypoint

    packages = [
        job_spec.roles[0].image,
        _STORAGE_PACKAGES[mast_env["STORAGE"]],
        "nfs.twmount:stable",
        "torchx_conda_mount:stable",
        "folly.symbolizer:stable",
    ]
    if additional_pkg:
        packages.append(additional_pkg)

    if py_spy_startup:
        packages.append("fb-py-spy:prod")
    job_spec.roles[0].image = ";".join(packages)

    if xzone:
        logger.warn(
            f"[NOTE] The --xzone parameter is being moved from component args toscheduler_args;"
            'Please set it with --scheduler_args="xzone=True" in the future!'
        )
        job_spec.roles[0].resource.capabilities[MAST_NETWORK_AFFINITY] = (
            _allow_xzone_affinity(nodes)
        )

    # Default to 3 retries; will be over-ridden by perpetualRun if set
    # 2 is not a typo here, it will attempt 0, 1 and 2
    job_spec.roles[0].max_retries = retries

    # Set a custom base image
    job_spec.roles[0].metadata["mast"] = {
        "HpcTaskGroupSpec": {
            "baseImage": {
                "baseImagePackage": {
                    "fbpkgIdentifier": "tupperware.image.sendstream.c8.flare",
                }
            }
        }
    }

    return job_spec


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


def _make_fbpkg(paths: List[str]):
    """
    Temporarily create a secondary fbpkg with additional folders the user wants
    packaged up.

    After code freeze we can update the TorchX Conda Workspace we can delegate packaging
    everything into a single fbpkg there.
    """
    import libfb.py.fbpkg as fbpkg  # works because torchx has this built in

    build_results = fbpkg.build_version(
        pkg_name="infra_mlperf_llama",
        build_config=fbpkg.BuildConfig(
            paths=paths,
        ),
        ephemeral=True,
        expire="4w",
        silent_duplicate_error=True,
    )
    return build_results[0]


def _allow_xzone_affinity(nodes: int):
    """
    Allow using multi zone network affinities.

    Must be kept in sync with https://fburl.com/code/lsdjlk0u for MAST validation.
    """
    if nodes >= 512:
        return mast.NetworkAffinity(
            preferredScope=mast.NetworkAffinityScope.MULTI_ZONE,
            fallbackScope=mast.NetworkAffinityScope.MULTI_ZONE,
        )

    return mast.NetworkAffinity(
        preferredScope=mast.NetworkAffinityScope.BACKEND_NETWORK,
        fallbackScope=mast.NetworkAffinityScope.MULTI_ZONE,
    )
