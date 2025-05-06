import sys
import os
import signal
import subprocess
from hydra.core.hydra_config import HydraConfig
from ecglib.version import COMMIT_HASH


def handle_main_process_sigterm(signum, frame, trainer):
    """This function manage behaviour when main process is terminated by user

    Args:
        signum (int): number of signal
        frame (signal.frame): current stack frame
    """
    print("Parent process received SIGTERM. Terminate all child processes...")
    trainer.stop_train()
    sys.exit(0)


def handle_client_process_sigterm(signum, frame, rank):
    """This function manage behaviour when child process is terminated by user

    Args:
        signum (int): number of signal
        frame (signal.frame): current stack frame
    """
    print(f"Child process {rank} received SIGTERM. Send it to parent...")

    os.kill(os.getppid(), signal.SIGTERM)  # Send SIGTERM to parent process


def get_repository_info(do_print: bool = False) -> dict:
    """Create dict with information about current repository state
    to be reproduced later

    :param do_print: Whether to print repository info

    :return: Dictionary with commit hash and diff
    """

    git_info = {
        "commit_hash": subprocess.run(
            "git rev-parse --verify HEAD".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .replace("\n", ""),
        "repo_name": subprocess.run(
            "git rev-parse --show-toplevel".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .split("/")[-1]
        .replace("\n", ""),
        "ecglib_commit_hash": COMMIT_HASH,
        "project_name": subprocess.run(
            "git rev-parse --show-toplevel".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .split("/")[-1]
        .replace("\n", ""),
    }
    return git_info


def get_run_command():
    script_name = sys.argv[0]
    args = " ".join(sys.argv[1:])
    run_dir = HydraConfig.get().run.dir
    run_command = f"CUDA_VISIBLE_DEVICES=0 nohup python {script_name} {args} > {run_dir}/train_log.txt &"
    return run_command


def create_model_info(model_state, metrics, checkpoint_path, cfg):

    model_info = {
        "git": get_repository_info(),
        "run_command": get_run_command(),
        "model": model_state,
        "metrics": {"metrics": metrics[0], "threshold": metrics[1]},
        "config_file": cfg,
    }
    filename = f"{HydraConfig.get().run.dir}/experiment_report.txt"
    with open(filename, "w") as f:
        f.write(
            f"- Чекпоинт модели:\n_{HydraConfig.get().runtime.cwd}/{checkpoint_path}/_\n"
        )
        f.write(
            f"- Хеш проекта _{model_info['git']['project_name']}_:\n*{model_info['git']['commit_hash']}*\n"
        )
        f.write("- Хеш проекта _ecglib_:\n*{}*\n".format(COMMIT_HASH))
        f.write(
            "- Код запуска:\n{{code:language=python}}{}{{code}}\n".format(
                model_info["run_command"]
            )
        )
        f.write("=============METRICS REPORT=============\n")
        f.write(
            "\n".join(
                [
                    line
                    for i, line in enumerate(
                        model_info["metrics"]["metrics"]
                        .to_markdown(index=True)
                        .split("\n")
                    )
                    if i != 1
                ]
            )
        )

    return model_info
