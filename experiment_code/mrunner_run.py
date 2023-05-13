import subprocess

import torch

from mrunner.helpers.client_helper import get_configuration


if __name__ == "__main__":
    cfg = get_configuration(print_diagnostics=True, with_neptune=False)

    del cfg["experiment_id"]

    # Start the moolib.broker module in a new process
    broker_process = subprocess.Popen(['python', '-m', 'moolib.broker'])

    eval_watcher = cfg.get("eval_watcher", None)
    if eval_watcher:

        # TODO: get project and group somehow / savedir -> right now hardcoded, don't want to touch hydra
        watcher_args = dict(
            checkpoint_dir = f"checkpoint/hackrl/nle/{cfg['group']}",
            checkpoint_step = cfg["eval_checkpoint_step"],
            max_step = cfg["eval_max_step"],
            rollouts = cfg["eval_rollouts"],
            batch_size = cfg["eval_batch_size"],
            device = cfg["eval_device"],
        )

        del cfg["eval_watcher"]
        del cfg["eval_rollouts"]
        del cfg["eval_batch_size"]
        del cfg["eval_checkpoint_step"]
        del cfg["eval_max_step"]
        del cfg["eval_device"]

        watcher_args = [item for key, value in watcher_args.items() for item in [f"--{key}", str(value)]]

        # Start the eval_watcher 
        watcher_cmd = ['python', '-m', 'hackrl.eval_watcher'] + watcher_args
        eval_process = subprocess.Popen(watcher_cmd)

    key_pairs = [f"{key}={value}" for key, value in cfg.items()]
    cmd = ['python', '-m', 'hackrl.experiment'] + key_pairs

    device_count = torch.cuda.device_count()
    if device_count > 1:
        if "device" not in cfg or cfg["device"] != "cpu":
            # Adding more peers to this experiment, starting more processes with the
            # same `project` and `group` settings, using a different setting for `device`           
            for i in range(1, device_count):
                subprocess.Popen(cmd + [f"device=cuda:{i}"])

    # default device is cuda:0
    subprocess.run(cmd) 

    # When you're done, terminate the broker process
    broker_process.terminate()

    if eval_watcher:
        eval_process.wait()