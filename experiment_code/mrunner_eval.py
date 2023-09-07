from mrunner.helpers.client_helper import get_configuration

from hackrl.eval import main as main_eval
from hackrl.eval_save import main as main_eval_save
from hackrl.eval_array import main as main_eval_array
from hackrl.rollout import main as main_rollout

MAIN_DICT = {"eval": main_eval, "eval_save": main_eval_save, "rollout": main_rollout, "eval_array": main_eval_array}


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=False)

    del config["experiment_id"]
    run_kind = config.pop("run_kind")

    main = MAIN_DICT[run_kind]
    main(variant=vars(config))
