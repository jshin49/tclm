from omegaconf import DictConfig
from omegaconf import OmegaConf


DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_OUTPUT_LENGTH = 200


def get_args() -> DictConfig:
    cli_args = OmegaConf.from_cli()
    args = OmegaConf.load(cli_args.base_config)
    for key in cli_args:
        assert key == "base_config" or key in args, \
            f"A command line argument '{key}' is not expected. \n Pre-defined keys: { ', '.join(args.keys())}."
    args = OmegaConf.merge(args, cli_args)

    # record it for later use
    if args.mode == "pfx_tune" and not args.do_prefix_mixing:
        # Note that when we trian or eval our model using mixing, tasks written in data_dir are not atomic tasks.
        # We record `activated_atomic_tasks` only if the training mode is simple prefix tuning.
        # For prefix mixing purpose, activated_atomic_tasks is still required to generate a base PrefixEncoder.
        # For that case, we should lookup the activated_atomic_tasks from the past configuration file \
        # that corresponds to the previous checkpoint.
        activated_atomic_tasks = set(k for k, v in args.data_dir.items() if v is not None)
        if args.pipeline_task:
            activated_atomic_tasks = activated_atomic_tasks | {args.pipeline_task}

        args.activated_atomic_tasks = list(activated_atomic_tasks)

    return args
