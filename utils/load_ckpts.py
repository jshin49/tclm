import os

from omegaconf import OmegaConf

from tclm.models.tclm import TCLM
from tclm.models.prefix_encoder import MergedPrefixEncoder

def parse_latest_dir(path):
    parent_dir, node = os.path.split(path)
    if node == "latest":
        latest_dir = sorted(os.listdir(parent_dir))[-1]
        print(latest_dir)
        return os.path.join(parent_dir, latest_dir)
    else:
        return path


def load_tclms(args, tokenizer, language_model):
    loaded_models = []
    for parent_dir in args.load_pretrained:
        parent_dir = parse_latest_dir(parent_dir)

        ckpts = [_ckpt for _ckpt in os.listdir(parent_dir) if ".ckpt" in _ckpt]
        ckpt_cfgs = [_ckpt for _ckpt in os.listdir(parent_dir) if "config.yml" in _ckpt]
        assert len(ckpts) == 1 and len(ckpt_cfgs) == 1

        file_path = os.path.join(parent_dir, ckpts[0])
        config_path = os.path.join(parent_dir, ckpt_cfgs[0])
        ckpt_args = OmegaConf.load(config_path)

        print("load pretrained model from: ", file_path)
        loaded_models.append(
            TCLM.load_from_checkpoint(
                file_path,
                args=ckpt_args,
                tokenizer=tokenizer,
                language_model=language_model,
                strict=False,  # allow to skip lm weights loading
            )
        )
        if args.do_prefix_mixing == True and args.composition_mode== 'concatenation':
            assert len(loaded_models) == 1
            loaded_models[0].prefix_encoder = MergedPrefixEncoder(
                                                    base_encoder=loaded_models[0].prefix_encoder,
                                                    composition_mode=args.composition_mode,
                                                    num_compositions=args.num_compositions
                                                )


    # Sanity check when loading multiple models or wants to make a composition
    if len(loaded_models) > 1:
        trained_tasks = []
        for _model in loaded_models:
            per_model_trained_tasks = [k for k, v in _model.args.data_dir.items() if v is not None]
            if _model.do_pseudo_label_gen:
                per_model_trained_tasks.append(_model.pipeline_task)
            trained_tasks.append(per_model_trained_tasks)

        assert len(loaded_models) == len(args.composition_tasks), "Each model should be allocated its task"
        for trained_task, composition_task in zip(trained_tasks, args.composition_tasks):
            assert composition_task in trained_task, "Composition task should have done at train time."

    return loaded_models

