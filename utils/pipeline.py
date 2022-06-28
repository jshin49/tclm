from tclm.utils.load_ckpts import load_tclms


def setup_pipeline(args, tokenizer, language_model):
    assert len(args.load_pretrained) > 0, "give minimum one checkpoint to build a pipeline"
    loaded_models = load_tclms(args, tokenizer, language_model)
    print("Created Models")

    composition_tasks = args.composition_tasks
    print("Generating a pipeline: " + " -> ".join(composition_tasks))

    tasks_from_cli = [k for k, v in args.data_dir.items() if v is not None]

    assert len(tasks_from_cli) == 1, "You should offer a data directory for only one initial task!"

    return loaded_models, composition_tasks, tasks_from_cli[0]
