import os
import shutil

from data.dataloader import DataloadersDict
from data.dataloader import Seq2SeqDataset
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tclm.models.modeling_auto import AutoModelForSeq2SeqLM
from tclm.utils.args import get_args
import tclm.utils.pipeline as pipeline_util


def gen_pipeline_pseudo_data(args: DictConfig):
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    language_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    loaded_models, composition_tasks, task_from_cli = pipeline_util.setup_pipeline(args, tokenizer, language_model)

    assert task_from_cli == composition_tasks[0], \
        "The task the first model learned must be the task given by the commandline."

    save_dir = f"pseudo_labeled_data/{'+'.join(composition_tasks)}"
    os.makedirs(save_dir, exist_ok=True)

    # initial_model = loaded_models[0]
    # TODO: Think about this carefully. We use true label y of the first task rather than using \hat{y}.
    initial_task_dir = args.data_dir[task_from_cli]

    trainer = Trainer(
        accelerator="gpu" if args.num_gpus > 0 else None,
        devices=args.num_gpus if args.num_gpus > 0 else None,
        strategy="ddp" if args.num_gpus > 1 else None,
        deterministic=True,
        val_check_interval=args.val_check_interval,
        resume_from_checkpoint=args.resume_from_ckpt,
    )

    for k in DataloadersDict.__annotations__:
        # k stands for "train", "val", or "test".
        # Input datafiles ('train.source', 'val.source', or 'test.source') are copied from the original path.
        shutil.copy(os.path.join(initial_task_dir, f"{k}.source"), os.path.join(save_dir, f"{k}.source"))
        # code hack. We pretend that a true label y is an inference output \hat{y}.
        shutil.copy(os.path.join(initial_task_dir, f"{k}.target"), "preds.txt")

        for _model, _task in zip(loaded_models[1:], composition_tasks[1:]):
            _dataset = Seq2SeqDataset(
                args,
                tokenizer,
                source_path="preds.txt",
                target_path="preds.txt",  # It is never used because we don't want to eval.
                task_type=_task,
            )
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=args.batch_size["test"],
                shuffle=False,
                collate_fn=_dataset.collate_fn,
                num_workers=args.num_workers,
            )
            trainer.test(_model, _dataloader)  # During test, `preds.txt` is overwritten with current outputs.

        # 'preds.txt' is the final output of the pipeline.
        # It is saved as 'train.target', 'val.target', or 'test.target'.
        os.rename("preds.txt", os.path.join(save_dir, f"{k}.target"))

    # We discard 'target.txt'
    os.remove("target.txt")


if __name__ == "__main__":
    args = get_args()
    gen_pipeline_pseudo_data(args)
