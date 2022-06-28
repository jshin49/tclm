from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tclm.models.tclm import TCLM
from tclm.data.dataloader import Seq2SeqDataset
from tclm.models.modeling_auto import AutoModelForSeq2SeqLM
from tclm.data.dataloader import get_dataloader
from tclm.utils.args import get_args
from tclm.utils.load_ckpts import load_tclms
from tclm.utils.task_relation import COMPOSITE_TASK_TO_COMPONENTS
import shutil

def eval_pipeline(args: DictConfig):
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    language_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    if args.load_pretrained is not None:
        loaded_models = load_tclms(args, tokenizer, language_model)
        assert len(loaded_models) == 1
        model = loaded_models[0]
    else:
        model = TCLM(
            args=args,
            tokenizer=tokenizer,
            language_model=language_model,
        )

    tasks_from_cli = [k for k, v in args.data_dir.items() if v is not None]
    assert len(tasks_from_cli) == 1, "You should offer a data directory for only one initial task!"
    task_from_cli = tasks_from_cli[0]
    trained_tasks = COMPOSITE_TASK_TO_COMPONENTS[task_from_cli] 

    trainer = Trainer(
        accelerator="gpu" if args.num_gpus > 0 else None,
        devices=args.num_gpus if args.num_gpus > 0 else None,
        strategy="ddp" if args.num_gpus > 1 else None,
        deterministic=True,
        val_check_interval=args.val_check_interval,
        resume_from_checkpoint=args.resume_from_ckpt,
    )

    initial_model = model 

    # CODE HACK!
    args.data_dir[trained_tasks[0]] = args.data_dir[task_from_cli]
    args.data_dir[task_from_cli] = None
    initial_dataloader = get_dataloader(args, tokenizer)["test"]
    trainer.test(initial_model, initial_dataloader)
    
    shutil.copy('./preds.txt', f'./preds_pipeline_{trained_tasks[0]}.txt')
    shutil.copy('./target.txt', f'./target_pipeline_{trained_tasks[0]}.txt')

    for _task in trained_tasks[1:]:
        _dataset = Seq2SeqDataset(
            args,
            tokenizer,
            source_path="./preds.txt",
            target_path="./target.txt",
            task_type=_task,
        )
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=args.batch_size["test"],
            shuffle=False,
            collate_fn=_dataset.collate_fn,
            num_workers=args.num_workers,
        )
        trainer.test(model, _dataloader)

        shutil.copy('./preds.txt', f'./preds_pipeline_{_task}.txt')
        shutil.copy('./target.txt', f'./target_pipeline_{_task}.txt')


if __name__ == "__main__":
    args = get_args()
    eval_pipeline(args)
