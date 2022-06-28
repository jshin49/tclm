from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoTokenizer

from tclm.models.tclm import TCLM
from tclm.models.modeling_auto import AutoModelForSeq2SeqLM
from tclm.data.dataloader import DataloadersDict
from tclm.data.dataloader import get_dataloader
from tclm.utils.args import get_args
from tclm.utils.load_ckpts import load_tclms
import shutil

def eval_tclm(args: DictConfig):
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    language_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    dataloaders: DataloadersDict = get_dataloader(args, tokenizer)

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

    print("Created Models")

    trainer = Trainer(
        accelerator="gpu" if args.num_gpus > 0 else None,
        devices=args.num_gpus if args.num_gpus > 0 else None,
        strategy="ddp" if args.num_gpus > 1 else None,
        deterministic=True,
        val_check_interval=args.val_check_interval,
        resume_from_checkpoint=args.resume_from_ckpt,
    )

    print("test start...")
    trainer.test(model, dataloaders["test"])
    
    tasks_from_cli = [k for k, v in args.data_dir.items() if v is not None]
    assert len(tasks_from_cli) == 1, "You should offer a data directory for only one initial task!"
    task_from_cli = tasks_from_cli[0]

    shutil.copy('./preds.txt', f'./preds_zeroshot_{task_from_cli}.txt')
    shutil.copy('./target.txt', f'./target_zeroshot_{task_from_cli}.txt')


if __name__ == "__main__":
    args = get_args()
    eval_tclm(args)
