import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer
from omegaconf.omegaconf import OmegaConf

from tclm.models.modeling_auto import AutoModelForSeq2SeqLM
from tclm.models.tclm import TCLM
from tclm.models.prefix_encoder import PrefixEncoder
from tclm.models.prefix_encoder import MergedPrefixEncoder
from tclm.data.dataloader import get_dataloader
from tclm.data.dataloader import DataloadersDict
from tclm.utils.args import get_args
from tclm.utils.load_ckpts import load_tclms


def train(args):
    seed_everything(args.seed)
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    language_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    dataloaders: DataloadersDict = get_dataloader(args, tokenizer)

    # logging
    log_path = os.path.join(
        "logs",
        args.mode,
        f"seed_{args.seed}",
        args.exp_name,  # write some identifiable description on this run
    )
    os.makedirs(log_path, exist_ok=True)

    if args.load_pretrained is not None:
        loaded_models = load_tclms(args, tokenizer, language_model)
        assert len(loaded_models) == 1
        model = loaded_models[0]
        if args.mode == "pfx_tune" and args.do_prefix_mixing:
            assert isinstance(model.prefix_encoder, PrefixEncoder)

            base_encoder = model.prefix_encoder
            model.prefix_encoder = MergedPrefixEncoder(
                base_encoder=base_encoder,
                composition_mode=args.composition_mode,
                num_compositions=args.num_compositions,
            )

            args.activated_atomic_tasks = model.args.activated_atomic_tasks
    else:
        model = TCLM(
            args=args,
            tokenizer=tokenizer,
            language_model=language_model,
        )
    print("Created Model")

    OmegaConf.save(config=args, f=os.path.join(log_path, "config.yml"))

    earlystopping_callback = EarlyStopping(
        monitor="val_loss" if args.eval_loss_only else "val_em_score",
        patience=args.patience,
        verbose=False,
        mode="min" if args.eval_loss_only else "max",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch:02d}-" + ("{val_loss:.3f}" if args.eval_loss_only else "{val_em_score:.3f}"),
        save_top_k=1,
        monitor="val_loss" if args.eval_loss_only else "val_em_score",
        mode="min" if args.eval_loss_only else "max",
    )
    callbacks = [earlystopping_callback, checkpoint_callback]

    trainer = Trainer(
        accumulate_grad_batches=args.grad_acc_steps,
        gradient_clip_val=args.max_norm,
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        callbacks=callbacks,
        accelerator="gpu" if args.num_gpus > 0 else None,
        devices=args.num_gpus if args.num_gpus > 0 else None,
        strategy="ddp" if args.num_gpus > 1 else None,
        deterministic=True,
        val_check_interval=args.val_check_interval,
        logger=CSVLogger(log_path),
        resume_from_checkpoint=args.resume_from_ckpt,
        # limit_val_batches=0.05,
    )

    trainer.fit(model, dataloaders["train"], dataloaders["val"])

    print("test start...")

    best_model = TCLM.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        args=args,
        tokenizer=tokenizer,
        language_model=language_model,
        strict=False  # allow to skip lm weights loading
    )
    trainer.test(best_model, dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    train(args)
