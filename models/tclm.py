import itertools
from collections import defaultdict
from typing import Union
from typing import Dict
from typing import Iterable
from typing import List

import torch
import pytorch_lightning as pl
import numpy as np
import rouge
from omegaconf import DictConfig
from transformers import AdamW
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu

from tclm.models.prefix_encoder import PrefixEncoder
from tclm.models.prefix_encoder import MergedPrefixEncoder
from tclm.models.prefix_encoder import ATTENTION_TYPES
from tclm.data.dataloader import Seq2SeqBatch
from tclm.utils.prompts import PROMPTS
from tclm.utils.prompts import PROMPTS_TYPE
from tclm.utils.prompts import ENCODED_PROMPTS_TYPE
from tclm.utils.args import DEFAULT_MAX_INPUT_LENGTH
from tclm.utils.args import DEFAULT_MAX_OUTPUT_LENGTH


class TCLM(pl.LightningModule):
    def __init__(self, args: DictConfig, tokenizer, language_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.lr = args.lr
        self.fuzzy_threshold = 95
        self.rouge_evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l'],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type='words',
            apply_avg=False,
            apply_best=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True
        )
        self.prompts = PROMPTS
        self.do_pseudo_label_gen = args.do_pseudo_label_gen
        self.pipeline_task = args.pipeline_task

        if self.do_pseudo_label_gen:
            self.pseudo_label_gen_sanity_check()

        if args.mode == "finetune":
            self.prefix_encoder = None
            self.encoded_prompts = None
        else:
            prefix_encoder = PrefixEncoder(args, language_model.config)
            if args.do_prefix_mixing:
                self.prefix_encoder = MergedPrefixEncoder(
                    base_encoder=prefix_encoder,
                    composition_mode=args.composition_mode,
                    num_compositions=args.num_compositions,
                )
                for param in self.prefix_encoder.base_encoder.parameters():
                    param.requires_grad = False
            else:
                self.prefix_encoder = prefix_encoder

            for param in self.language_model.parameters():
                param.requires_grad = False

            if args.mode == "pfx_enc_tune":
                self.encoded_prompts = self.get_encoded_prompts(self.prompts)
            elif args.mode == "pfx_tune":
                self.encoded_prompts = None
            else:
                raise ValueError("Invalid mode. It should be one of followings: `finetune`, `pfx_enc_tune`, `pfx_tune`.")

    def pseudo_label_gen_sanity_check(self):
        assert self.pipeline_task is not None
        assert self.pipeline_task in self.prompts

        assert self.args.composition_tasks is not None
        assert len(self.args.composition_tasks) > 1

        edge_tasks = [k for k, v in self.args.data_dir.items() if v is not None]
        assert set(edge_tasks) == set(self.args.composition_tasks)

    def state_dict(self, *args):
        # we overload state_dict function from nn.Module not to save & load lm weights
        _state_dict = super(TCLM, self).state_dict(*args)
        if self.args.mode in ["pfx_enc_tune", "pfx_tune"]:
            assert "language_model" in self._modules
            _state_dict = {k: x for k, x in _state_dict.items() if "language_model" not in k}
        return _state_dict

    def get_encoded_prompts(self, prompts: PROMPTS_TYPE) -> ENCODED_PROMPTS_TYPE:
        self.language_model.to("cpu")
        for task, prompt in prompts.items():
            if self.args.debug == True:
                print(f"The prompt for the task {task}: {prompt}")

        tokenizer_out = self.tokenizer(
            list(prompts.values()),
            padding='max_length',
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
            truncation=True,
            max_length=self.args.preseqlen,
        )
        encoder = self.language_model.encoder if "t5" in self.args.model_checkpoint else self.language_model.get_encoder()
        encoder.eval()
        encoded = encoder(
            input_ids=tokenizer_out["input_ids"],
            attention_mask=tokenizer_out["attention_mask"],
            return_dict=True
        )

        encoded = {
            task: {'prompt_embedding': x, 'padding_mask': y}
            for task, x, y in
            zip(prompts.keys(), encoded['last_hidden_state'].detach(), ~tokenizer_out['attention_mask'].bool())
        }
        encoded_prompts = {key: encoded for key in ATTENTION_TYPES}  # repetition for `enc`, `dec`, `cross`
        return encoded_prompts

    def move_encoded_prompts(self) -> None:
        if self.encoded_prompts is not None:
            self.encoded_prompts = {
                task: {k: x.to(self.device) for k, x in dictvalue.items()}
                for task, dictvalue in self.encoded_prompts.items()
            }

    def on_train_start(self) -> None:
        self.move_encoded_prompts()

    def on_validation_start(self) -> None:
        self.move_encoded_prompts()

    def on_test_start(self) -> None:
        self.move_encoded_prompts()

    def gen_pseudo_labels(self, input_text: List[str]) -> List[str]:
        temp = input_text
        # We skip the first edge task. We suppose input_text consists of gold labels of the first task's data.
        for _task in self.args.composition_tasks[1:]:
            if "max_length" in self.args:
                max_input_length = self.args.max_length[_task]["in"]
            else:
                max_input_length = DEFAULT_MAX_INPUT_LENGTH

            if "t5" in self.args.model_checkpoint and self.args.mode == "finetune":
                temp = [f"{self.prompts[_task]} {x}" for x in temp]

            _tokenizer_out = self.tokenizer(
                temp,
                padding=True,
                return_tensors="pt",
                verbose=False,
                truncation=True,
                max_length=max_input_length,
            )
            _pipeline_input_batch = Seq2SeqBatch(
                input_text=temp,
                output_text=temp,  # not used in prediction
                task_type=len(temp) * [_task],
                encoder_input=_tokenizer_out.input_ids.to(self.device),
                attention_mask=_tokenizer_out.attention_mask.to(self.device),
                labels=_tokenizer_out.input_ids,  # not used in prediction
            )
            pred_token: List[List[str]] = self.pred_step(_pipeline_input_batch, -1)["pred_token"]
            temp = self.tokenizer.batch_decode(pred_token, skip_special_tokens=True)

        return temp

    def add_pseudo_samples_to_batch(self, batch: Seq2SeqBatch) -> Seq2SeqBatch:
        first_task = self.args.composition_tasks[0]
        last_task = self.args.composition_tasks[-1]

        is_first_task = [x == first_task for x in batch["task_type"]]
        num_first_task_samples = sum(is_first_task)
        if num_first_task_samples == 0:
            return batch
        else:
            # Samples for the first task can be used to generate pseudo labels
            # Also, we skip the first edge task. We want to use gold labels directly.
            pipeline_input_text = [x for is_initial, x in zip(is_first_task, batch["output_text"]) if is_initial]

            with torch.no_grad():
                pseudo_labels = self.gen_pseudo_labels(pipeline_input_text)

            pseudo_input_text = [x for is_first, x in zip(is_first_task, batch["input_text"]) if is_first]
            pseudo_task_type = num_first_task_samples * [self.pipeline_task]
            pseudo_encoder_input = batch["encoder_input"][is_first_task]
            pseudo_attention_mask = batch["attention_mask"][is_first_task]

            if "max_length" in self.args:
                max_output_length = self.args.max_length[last_task]["out"]
            else:
                max_output_length = DEFAULT_MAX_OUTPUT_LENGTH

            with self.tokenizer.as_target_tokenizer():
                merged_labels = self.tokenizer(
                    batch["output_text"] + pseudo_labels,
                    padding=True,
                    return_tensors="pt",  # non-padded return List[List[Int]]
                    return_attention_mask=True,
                    truncation=True,
                    max_length=max_output_length,
                ).input_ids.to(self.device)

            ret = Seq2SeqBatch(
                input_text=batch["input_text"] + pseudo_input_text,
                output_text=batch["output_text"] + pseudo_labels,
                task_type=batch["task_type"] + pseudo_task_type,
                encoder_input=torch.cat((batch["encoder_input"], pseudo_encoder_input), dim=0),
                attention_mask=torch.cat((batch["attention_mask"], pseudo_attention_mask), dim=0),
                labels=merged_labels,
            )

            return ret

    def training_step(self, batch: Seq2SeqBatch, index):
        if self.do_pseudo_label_gen:
            batch = self.add_pseudo_samples_to_batch(batch)

        if self.args.mode in ["pfx_enc_tune", "pfx_tune"]:
            past_prompt = self.prefix_encoder.get_prefix(batch, encoded_prompts=self.encoded_prompts)
            self.language_model.eval()
        else:
            past_prompt = None
        outputs = self.language_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            past_prompt=past_prompt
        )

        return {'loss': outputs.loss, 'log': {'train_loss': outputs.loss.detach()}}

    def eval_step(self, batch: Seq2SeqBatch):
        if self.args.mode in ["pfx_enc_tune", "pfx_tune"]:
            past_prompt = self.prefix_encoder.get_prefix(batch, encoded_prompts=self.encoded_prompts)
            self.language_model.eval()
        else:
            past_prompt = None

        outputs = self.language_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            past_prompt=past_prompt
        )

        return outputs.loss.item()

    def eval_epoch_end(self, outputs):
        res = {"loss": np.mean(outputs)}
        print(res)
        return res

    def pred_step(self, batch: Seq2SeqBatch, index: int) -> Dict[str, List[Union[List[str], str]]]:
        if self.args.mode in ["pfx_enc_tune", "pfx_tune"]:
            past_prompt = self.prefix_encoder.get_prefix(
                batch,
                num_beams=self.args.num_beams,
                encoded_prompts=self.encoded_prompts if self.args.mode == "pfx_enc_tune" else None
            )
            self.language_model.eval()
        else:
            past_prompt = None

        if "max_length" in self.args:
            max_output_length = max(self.args.max_length[_task]["out"] for _task in set(batch["task_type"]))
        else:
            max_output_length = DEFAULT_MAX_OUTPUT_LENGTH

        pred_token = self.language_model.generate(
            batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            num_beams=self.args.num_beams,
            min_length=5,
            max_length=max_output_length,
            early_stopping=True,
            past_prompt=past_prompt,
        )

        if self.args.debug == True and index % 1 == 0:
            inpt_example = self.tokenizer.decode(
                    self.tokenizer(batch["input_text"][0]).input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            pred_example = self.tokenizer.decode(
                    pred_token[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            tagt_example = self.tokenizer.decode(
                    self.tokenizer(batch["output_text"][0]).input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print()
            print("[INPT]: " + inpt_example)
            print("[PRED]: " + pred_example)
            print("[TAGT]: " + tagt_example)

        return {
            "pred_token": pred_token,
            "label": batch["output_text"],
            "task_type": batch["task_type"],
        }

    def pred_epoch_end(self, outputs: List[Dict[str, Iterable]]):
        outputs: Dict[str, List] = {
            k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]
        }  # flattening the doubly iterable output across pred-steps.
        outputs['label'] = self.tokenizer.batch_decode(
                self.tokenizer(outputs['label']).input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        preds: List[str] = self.tokenizer.batch_decode(
            outputs["pred_token"], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        self.save_samples([x + '\n' for x in preds], "./preds.txt")
        self.save_samples([x + '\n' for x in outputs["label"]], "./target.txt")

        task_index = defaultdict(list)
        for i, _task in enumerate(outputs["task_type"]):
            task_index[_task].append(i)
        task_index["full"] = list(range(len(outputs["task_type"])))

        res = {}
        for _task, index_list in task_index.items():
            _labels, _preds = [outputs["label"][i] for i in index_list], [preds[i] for i in index_list]
            per_task_res = self.pred_metrics(_labels, _preds)

            if _task == "full":
                res.update(per_task_res)
            else:
                res.update({f"{_task}-{k}": v for k, v in per_task_res.items()})

        print(res)
        return res

    def pred_metrics(self, labels: List[str], preds: List[str]) -> Dict[str, float]:
        em_score = 1.0 * sum([o.strip().lower() == p.strip().lower() for (o, p) in zip(labels, preds)]) / len(preds)
        fuzzy_score = sum([fuzz.ratio(o, p) > self.fuzzy_threshold for o, p in zip(labels, preds)]) / len(preds)
        rouge_score = self.rouge_evaluator.get_scores(preds, labels)

        bleu_score = np.array([
            sentence_bleu([_label], _pred, weights=[(1.0,), (1./2, 1./2), (1./3, 1./3, 1./3), (1./4, 1./4, 1./4, 1./4)])
            for _pred, _label in zip(preds, labels)
        ]).mean(axis=0)

        return {
            'em_score': em_score,
            'fuzzy_score': fuzzy_score,
            'rouge-1': rouge_score["rouge-1"]["f"],
            'rouge-2': rouge_score["rouge-2"]["f"],
            'rouge-3': rouge_score["rouge-3"]["f"],
            'rouge-4': rouge_score["rouge-4"]["f"],
            'rouge-l': rouge_score["rouge-l"]["f"],
            'bleu-1': bleu_score[0],
            'bleu-2': bleu_score[1],
            'bleu-3': bleu_score[2],
            'bleu-4': bleu_score[3],
        }

    def validation_step(self, batch: Seq2SeqBatch, index):
        if self.do_pseudo_label_gen:
            batch = self.add_pseudo_samples_to_batch(batch)

        if self.args.eval_loss_only:
            return self.eval_step(batch)
        else:
            return self.pred_step(batch, index)

    def validation_epoch_end(self, outputs):
        if self.args.eval_loss_only:
            res = self.eval_epoch_end(outputs)
        else:
            res = self.pred_epoch_end(outputs)

        res = {f'val_{k}': v for k, v in res.items()}
        self.log_dict(res)

        return res

    def test_step(self, batch: Seq2SeqBatch, index):
        if self.do_pseudo_label_gen:
            batch = self.add_pseudo_samples_to_batch(batch)

        return self.pred_step(batch, index)

    def test_epoch_end(self, outputs):
        res = {f'test_{k}': v for k, v in self.pred_epoch_end(outputs).items()}
        self.log_dict(res)
        return res

    def save_samples(self, samples: List[str], filename):
        with open(filename, 'w') as f:
            f.writelines(samples)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)
