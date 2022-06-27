from typing import Dict
from typing import List
from typing import TypedDict

import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from tclm.utils.args import DEFAULT_MAX_INPUT_LENGTH
from tclm.utils.args import DEFAULT_MAX_OUTPUT_LENGTH
from tclm.utils.prompts import PROMPTS


class Seq2SeqSample(TypedDict):
    input_text: str
    output_text: str
    task_type: str


class DataloadersDict(TypedDict):
    train: DataLoader
    val: DataLoader
    test: DataLoader


class Seq2SeqBatch(TypedDict):
    input_text: List[str]
    output_text: List[str]
    task_type: List[str] 
    encoder_input: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        source_path: str,
        target_path: str,
        task_type: str,
    ):
        super(Seq2SeqDataset, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        input_texts = self.parse_file(source_path)
        output_texts = self.parse_file(target_path)
        assert len(input_texts) == len(output_texts)

        if "t5" in self.args.model_checkpoint and self.args.mode == "finetune":
            self.prompt = PROMPTS[task_type]
            input_texts = [f"{self.prompt} {x}" for x in input_texts]

        self.sample_list = [
            Seq2SeqSample(input_text=_x, output_text=_y, task_type=task_type)
            for _x, _y in zip(input_texts, output_texts)
        ]

    def __getitem__(self, i: int) -> Seq2SeqSample:
        return self.sample_list[i]

    def __len__(self):
        return len(self.sample_list)

    def parse_file(self, path) -> List[str]:
        with open(path, "r") as f:
            res = f.readlines()

        assert all([len(x) > 0 for x in res]), f"{path}: Some lines in the file is missing!"
        return res

    def collate_fn(self, batch: List[Seq2SeqSample]) -> Seq2SeqBatch:
        batch: Dict[str, List[str]] = {
            k: [x[k] for x in batch]
            for k in Seq2SeqSample.__annotations__
        }
        if "max_length" in self.args:
            max_input_length = max(self.args.max_length[_task]["in"] for _task in set(batch["task_type"]))
            max_output_length = max(self.args.max_length[_task]["out"] for _task in set(batch["task_type"]))
        else:
            max_input_length = DEFAULT_MAX_INPUT_LENGTH
            max_output_length = DEFAULT_MAX_OUTPUT_LENGTH

        enc_tokenizer_out = self.tokenizer(
            batch["input_text"],
            padding=True,
            return_tensors="pt",
            verbose=False,
            truncation=True,
            max_length=max_input_length,
        )
        encoder_input, enc_attention_mask = enc_tokenizer_out.input_ids, enc_tokenizer_out.attention_mask

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch["output_text"],
                padding=True,
                return_tensors="pt",  # non-padded return List[List[Int]]
                truncation=True,
                max_length=max_output_length,
            ).input_ids

        res = Seq2SeqBatch(
            input_text=batch["input_text"],
            output_text=batch["output_text"],
            task_type=batch["task_type"],
            encoder_input=encoder_input,
            attention_mask=enc_attention_mask,
            labels=labels,
        )

        return res


def get_dataloader(args, tokenizer) -> DataloadersDict:
    datasets = {k: [] for k in DataloadersDict.__annotations__}

    for task_type, task_dir in args.data_dir.items():
        if task_dir is None:
            continue
        for k in DataloadersDict.__annotations__:
            datasets[k].append(
                Seq2SeqDataset(
                    args,
                    tokenizer,
                    source_path=f"{task_dir}/{k}.source",
                    target_path=f"{task_dir}/{k}.target",
                    task_type=task_type,
                )
            )

    # val_reduction_size = -1
    # val_length = [len(dset) for dset in datasets['val']]
    # if val_reduction_size != -1:
    #     val_indices = [random.choices(range(vl), k=int(vl * val_reduction_size)) for vl in val_length]
    # else:
    #     val_indices = [range(vl) for vl in val_length]

    res: DataloadersDict = {
        k: DataLoader(
            # dataset=ConcatDataset(dset) if k != 'val' else ConcatDataset([Subset(d, idx) for d, idx in zip(dset, val_indices)]),
            dataset=ConcatDataset(dset),
            batch_size=args.batch_size[k],
            shuffle=(k == "train"),
            collate_fn=dset[0].collate_fn,
            num_workers=args.num_workers,
        )
        for k, dset in datasets.items()
    }

    return res
