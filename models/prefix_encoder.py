import random
from collections import defaultdict
from typing import Optional

import torch
from torch import nn

from tclm.data.dataloader import Seq2SeqBatch
from tclm.utils.prompts import ENCODED_PROMPTS_TYPE
from tclm.utils.task_relation import COMPOSITE_TASK_TO_COMPONENTS

ATTENTION_TYPES = {"cross", "enc", "dec"}
ATTENTION_TYPES_TO_PREDIFINED_KEYS = {"dec": "decoder_prompt", "cross": "cross_attention_prompt", "enc": "encoder_prompt"}


class PrefixEncoder(torch.nn.Module):
    def __init__(self, args, lm_config):
        super().__init__()
        self.args = args
        self.lm_config = lm_config
        self.preseqlen = self.args.preseqlen
        self.mid_dim = self.args.mid_dim
        self.match_n_layer = self.lm_config.num_layers
        self.match_n_head = self.lm_config.num_heads
        self.n_embd = self.lm_config.d_model
        self.match_n_embd = self.lm_config.d_kv
        self.dropout = nn.Dropout(self.args.pfx_dropout_rate)
        self.activated_atomic_tasks = self.args.activated_atomic_tasks
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        assert self.args.mode in ["pfx_enc_tune", "pfx_tune"], \
            "Prefix encoder works for two modes: `pfx_enc_tune` and `pfx_tune`."

        if self.args.mode == "pfx_tune":
            self.wte = nn.ModuleDict({
                key: nn.ModuleDict({
                    task_type: nn.Embedding(self.preseqlen, self.n_embd)
                    for task_type in self.activated_atomic_tasks
                })
                for key in ATTENTION_TYPES
            })

        self.control_trans = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )
            for key in ATTENTION_TYPES
        })

    def add_new_task(self, new_task: str):
        if new_task not in self.activated_atomic_tasks and self.args.mode == "pfx_tune":
            for attention_type in ATTENTION_TYPES:
                self.wte[attention_type][new_task] = nn.Embedding(self.preseqlen, self.n_embd)

    def get_prefix(
        self,
        batch: Seq2SeqBatch,
        num_beams: int = 1,
        encoded_prompts: Optional[ENCODED_PROMPTS_TYPE] = None
    ):
        task_type = batch["task_type"]
        bsz = len(task_type)

        reduced_tasks = list(set(task_type))
        to_reduced_idx = {_task: reduced_idx for reduced_idx, _task in enumerate(reduced_tasks)}

        ret = {}
        for attention_type in ATTENTION_TYPES:

            temp_controls = []
            for _task in reduced_tasks:
                if encoded_prompts is None:
                    temp_control = self.wte[attention_type][_task](self.input_tokens)
                else:
                    temp_control = encoded_prompts[attention_type][_task]["prompt_embedding"]
                temp_controls.append(temp_control)

            encoded = self.control_trans[attention_type](torch.stack(temp_controls, dim=0))
            past_key_values = torch.stack([encoded[to_reduced_idx[_task]] for _task in task_type])  # bsz * seqlen * _

            seqlen = past_key_values.shape[1]
            if attention_type != "enc":
                past_key_values = past_key_values.expand(num_beams, bsz, seqlen, -1).transpose(0, 1).reshape(bsz * num_beams, seqlen, -1)

            # dropout and reshape
            past_key_values = past_key_values.view(
                bsz * num_beams if attention_type != "enc" else bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd
            )
            past_key_values = self.dropout(past_key_values).permute([2, 0, 3, 1, 4]).split(2)

            current_device = past_key_values[0].device
            if encoded_prompts is None:
                prev_key_padding_mask = torch.zeros(bsz, seqlen).bool()
            else:
                prev_key_padding_mask = torch.stack([encoded_prompts[attention_type][_task]["padding_mask"] for _task in task_type])

            if attention_type != "enc":
                prev_key_padding_mask = prev_key_padding_mask.expand(num_beams, bsz, -1).transpose(0, 1).reshape(bsz * num_beams, -1)
            prev_key_padding_mask = prev_key_padding_mask.to(current_device)

            ret[attention_type] = [
                {
                    "prev_key": kv[0].contiguous(),
                    "prev_value": kv[1].contiguous(),
                    "prev_key_padding_mask": prev_key_padding_mask,
                } for kv in past_key_values
            ]

        ret = [
            {
                matched_key: ret[attention_type][i]
                for attention_type, matched_key in ATTENTION_TYPES_TO_PREDIFINED_KEYS.items()
            } for i in range(len(ret["cross"]))
        ]

        return ret


# Skeleton class needed (function)
class MergedPrefixEncoder(torch.nn.Module):
    def __init__(self, base_encoder: PrefixEncoder, composition_mode: str, num_compositions: int = 2):
        super().__init__()
        self.mode = base_encoder.args.mode
        self.base_encoder = base_encoder
        self.composition_mode = composition_mode
        self.num_compositions = num_compositions

        # TODO: This code for freezing parameters need not be here
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        projection_dim = base_encoder.lm_config.d_kv
        print(f"Training for {num_compositions} task compositions")

        if self.composition_mode.startswith("cross_projection"):
            self.projection = nn.Linear(projection_dim * num_compositions, projection_dim)

        elif self.composition_mode.startswith("bi_projection"):
            self.projection = nn.Linear(projection_dim, projection_dim)

        elif self.composition_mode.startswith("layerwise_bi_projection"):
            self.projection = nn.ModuleList([
                nn.ModuleDict({
                    key: nn.Linear(projection_dim, projection_dim)
                    for key in ATTENTION_TYPES_TO_PREDIFINED_KEYS.values()
                }) for _ in range(base_encoder.lm_config.num_layers)
            ])

        elif "attention" in self.composition_mode:
            n_embd = base_encoder.n_embd
            n_head = base_encoder.lm_config.num_heads
            kv_dim = base_encoder.lm_config.d_kv
            preseqlen = base_encoder.preseqlen
            if (n_embd % n_head) != 0:
                n_head = n_embd // kv_dim
                print(f"Can't divide n_embd to n_head. Set number of head as {n_head}")
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)

            if self.composition_mode in ("concat_attention", "concat_attention_projection", "concat_aggregated_attention"):
                self.aggregated_attn = nn.MultiheadAttention(n_embd * 2, n_head, batch_first=True)
                self.projection = nn.Linear(num_compositions * preseqlen, preseqlen)

    def get_prefix(
        self,
        batch: Seq2SeqBatch,
        num_beams: int = 1,
        encoded_prompts: Optional[ENCODED_PROMPTS_TYPE] = None
    ):
        if self.composition_mode == "embedding_attention":
            encoded_prompts = {}
            task_type = batch["task_type"]  # ["TFU+PTA", "PPR+ATP", "PPR+ATP"],
            reduced_tasks = list(set(task_type))  # {"TFU+PTA", "PPR+ATP"}

            for attention_type in ATTENTION_TYPES:
                concatenated_vectors = []
                for composition_task in reduced_tasks:
                    decomposed_tasks = COMPOSITE_TASK_TO_COMPONENTS[composition_task]  # ("TFU", "PTA")
                    concatenated = torch.cat(
                        [
                            self.base_encoder.wte[attention_type][atomic_task].weight  # preseqlen * n_embd
                            for atomic_task in decomposed_tasks
                        ],
                        dim=0,
                    ).detach()  # (num_compositions * preseqlen) * n_embd
                    concatenated_vectors.append(concatenated)
                concatenated_vectors = torch.stack(concatenated_vectors, dim=0)
                embedding = self.attn(concatenated_vectors, concatenated_vectors, concatenated_vectors)[0]
                padding_mask = torch.zeros(embedding.shape[1]).bool()

                encoded_prompts[attention_type] = {
                    composition_task: {"prompt_embedding": x, "padding_mask": padding_mask}
                    for x, composition_task in zip(embedding, reduced_tasks)
                }
            composed_prefix = self.base_encoder.get_prefix(batch=batch, num_beams=num_beams, encoded_prompts=encoded_prompts)
        else:
            prefixes = []
            for _encoder_task_type in zip(*[COMPOSITE_TASK_TO_COMPONENTS[_task_type] for _task_type in batch['task_type']]):
                # >>> batch['task_type']
                # ["TFU+PTA", "PPR+ATP", "PPR+ATP"],
                # >>> _encoder_task_type
                # ("TFU", "PPR", "PPR") or ("PTA", "ATP", "ATP")

                # TODO:To generate prefixes for each encoder,
                # This part first exchanges batch's task to each encoder task then generate prefix for its batch.
                # After that, it restores original batch task.
                # This part is too hard to understand without background knowledge. we need to improve code readability.
                # For example, we can give encoded_prompt as list(prompt_embeddings) instead of dictionary {prompt: embedding}
                _batch = {**batch, **{'task_type': list(_encoder_task_type)}}

                with torch.no_grad():
                    self.base_encoder.eval()
                    _encoded = self.base_encoder.get_prefix(
                        _batch,
                        num_beams,
                        encoded_prompts
                    )
                    prefixes.append(_encoded)

            composed_prefix = self.get_composed_prefix(prefixes)

        return composed_prefix

    #Prompt aggregation
    def get_composed_prefix(self, prefixes):
        def _add(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = torch.sum(torch.stack([pkv[k] for pkv in pkv_per_task]), dim=0)
            return composed_pkv

        def _avg(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = torch.mean(torch.stack([pkv[k] for pkv in pkv_per_task]), dim=0)
            return composed_pkv

        def _concat(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = torch.cat([pkv[k] for pkv in pkv_per_task], dim=-1)
                else:
                    composed_pkv[k] = torch.cat([pkv[k] for pkv in pkv_per_task], dim=-2)
            return composed_pkv

        def _attn(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = torch.cat([pkv[k] for pkv in pkv_per_task], dim=-1)
                else:
                    concatenation = torch.cat([pkv[k] for pkv in pkv_per_task], dim=-2)
                    bsz, n_heads, seqlen, kv_dim = concatenation.shape
                    """
                    concatenation = concatenation.transpose(1, 2).contiguous().view(
                        bsz, seqlen, n_heads * kv_dim
                    )
                    """
                    concatenation = concatenation.permute(0, 2, 3, 1).contiguous().view(
                        bsz, seqlen, kv_dim * n_heads
                    )
                    attended_concatenation = self.attn(concatenation, concatenation, concatenation)[0]
                    """
                    composed_pkv[k] = attended_concatenation.view(
                        bsz, seqlen, n_heads, kv_dim
                    ).transpose(1, 2)
                    """
                    composed_pkv[k] = attended_concatenation.view(
                        bsz, seqlen, kv_dim, n_heads
                    ).permute(0, 3, 1, 2)

            return composed_pkv

        ### TMP! For developing ###
        def _attn_aggregated(pkv_per_task):
            composed_pkv = {}
            composed_pkv["prev_key_padding_mask"] = torch.cat([pkv["prev_key_padding_mask"] for pkv in pkv_per_task], dim=-1)
              
            aggregated_key_values = [torch.cat([pkv['prev_key'], pkv['prev_value']], dim=-1) for pkv in pkv_per_task]
            concatenation = torch.cat(aggregated_key_values, dim=-2)
            bsz, n_heads, seqlen, agg_kv_dim = concatenation.shape
                    
            concatenation = concatenation.permute(0, 2, 3, 1).contiguous().view(
                bsz, seqlen, agg_kv_dim * n_heads
            )
            attended_concatenation = self.aggregated_attn(concatenation, concatenation, concatenation)[0]
                    
            attended_concatenation = attended_concatenation.view(
                bsz, seqlen, agg_kv_dim, n_heads
            ).permute(0, 3, 1, 2)
           
            composed_pkv['prev_key'], composed_pkv['prev_value'] = attended_concatenation.chunk(2, dim=-1)
             
            return composed_pkv

        def _attn_project(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    concatenation = torch.cat([pkv[k] for pkv in pkv_per_task], dim=-2)
                    bsz, n_heads, seqlen, kv_dim = concatenation.shape
                    concatenation = concatenation.permute(0, 2, 3, 1).contiguous().view(
                        bsz, seqlen, kv_dim * n_heads
                    )
                    attended_concatenation = self.attn(concatenation, concatenation, concatenation)[0]

                    attended_concatenation = attended_concatenation.view(
                        bsz, seqlen, kv_dim, n_heads
                    ).permute(0, 3, 1, 2)

                    composed_pkv[k] = self.projection(attended_concatenation.transpose(-1, -2)).transpose(-1, -2)

            return composed_pkv

        def _cross_projection(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = self.projection(torch.cat([pkv[k] for pkv in pkv_per_task], dim=-1))
            return composed_pkv

        def _cross_projection_random(pkv_per_task):
            composed_pkv = {}
            permutation = list(range(self.num_compositions))
            random.shuffle(permutation)

            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = self.projection(
                        torch.cat([pkv_per_task[permutation[i]][k] for i in range(self.num_compositions)], dim=-1)
                    )
            return composed_pkv

        def _bi_projection_mean(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = torch.mean(
                        self.projection(torch.stack([pkv[k] for pkv in pkv_per_task], dim=0)), dim=0
                    )
            return composed_pkv

        def _layerwise_bi_projection_mean(pkv_per_task, layer_idx, model_type):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = torch.mean(
                        self.projection[layer_idx][model_type](torch.stack([pkv[k] for pkv in pkv_per_task], dim=0)), dim=0
                    )
            return composed_pkv

        def _bi_projection_concat(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = torch.cat([_pkv[k] for _pkv in pkv_per_task], dim=-1)
                else:
                    projected = self.projection(
                        torch.stack([pkv[k] for pkv in pkv_per_task], dim=0)
                    )
                    composed_pkv[k] = torch.cat(list(projected), dim=-2)
            return composed_pkv

        def _layerwise_bi_project_concat(pkv_per_task, layer_idx, model_type):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = torch.cat([_pkv[k] for _pkv in pkv_per_task], dim=-1)
                else:
                    projected = self.projection[layer_idx][model_type](
                        torch.stack([pkv[k] for pkv in pkv_per_task], dim=0)
                    )
                    composed_pkv[k] = torch.cat(list(projected), dim=-2)
            return composed_pkv

        def _multiply(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = pkv_per_task[0][k]
                    for idx in range(1, len(pkv_per_task)):
                        composed_pkv[k] = torch.mul(composed_pkv[k], pkv_per_task[idx][k])
            return composed_pkv

        def _maxpool(pkv_per_task):
            composed_pkv = {}
            for k in pkv_per_task[0].keys():
                if k == "prev_key_padding_mask":
                    composed_pkv[k] = pkv_per_task[0][k]
                else:
                    composed_pkv[k] = torch.max(torch.stack([pkv[k] for pkv in pkv_per_task]), dim=0)
            return composed_pkv

        def _layermix_past_key_values(prefixes):
            composed_prefix = []
            assert len(prefixes[0]) % len(prefixes) == 0
            component_layer_cnt = len(prefixes[0]) // len(prefixes)
            for idx, prefix in enumerate(prefixes):
                composed_prefix += prefix[component_layer_cnt * idx:component_layer_cnt * (idx + 1)]
            return composed_prefix

        composed_prefix = []

        if self.composition_mode == "layermix":
            composed_prefix = _layermix_past_key_values(prefixes)
            return composed_prefix

        for layer_idx in range(len(prefixes[0])):
            temp_dict = {}
            for model_type in prefixes[0][layer_idx].keys():
                pkv_per_task = [_prefix[layer_idx][model_type] for _prefix in prefixes]

                if self.composition_mode == "addition":
                    temp_dict[model_type] = _add(pkv_per_task)
                elif self.composition_mode == "average":
                    temp_dict[model_type] = _avg(pkv_per_task)
                elif self.composition_mode == "multiply":
                    temp_dict[model_type] = _multiply(pkv_per_task)
                elif self.composition_mode == "concatenation":
                    temp_dict[model_type] = _concat(pkv_per_task)
                elif self.composition_mode == "maxpooling":
                    temp_dict[model_type] = _maxpool(pkv_per_task)
                elif self.composition_mode == "cross_projection":
                    temp_dict[model_type] = _cross_projection(pkv_per_task)
                elif self.composition_mode == "cross_projection_random":
                    temp_dict[model_type] = _cross_projection_random(pkv_per_task)
                elif self.composition_mode == "bi_projection_mean":
                    temp_dict[model_type] = _bi_projection_mean(pkv_per_task)
                elif self.composition_mode == "layerwise_bi_projection_mean":
                    temp_dict[model_type] = _layerwise_bi_projection_mean(pkv_per_task, layer_idx, model_type)
                elif self.composition_mode == "concat_attention":
                    temp_dict[model_type] = _attn(pkv_per_task)
                elif self.composition_mode == "concat_aggregated_attention":
                    temp_dict[model_type] = _attn_aggregated(pkv_per_task)
                elif self.composition_mode == "concat_attention_projection":
                    temp_dict[model_type] = _attn_project(pkv_per_task)
                elif self.composition_mode == "layerwise_bi_projection_concat":
                    temp_dict[model_type] = _layerwise_bi_project_concat(pkv_per_task, layer_idx, model_type)
                elif self.composition_mode == "bi_projection_concat":
                    temp_dict[model_type] = _bi_projection_concat(pkv_per_task)
                else:
                    raise NotImplementedError("Please select between ['addition', 'average', 'concatenation', 'layermix', 'multiply', 'maxpooling', 'projection', 'concat_attention']")
            composed_prefix.append(temp_dict)
        return composed_prefix
