from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple

STYLEPTB_ATOMIC_TASKS = ["TFU", "TPR", "TPA", "PPR", "ATP", "PTA", "ARR", "PBF", "PFB"]
STYLEPTB_COMPOSITE_TASKS = ["TFU+PPR", "PPR+ATP", "PPR+PTA", "TFU+ATP", "TFU+PTA", "TPR+ATP", "TPR+PTA", "TPA+ATP", "TPA+PTA", "TPR+PPR", "TPA+PPR", "ARR+PFB", "ARR+PBF", "TFU+ARR", "TPA+ARR", "TPR+ARR", "TFU+PBF", "TFU+PFB", "TPA+PFB", "TPA+PBF", "TPR+PBF", "TPR+PFB"]
SHUFFLED_STYLEPTB_COMPOSITE_TASKS = ['PPR+PTA', 'TPR+PBF', 'TFU+PPR', 'PPR+ATP', 'ARR+PFB', 'TFU+PTA', 'TFU+PFB', 'TFU+ATP', 'TPR+ATP', 'TPA+PBF', 'TFU+PBF', 'ARR+PBF', 'TPR+PFB', 'TFU+ARR', 'TPR+PTA', 'TPA+ARR', 'TPA+PTA', 'TPA+PFB', 'TPA+ATP', 'TPA+PPR', 'TPR+ARR', 'TPR+PPR']
# XNLI_COMPOSITE_TASKS = ["ar_en+NLI", "bg_en+NLI", "de_en+NLI", "el_en+NLI", "es_en+NLI", "fr_en+NLI", "hi_en+NLI", "ru_en+NLI", "sw_en+NLI", "th_en+NLI", "tr_en+NLI", "ur_en+NLI", "vi_en+NLI", "zh_en+NLI"]
# XNLI_ATOMIC_TASKS = ["ar_en", "bg_en", "de_en", "el_en", "es_en", "fr_en", "hi_en", "ru_en", "sw_en", "th_en", "tr_en", "ur_en", "vi_en", "zh_en", "NLI"]
# XNLI_TRANSLATION_TASKS = list(set(XNLI_ATOMIC_TASKS) - {"NLI"})


class ScriptGenerator:
    def __init__(self, allocated_gpus: List, experiment: str, mode: str) -> None:
        print(allocated_gpus)
        assert all([n in range(0, 8) for n in allocated_gpus])
        self.allocated_gpus = list(set(allocated_gpus))
        self.experiment = experiment
        self.mode = mode

        self.num_allocation = len(self.allocated_gpus)
        self.base_config = "config/stylePTB_config.yml"
        self.seed = 0
        self.num_gpus = 1  # always single gpu

    def distribute_commands(self, commands: List[Tuple[str, str]]) -> Dict[int, List[str]]:
        per_gpu_commands = defaultdict(list)
        for i, train_and_eval_commands in enumerate(commands):
            gpu_idx = self.allocated_gpus[i % self.num_allocation]
            for _command in train_and_eval_commands:
                per_gpu_commands[gpu_idx].append(f"CUDA_VISIBLE_DEVICES={gpu_idx} {_command} \n")
        return per_gpu_commands

    def save_shell_script(self, per_gpu_commands: Dict[int, List[str]]) -> None:
        for gpu_idx, commands in per_gpu_commands.items():
            with open(f"{gpu_idx}.sh", "w") as f:
                f.write('#!/bin/bash\n')
                f.writelines(commands)

    def gen_scripts(self) -> None:
        if self.experiment == "full":
            commands = self.stylePTB_full_finetune()
        elif self.experiment == "hold1out":
            commands = self.stylePTB_hold1out_finetune()
        elif self.experiment == "holdkout":
            commands = self.stylePTB_holdkout_finetune()
        elif self.experiment == "unseen_one":
            commands = self.stylePTB_unseen_one_in_composition_finetune()
        elif self.experiment == "unseen_two":
            commands = self.stylePTB_unseen_two_in_composition_finetune()
        elif self.experiment == "prelim_pfx":
            commands = self.stylePTB_prelim_pfx_tune()
        elif self.experiment == "full_pfx":
            commands = self.stylePTB_full_pfx_tune()
        elif self.experiment == "hold1out_pfx":
            commands = self.stylePTB_hold1out_pfx_tune()
        elif self.experiment == "holdkout_pfx":
            commands = self.stylePTB_holdkout_pfx_tune()
        elif self.experiment == "unseen_one_pfx":
            commands = self.stylePTB_unseen_one_in_composition_pfx_tune()
        elif self.experiment == "unseen_two_pfx":
            commands = self.stylePTB_unseen_two_in_composition_pfx_tune()
        elif self.experiment == "single":
            commands = self.stylePTB_single_finetune()
        elif self.experiment == 'concatenation':
            commands = self.stylePTB_concatenation_pfx_tune()
        elif self.experiment == "minimal":
            commands = self.stylePTB_minimal_learning_finetune()
        elif self.experiment == "pipeline":
            commands = self.stylePTB_eval_pipeline_finetune()
        else:
            raise NotImplementedError

        self.save_shell_script(self.distribute_commands(commands))

    def task_name_to_datapath(self, task_name: str) -> str:
        path = f"AggregatedDataset/{task_name}"
        return path

    def stylePTB_full_finetune(self) -> List[Tuple[str, str]]:
        exp_name = f"finetune_full"
        data_dirs = " ".join([
            f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in STYLEPTB_COMPOSITE_TASKS + STYLEPTB_ATOMIC_TASKS
        ])

        log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"
        common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config} {data_dirs}"

        commands = [(
            f"python scripts/train.py exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}",
            f"python scripts/eval.py load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"
        )]

        return commands

    def stylePTB_single_finetune(self) -> List[Tuple[str, str]]:
        commands = []
        for _task in STYLEPTB_COMPOSITE_TASKS:
            exp_name = f"finetune_single_{_task}"
            data_dir = f'data_dir.{_task}={self.task_name_to_datapath(_task)}'
            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config} {data_dir}"

            commands.append((
                f"python scripts/train.py exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}",
                f"python scripts/eval.py load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"
            ))

        return commands
   
    def stylePTB_prelim_pfx_tune(self) -> List[Tuple[str, str]]:
        exp_name = f"pfx_tune_atomics"
        data_dirs = " ".join([
            f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in STYLEPTB_ATOMIC_TASKS
        ])
        log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"
        common = f"num_gpus={self.num_gpus} mode=pfx_tune seed={self.seed} base_config={self.base_config} {data_dirs}"
        commands = [(
            f"python scripts/train.py exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}",
            f"python scripts/eval.py load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"
        )]
        return commands

    def stylePTB_full_pfx_tune(self) -> List[Tuple[str, str]]:
        exp_name = f"pfx_tune_full"
        data_dirs = " ".join([
            f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in STYLEPTB_COMPOSITE_TASKS + STYLEPTB_ATOMIC_TASKS
        ])

        log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"
        common = f"num_gpus={self.num_gpus} mode=pfx_tune seed={self.seed} base_config={self.base_config} {data_dirs}"

        commands = [(
            f"python scripts/train.py exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}",
            f"python scripts/eval.py load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"
        )]

        return commands
 

    # Learn only Mixer?
    """
    def stylePTB_full_pfx_tune(self) -> List[Tuple[str, str]]:
        prelim_name = f"pfx_tune_atomics"
        exp_name = f"pfx_tune_full"
        data_dirs = " ".join([
            f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in STYLEPTB_COMPOSITE_TASKS
        ])

        prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"
        log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"
        common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=embedding_attention do_prefix_mixing=true num_compositions=2 seed={self.seed} base_config={self.base_config} {data_dirs}"

        commands = [(
            f"python scripts/train.py load_pretrained='[{prelim_directory}]' exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}",
            f"python scripts/eval.py load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"
        )]

        return commands
    """

    def stylePTB_hold1out_pfx_tune(self) -> List[Tuple[str, str]]:
        commands = []
        for out_task in STYLEPTB_COMPOSITE_TASKS:
            prelim_name = f"pfx_tune_atomics"
            exp_name = f"pfx_tune_except_{out_task}"
            common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=embedding_attention do_prefix_mixing=true num_compositions=2 seed={self.seed} base_config={self.base_config}"

            train_data_dirs = " ".join([
                f'data_dir.{x}={self.task_name_to_datapath(x)}'
                for x in STYLEPTB_COMPOSITE_TASKS if x != out_task
            ])
            eval_data_dir = f"data_dir.{out_task}={self.task_name_to_datapath(out_task)}"

            prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"
            log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} load_pretrained='[{prelim_directory}]' exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dir} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands


    def stylePTB_concatenation_pfx_tune(self) -> List[Tuple[str, str]]:
        commands = []
        for out_task in STYLEPTB_COMPOSITE_TASKS:
            prelim_name = f"pfx_tune_atomics"
            exp_name = f"pfx_tune_except_{out_task}"
            common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=concatenation do_prefix_mixing=true num_compositions=2 seed={self.seed} base_config={self.base_config}"

            eval_data_dir = f"data_dir.{out_task}={self.task_name_to_datapath(out_task)}"

            prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"

            eval_command = f"python scripts/eval.py {eval_data_dir} load_pretrained='[{prelim_directory}]' {common} &> concatenation_{out_task}_eval.log"

            commands.append((eval_command, ))
        return commands

    def stylePTB_holdkout_pfx_tune(self):
        commands = []
        for num_outs in range(2, len(STYLEPTB_COMPOSITE_TASKS), 2):
            out_tasks = SHUFFLED_STYLEPTB_COMPOSITE_TASKS[:num_outs]
            prelim_name = f"pfx_tune_atomics"
            exp_name = f"pfx_tune_except_{num_outs}"
            common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=embedding_attention do_prefix_mixing=true num_compositions=2 seed={self.seed} base_config={self.base_config}"

            train_data_dirs = " ".join([
                f'data_dir.{x}={self.task_name_to_datapath(x)}'
                for x in STYLEPTB_COMPOSITE_TASKS if x not in out_tasks
            ])
            eval_data_dirs = " ".join([f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in out_tasks])

            prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"
            log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} load_pretrained='[{prelim_directory}]' exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_unseen_one_in_composition_pfx_tune(self):
        commands = []
        for out_task in STYLEPTB_ATOMIC_TASKS:
            prelim_name = f"pfx_tune_atomics"
            exp_name = f"pfx_tune_unseen_{out_task}"
            common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=embedding_attention do_prefix_mixing=true num_compositions=2 seed={self.seed} base_config={self.base_config}"

            filtered_composite_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if out_task not in x]
            eval_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if x not in filtered_composite_tasks]

            train_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in filtered_composite_tasks
            ])
            eval_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in eval_tasks
            ])

            prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"
            log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} load_pretrained='[{prelim_directory}]' exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_unseen_two_in_composition_pfx_tune(self):
        commands = []
        for out_composite_task in STYLEPTB_COMPOSITE_TASKS:
            out_components = out_composite_task.split("+")

            prelim_name = f"pfx_tune_atomics"
            exp_name = f"pfx_tune_unseen_{'_'.join(out_components)}"
            common = f"num_gpus={self.num_gpus} mode=pfx_tune composition_mode=embedding_attention do_prefix_mixing=true seed={self.seed} base_config={self.base_config}"

            filtered_composite_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if all(_task not in x for _task in out_components)]
            eval_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if x not in filtered_composite_tasks]

            train_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in filtered_composite_tasks
            ])
            eval_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in eval_tasks
            ])
            
            prelim_directory = f"logs/pfx_tune/seed_{self.seed}/{prelim_name}"
            log_directory = f"logs/pfx_tune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} load_pretrained='[{prelim_directory}]' exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_hold1out_finetune(self) -> List[Tuple[str, str]]:
        commands = []
        for out_task in STYLEPTB_COMPOSITE_TASKS:
            exp_name = f"finetune_except_{out_task}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            train_data_dirs = " ".join([
                f'data_dir.{x}={self.task_name_to_datapath(x)}'
                for x in STYLEPTB_COMPOSITE_TASKS + STYLEPTB_ATOMIC_TASKS if x != out_task
            ])
            eval_data_dir = f"data_dir.{out_task}={self.task_name_to_datapath(out_task)}"

            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dir} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_holdkout_finetune(self):
        commands = []
        for num_outs in range(2, len(STYLEPTB_COMPOSITE_TASKS) + 1, 2):
            out_tasks = SHUFFLED_STYLEPTB_COMPOSITE_TASKS[:num_outs]
            print(out_tasks)
            exp_name = f"finetune_except_{num_outs}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            train_data_dirs = " ".join([
                f'data_dir.{x}={self.task_name_to_datapath(x)}'
                for x in STYLEPTB_COMPOSITE_TASKS + STYLEPTB_ATOMIC_TASKS if x not in out_tasks
            ])
            eval_data_dirs = " ".join([f'data_dir.{x}={self.task_name_to_datapath(x)}' for x in out_tasks])

            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_unseen_one_in_composition_finetune(self):
        commands = []
        for out_task in STYLEPTB_ATOMIC_TASKS:
            exp_name = f"finetune_unseen_{out_task}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            filtered_composite_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if out_task not in x]
            eval_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if x not in filtered_composite_tasks]

            train_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in filtered_composite_tasks + STYLEPTB_ATOMIC_TASKS
            ])
            eval_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in eval_tasks
            ])

            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_unseen_two_in_composition_finetune(self):
        commands = []
        for out_composite_task in STYLEPTB_COMPOSITE_TASKS:
            out_components = out_composite_task.split("+")

            exp_name = f"finetune_unseen_{'_'.join(out_components)}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            filtered_composite_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if all(_task not in x for _task in out_components)]
            eval_tasks = [x for x in STYLEPTB_COMPOSITE_TASKS if x not in filtered_composite_tasks]

            train_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in filtered_composite_tasks + STYLEPTB_ATOMIC_TASKS
            ])
            eval_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}"
                for x in eval_tasks
            ])
            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dirs} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_minimal_learning_finetune(self):
        commands = []
        for target_task in STYLEPTB_COMPOSITE_TASKS:
            components = target_task.split("+")

            exp_name = f"finetune_minimal_{target_task}"
            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            train_data_dirs = " ".join([
                f"data_dir.{x}={self.task_name_to_datapath(x)}" for x in components
            ])
            eval_data_dir = f"data_dir.{target_task}={self.task_name_to_datapath(target_task)}"
            log_directory = f"logs/finetune/seed_{self.seed}/{exp_name}"

            train_command = f"python scripts/train.py {train_data_dirs} exp_name={exp_name} num_epochs=-1 patience=5 max_steps=-1 {common}"
            eval_command = f"python scripts/eval.py {eval_data_dir} load_pretrained='[{log_directory}]' {common} &> {log_directory}/eval.log"

            commands.append((train_command, eval_command))
        return commands

    def stylePTB_eval_pipeline_finetune(self):
        commands = []
        for target_task in STYLEPTB_COMPOSITE_TASKS:
            components = target_task.split("+")
            prelim_directory = f"logs/finetune/seed_{self.seed}/finetune_ingredients"

            common = f"num_gpus={self.num_gpus} mode=finetune seed={self.seed} base_config={self.base_config}"

            eval_data_dir = f"data_dir.{target_task}={self.task_name_to_datapath(target_task)}"

            command1 = f"python scripts/eval_pipeline.py {eval_data_dir} " \
                      f"composition_tasks='[{components[0]}, {components[1]}]' load_pretrained='[{prelim_directory}, {prelim_directory}]' {common} " \
                      f"&> {components[0]}--{components[1]}.log"
            command2 = f"python scripts/eval_pipeline.py {eval_data_dir} " \
                      f"composition_tasks='[{components[1]}, {components[0]}]' load_pretrained='[{prelim_directory}, {prelim_directory}]' {common} " \
                      f"&> {components[1]}--{components[0]}.log"

            commands.append((command1, command2))
        return commands


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--mode', choices=["finetune"], default="no_use")
    #parser.add_argument('--experiment', choices=["full", "single", "hold1out", "holdkout", "unseen_one", "unseen_two", "prelim_pfx", "full_pfx", "hold1out_pfx", "holdkout_pfx", "unseen_one_pfx", "unseen_two_pfx", "concatenation"])
    parser.add_argument('--experiment', choices=["full", "single", "pipeline", "minimal", "hold1out", "holdkout", "unseen_one", "unseen_two", "prelim_pfx", "full_pfx", "hold1out_pfx", "holdkout_pfx", "unseen_one_pfx", "unseen_two_pfx", "concatenation"])
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]

    generator = ScriptGenerator(allocated_gpus=gpus, mode=args.mode, experiment=args.experiment)
    generator.gen_scripts()
