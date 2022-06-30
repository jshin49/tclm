# Task Compositional LM Based on Prefix-Tuning

This repository contains modification and application of prefix tuning for research purposes. 
See the original code here: https://github.com/XiangLi1999/PrefixTuning 
Original paper: https://aclanthology.org/2021.acl-long.353.pdf

## Requirements
The orginal work of prefix tuning already directly modifies and imports `transformers` 3.2.0, which is a well known python library.
You are recommended to have a virtual Python environment for this repo to avoid conflicts.


### Start with Python 3.9 virtual environment
```
conda create -n tclm python=3.9
conda activate tclm
```

### install packages
```
pip install -r tclm/requirements.txt
pip install -e tclm
pip install -e .
```
Open a Python shell and run following code:
```
import nltk
nltk.download('punkt')
```
or you can just run `python tclm/utils/nltk_setup.py`.


### Example run 
```
cd tclm
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py data_dir.PTA+PPR=../StylePTB/XsumReformedDatasets/Compositional/Tense+Voice+PP_Removal/PTA+PPR num_gpus=1
```

### Evaluation 
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py data_dir.PTA+PPR=../StylePTB/XsumReformedDatasets/Compositional/Tense+Voice+PP_Removal/PTA+PPR num_gpus=1 load_pretrained='[logs/PTA/pfx_embed_tune/seed_1, logs/PPR/pfx_embed_tune/seed_1]' composition_mode=concatenation
```

### Pipeline Evaluation
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_pipeline.py data_dir.PTA+PPR=../StylePTB/XsumReformedDatasets/Compositional/Tense+Voice+PP_Removal/PTA+PPR num_gpus=1 load_pretrained='[logs/PTA/pfx_embed_tune/seed_1, logs/PPR/pfx_embed_tune/seed_1]'
```

