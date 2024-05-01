# AbGraftBERT

## Introduction

<p align="center"><img src="AbGraftBERT/figures/graft.png" width=70%></p>
<p align="center"><b>Schematic illustration of AbGraftBERT</b></p>



## Dependencies

- pytorch==1.12.0
- fairseq==0.10.2
- numpy==1.23.3

## Pretrain a model on sequence data

All preprocessed data is from [ABGNN](https://github.com/KyGao/ABGNN), with the code primarily based on this repository, thanks!
For  training, we can run:
```shell
bash pretrain-selfgrafting.sh
```

## Finetune on sequence and structure co-design tasks

For experiment 1, raw data is from [MEAN](https://github.com/THUNLP-MT/MEAN).

For experiment 2, raw data is from [HSRN](https://github.com/wengong-jin/abdockgen). 

For experiment 3, raw data is from [RefineGNN](https://github.com/wengong-jin/RefineGNN). 

The finetuning scripts are following:

```shell
# for exp1
bash finetune-exp1.sh

# for exp2
bash finetune-exp2.sh

# for exp3
# have to additionally install pytorch_lightning, matplotlib, and igfold
bash finetune-exp3.sh
bash covid-optimize.sh
```

We can simply run the following code for inference:

```shell
python inference.py \
    --cdr_type ${CDR} \
    --cktpath ${model_ckpt_path}}/checkpoint_best.pt \
    --data_path ${dataset_path}
```
