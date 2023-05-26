# Mistry_Strock_NatureComm_2023

In this paper we use artificial neural networks to model the human visual perception of numerosity.
We use [CORnet](https://github.com/dicarlolab/CORnet) as a pre-trained visual model that we fine-tune to identify how many dots are present on a picture.

## Setting up environment

```bash
source $OAK/projects/astrock/2021_common/scripts/slurm/slurm_aliases.sh
```

## Generation of the dot dataset

```bash
submit8c dataset/enumeration9.py
```

## Model

Training the model
```bash
submit4g model/MODEL/train_enumeration9.py
```
Testing the model
```bash
submit1g model/MODEL/test_enumeration9.py --pepochs $(seq -1 49)
```

Three different learning algorithms are compared, ADAM (i.e. MODEL=cornet_adam), RMSProp (i.e. MODEL=cornet_rmsprop) and SGD (i.e. MODEL=cornet_sgd).
In the paper we use cornet_adam as main model. We use cornet_rmsprop and cornet_sgd as controls.

## Ablated models

Ablation of selective spontaneous number neurons (SPONs)
```bash
submit1g ablation/ablation_selective_spons_enumeration9.py --pepochs $(seq -1 49)
```

Ablation of selective persistents spontaneous number neurons (P-SPONs)
```bash
submit1g ablation/ablation_selective_pspons_enumeration9.py --pepochs $(seq -1 8) $(seq 9 10 49)
```

Ablation of all spontaneous number neurons (SPONs)
```bash
submit1g analysis/ablation/ablation_all_spons_enumeration9.py --pepochs $(seq -1 49)
```

Ablation of all persistents spontaneous number neurons (P-SPONs)
```bash
submit1g analysis/ablation/ablation_all_pspons_enumeration9.py --pepochs $(seq -1 49)
