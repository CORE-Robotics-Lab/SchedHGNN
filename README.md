# RA-L Submission
This repository contains 

1) a multi-agent implementation of PPO and is heavily based on https://github.com/zoeyuchao/mappo, where installation steps can be found.
2) a fast implementation of HetNet
3) the implementation and running scripts for Marine environment

## Note
You might have to add your own path variables in order to run the codes smoothly.

## Run
To change the level of game, please change the variable ```MODE``` defined in line 7 and 8 in the code ```waves.py```

To train and evaluate SchedHGNN,
```
cd onpolicy/scripts
sh train_marine_schedhgnn_5x5.sh
sh train_marine_schedhgnn_10x10.sh
```

To train and evaluate MAPPO,
```
cd onpolicy/scripts
sh train_marine_mappo_5x5.sh
sh train_marine_mappo_10x10.sh
```

To train and evaluate HetNet,
```
cd onpolicy/scripts
sh train_marine_hetnet_5x5.sh
sh train_marine_hetnet_10x10.sh
```

## Installation Notes
To satisfy the minimum torch requirements do:

- CPU:
```
pip install torch===1.9.0+cpu torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

- GPU:
```
pip install torch===1.9.0+cu102 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

note that in case of using GPU version, you need to also install the GPU version of the DGL by: ```pip install dgl-cu110```. For CPU mode, just do ```pip install dgl```.

