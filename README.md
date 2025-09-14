
## Training
Use the following commands for training:
```
cd moment_localization && export CUDA_VISIBLE_DEVICES=0
python dense_train.py --verbose --cfg ../experiments/dense_activitynet/acnet.yaml
```
## test
Use the following commands for test:
```
cd moment_localization && export CUDA_VISIBLE_DEVICES=0
python best_test.py --verbose --cfg ../experiments/dense_activitynet/acnet.yaml
```

We also provide several checkpoints for reproducing our experiment results. You can download them from [baidu drive](https://pan.baidu.com/s/1xWC90AIDImVJfKV9qcah4Q), put them under ```checkpoint/``` and use the above scripts to evaluate them.


## e-mail

call me: mengzhaowangg@163.com
