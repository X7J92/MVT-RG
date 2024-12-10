# Disentangling Inter- and Intra-Video Relations for Multi-event Video-Text Retrieval and Grounding

<div align="center">
    <img src="images/01.png" alt="Main results on TACoS" width="400">
</div>

Abstract:Video-text retrieval aims to precisely search for videos most relevant to a text query within a video corpus. However, existing methods are largely limited to single-text (single-event) queries, which are not effective at handling multi-text (multi-event) queries. Furthermore, these methods typically focus solely on retrieval and do not attempt to locate multiple events within the retrieved videos. To address these limitations, our paper proposes a novel method named Disentangling Inter- and Intra-Video relations, which jointly considers multi-event video-text retrieval and grounding. This method leverages both inter-video and intra-video event relationships to enhance the performance of retrieval and grounding. At the retrieval level, we devise a Relational Event-Centric Video-Text Retrieval module, based on the principle that more comprehensive textual information leads to a more precise correspondence between text and video. It incorporates event relationship features at different hierarchical levels and exploits the hierarchical structure of corresponding video relationships to achieve multi-level contrastive learning between events and videos. This approach enhances the richness, accuracy, and comprehensiveness of event descriptions, improving alignment precision between text and video and enabling effective differentiation among videos. For event localization, we propose Event Contrast-Driven Video Grounding, which accounts for positional differences between different events and achieves precise grounding of multiple events through divergence learning of event locations. Our solution not only provides efficient text-to-video retrieval capabilities but also accurately locates events within the retrieved videos, addressing the shortcomings of existing methods. Extensive experimental results on the ActivityNet-Captions and Charades-STA benchmark datasets demonstrate the superior performance of our method, clearly validating its effectiveness. The innovation of this research lies in introducing a new joint framework for video-text retrieval and multi-event localization, while offering new ideas for further research and applications in related fields.


## News
- :beers: Our paper has been submitted to the TMM.

## Framework
![alt text](images/03.png)

## Main Results


#### Main results on ActivityNet Captions and Charades-STA
![alt text](images/t1.png)

#### Main results on TACoS

<div align="center">
    <img src="images/t2.png" alt="Main results on TACoS" width="550">
</div>

### Data Preparation
Please download the visual features from [here](https://pan.baidu.com/s/1_JiOUG3FKkKXij-0kVfkuA?pwd=ryeh) and save it to the `data/` folder. We expect the directory structure to be the following:

```
data
├── activitynet
│   ├── sub_activitynet_v1-3.c3d.hdf5
│   ├── glove.pkl
│   ├── train_data.json
│   ├── val_data.json
│   ├── test_data.json

```

## Prerequisites
- python 3.5
- pytorch 1.4.0
- torchtext
- easydict
- terminaltables

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
