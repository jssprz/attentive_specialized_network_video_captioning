# Attentive Visual Semantic Specialized Network for Video Captioning

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 
![DeepLearning](https://img.shields.io/badge/VideoCaptioning-DeepLearning-orange)
![](https://img.shields.io/badge/ICPRpaper-SourceCode-yellow)
![License](https://img.shields.io/github/license/WingsBrokenAngel/delving-deeper-into-the-decoder-for-video-captioning.svg?color=brightgreen&style=flat)

This repository is the source code for the paper named ***Attentive Visual Semantic Specialized Network for Video Captioning***.
In this paper, we present a new architecture that we call *Attentive Visual Semantic Specialized Network* (AVSSN), which is an encoder-decoder model based on our Adaptive Attention Gate and Specialized LSTM layers. 
This architecture can selectively decide when to use visual or semantic information into the text generation process. 
The adaptive gate makes the decoder to automatically select the relevant information for providing a better temporal state representation than the existing decoders. 
We evaluate the effectiveness of the proposed approach on the Microsoft Video Description (MSVD) and the Microsoft Research Video-to-Text (MSR-VTT) datasets, achieving state-of-the-art performance with several popular evaluation metrics: BLEU-4, METEOR, CIDEr, and ROUGE_L.

## Table of Contents
1. [Model](#model)
2. [Requirement](#requirement)
3. [Manual](#manual)
4. [Qualitative Results](#qualitative)
5. [Quantitative Results](#quantitative)
7. [Citation](#citation)

## <a name="model"></a>Model

<table>
  <tr>
    <td style="text-align: center;"><img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_s-lstm-model.png" height=300></td>
    <td style="text-align: center;"><img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_adaptive-fusion.png" height=300></td>
  </tr>
  <tr>
    <td>Proposed  Adaptive  Visual  Semantic  Specialized  Network  (AVSSN)</td>
    <td>Adaptive Attention Gate</td>
  </tr>
 </table>

## <a name="requirement"></a>Requirement
1. Python 3.6
2. PyTorch 1.2.0
4. NumPy

## <a name="manual"></a>Manual
### Download code
```
git clone --recursive https://github.com/jssprz/attentive-specialized-network-video-captioning.git
```

### Download Data
```
mkdir -p data/MSVD && wget -i msvd_data.txt -P data/MSVD
mkdir -p data/MSR-VTT && wget -i msrvtt_data.txt -P data/MSR-VTT
```

### Testing

1. Download pre-trained models (at epoch 15)

```
wget https://s06.imfd.cl/04/github-data/AVSSN/MSVD/captioning_chkpt_15.pt -P data/MSVD
wget https://s06.imfd.cl/04/github-data/AVSSN/MSR-VTT/captioning_chkpt_15.pt -P data/MSR-VTT
```

2. Generate captions for test samples

```
python test.py -chckpt pretrain/MSVD/captioning_chkpt_15.pt -data data/MSVD/ -out results/MSVD/
python test.py -chckpt pretrain/MSR-VTT/captioning_chkpt_15.pt -data data/MSR-VTT/ -out results/MSR-VTT/
```

3. Metrics

- MSVD
```
evaluate.py -gen results/MSVD/preductions.txt -ref data/MSVD/test_references.txt
```

- MSR-VTT
```
evaluate.py -gen results/MSR-VTT/preductions.txt -ref data/MSR-VTT/test_references.txt
```

## <a name="qualitative"></a>Qualitative Results
<img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_examples.png" alt="qualitative results" height="400"/>

## <a name="quantitative"></a>Quantitative Results

| Dataset | epoch    | B-4      | C        | M        | R        
| :------ | :------: | :------: | :------: | :------: | :------:
|MSVD     | 100      | 62.3     | 39.2     | 107.7    | 78.3
|MSR-VTT  | 60       | 45.5     | 31.4     | 50.6     | 64.3

## <a name="citation"></a>Citation
```
@article{PerezMartin2020AttentiveCaptioning,
	title={Attentive Visual Semantic Specialized Network for Video Captioning},
	author={Jesus Perez-Martin and Benjamin Bustos and Jorge Pérez},
	booktitle={25th International Conference on Pattern Recognition},
	year={2020}
}
```
