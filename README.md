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
4. [Results](#results)
    1. [Comparison on MSVD](#msvd)
    2. [Comparison on MSR-VTT](#cm)
5. [Data](#data)
6. [Citation](#citation)

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
2. Pytorch 1.2.0
3. pycocoevalcap (Python3)
4. NumPy

## <a name="manual"></a>Manual

## <a name="results"></a>Qualitative Results
<img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_examples.png" alt="qualitative results" height="400"/>

## <a name="msvd"></a>Quantitative Results

| Dataset | B-4      | C        | M        | R        
| :------ | :------: | :------: | :------: | :------:
|MSVD     | 62.3     | 39.2     | 107.7    | 78.3
|MSR-VTT  | 45.5     | 31.4     | 50.6     | 64.3

## <a name="data"></a>Data
## <a name="citation"></a>Citation
```
@article{PerezMartin2020AttentiveCaptioning,
	title={Attentive Visual Semantic Specialized Network for Video Captioning},
	author={Jesus Perez-Martin and Benjamin Bustos and Jorge Pérez},
	booktitle={25th International Conference on Pattern Recognition},
	year={2020}
}
```
