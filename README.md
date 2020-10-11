# Attentive Visual Semantic Specialized Network for Video Captioning

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 
![DeepLearning](https://img.shields.io/badge/VideoCaptioning-DeepLearning-orange)
![](https://img.shields.io/badge/ICPRpaper-SourceCode-yellow)
![License](https://img.shields.io/github/license/WingsBrokenAngel/delving-deeper-into-the-decoder-for-video-captioning.svg?color=brightgreen&style=flat)

This repository is the source code for the paper named ***Attentive Visual Semantic Specialized Network for Video Captioning***

## Table of Contents
1. [Abstract](#abstract)
2. [Requirement](#requirement)
3. [Manual](#manual)
4. [Results](#results)
    1. [Comparison on MSVD](#msvd)
    2. [Comparison on MSR-VTT](#cm)
5. [Data](#data)
6. [Citation](#citation)

## <a name="abstract"></a> Abstract

As an essential high-level task of video understanding topic, automatically describing a video with natural language has recently gained attention as a fundamental challenge in computer vision. 
Previous models for video captioning have several limitations, such as the existence of gaps in current semantic representations and the inexpressibility of the generated captions. 
To deal with these limitations, in this paper, we present a new architecture that we call *Attentive Visual Semantic Specialized Network* (AVSSN), which is an encoder-decoder model based on our Adaptive Attention Gate and Specialized LSTM layers. 
This architecture can selectively decide when to use visual or semantic information into the text generation process. 
The adaptive gate makes the decoder to automatically select the relevant information for providing a better temporal state representation than the existing decoders. 
Besides, the model is capable of learning to improve the expressiveness of generated captions attending to their length, using a sentence-length-related loss function. 
We evaluate the effectiveness of the proposed approach on the Microsoft Video Description (MSVD) and the Microsoft Research Video-to-Text (MSR-VTT) datasets, achieving state-of-the-art performance with several popular evaluation metrics: BLEU-4, METEOR, CIDEr, and ROUGE$_L$


<p float="left">
  <img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_s-lstm-model.png" width="400" />
  <img src="https://users.dcc.uchile.cl/~jeperez/media/2020/AVSSN_adaptive-fusion.png" width="300" /> 
</p>

## <a name="requirement"></a>Requirement
1. Python 3.6
2. Pytorch 1.2.0
3. pycocoevalcap (Python3)
4. NumPy

## <a name="manual"></a>Manual
## <a name="results"></a>Results
### <a name="msvd"></a>Comparison on MSVD
### <a name="msrvtt"></a>Comparison on MSR-VTT
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
