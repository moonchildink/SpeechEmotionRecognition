# 基于MFCC特征提取以及LSTM的语音情感识别
*author: moonchild*
*date: 2022-12-28*

## 1. 项目简介

本项目目前主要包括以下几个部分：
- 语音数据集的准备
  - 此处使用了kaggle上的toronto数据集,[链接](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
- MFCC特征提取
  - > MFCC:在声音处理领域中，梅尔频率倒谱(Mel-Frequency Cepstrum)是基于声音频率的非线性梅尔刻度(mel scale)的对数能量频谱的线性变换。梅尔频率倒谱系数 (Mel-Frequency Cepstral Coefficients，MFCCs)就是组成梅尔频率倒谱的系数。它衍生自音讯片段的倒频谱(cepstrum)。倒谱和梅尔频率倒谱的区别在于，梅尔频率倒谱的频带划分是在梅尔刻度上等距划分的，它比用于正常的对数倒频谱中的线性间隔的频带更能近似人类的听觉系统。 这样的非线性表示，可以在多个领域中使声音信号有更好的表示。例如在音讯压缩中。梅尔频率倒谱系数（MFCC）广泛被应用于语音识别的功能。他们由Davis和Mermelstein在1980年代提出，并在其后持续是最先进的技术之一。在MFCC之前，线性预测系数（LPCS）和线性预测倒谱系数（LPCCs）是自动语音识别的的主流方法

    -此处我使用的`librosa`提供mfcc函数
- LSTM模型的构建
- 模型的训练与评估

目前已完成模型的训练与评估部分.
