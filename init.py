# 读入文件,初始化
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class PreProcessing:
    def __init__(self):
        fileList = []
        filePath = os.walk(r'D:\SpeechEmotionAnalysis\dataset\toronto')
        for path, dirs, files in filePath:
            for file in files:
                fileList.append(os.path.join(path, file))
        fileList = fileList[4:]

        label = [file.split('_')[-1].split('.')[0] for file in fileList]
        self.df = pd.DataFrame({'file': fileList, 'label': label})

    def getFileNameList(self):
        return self.df['file'].tolist()

    def getLabelList(self):
        return self.df['label'].tolist()
