from LSTM import LSTMModel
import IPython.display as ipd
import librosa
import numpy as np
import random
from init import PreProcessing


class Predict:
    def __init(self):
        self.model = LSTMModel()
        self.dataframe = PreProcessing()

    def getSentiment(self, address):
        data, sampling_rate = librosa.load(address, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        mfccsscaled = np.expand_dims(mfccsscaled, axis=-1)
        mfccsscaled = np.expand_dims(mfccsscaled, axis=0)
        prediction = self.model.predict_on_batch(mfccsscaled)
        return prediction

    def getSentimentType(li):
        li = li.tolist()[0]
        print(li)
        emotionList = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps']
        Credibility = max(li) / sum(li)
        # print(li.index(max(li)))
        type = emotionList[li.index(max(li))]
        print('该条语音的情感类型为:', type, '. 预测置信度为:', Credibility)
        return type, Credibility


if __name__ == '__main__':
    # 随机选择num条语音进行测试
    num = random.randint(1, 20)
    pre = Predict()
    for i in range(num):
        randomNum = random.randrange(2799)
        filePath = pre.dataframe.at[randomNum, 'file']
        print("选中的数据是", filePath)
        fileSentiment = pre.dataframe.at[randomNum, 'label']
        print('该条语音的情感状态是', fileSentiment)
        ans = pre.getSentiment(filePath)
        # ipd.Audio(filePath)
        type, Credibility = pre.getSentimentType(ans)
        if type == fileSentiment:
            print('预测正确')
        else:
            print('预测错误')
