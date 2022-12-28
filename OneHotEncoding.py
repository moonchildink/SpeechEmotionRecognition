from init import PreProcessing
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def getOneHotLabel():
    enc = OneHotEncoder()
    y = enc.fit_transform(PreProcessing()['label'].values.reshape(-1, 1))
    return y.toarray()
