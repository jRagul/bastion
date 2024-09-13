from keras.utils import pad_sequences
from keras.models import load_model
import numpy as np
model = load_model("app/three_column.h5")
def preder(input):
    padded = pad_sequences(input[0], maxlen=200).tolist()
    res = (model.predict(padded) > 0.55).astype(int).tolist()
    return res
    