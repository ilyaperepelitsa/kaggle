import tensorflow as tf

import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd


pew = pd.read_csv("/Users/ilyaperepelitsa/quant/kaggle/mnist/data/external_tripple_3800_submission.csv")
pew = pew.loc[~(pew["Label"] == "Label"),:]
pew.to_csv("/Users/ilyaperepelitsa/quant/kaggle/mnist/data/external_tripple_3800_submission.csv", index=False)

# pew[['ImageId', 'Label']] = pew[['ImageId', 'Label']].astype(int)
