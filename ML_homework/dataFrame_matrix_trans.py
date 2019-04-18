import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.DataFrame(np.random.randn(3, 4), columns=list("abcd"))
print(df)
print("_____________")
print(df.values)
print("_____________")
dfT = df.transpose
dfdfT = tf.matmul(df, dfT)
print("dfdfT type", type(dfdfT))
print(dfdfT)
print("df type", type(df))
