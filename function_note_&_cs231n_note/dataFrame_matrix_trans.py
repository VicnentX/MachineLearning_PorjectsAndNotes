import pandas as pd
import numpy as np
import tensorflow as tf


df = pd.DataFrame(np.random.randn(3, 4), columns=list("abcd"))
print(df)
print("_____________")
print(df.values)
print("_____________")
dfT = df.transpose()
print(dfT)
print(type(dfT))


