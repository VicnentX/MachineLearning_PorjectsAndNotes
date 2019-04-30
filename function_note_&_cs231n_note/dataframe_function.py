import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.rand(5, 2), index=range(0, 10, 2), columns=list("AB"))

print(df)

print(df.iloc[[2]])

print(df.loc[[2]])

# slice

print(df._slice(slice(0, 2)))
print(df._slice(slice(0, 2)), 0)
print(df._slice(slice(0, 2)), 1)
