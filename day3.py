#FIRST STEP OF EDA
import pandas as pd
df =pd.read_csv("sales.csv")
print("df.shape:", df.shape, sep="\n")
print("df.columns:", df.columns.tolist(),sep=
print("df.info():", sep="\n")
print(df.describe(),sep="\n")
df.info()