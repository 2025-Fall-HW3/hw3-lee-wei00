import pandas as pd

# 換成你下載的檔案路徑
df = pd.read_pickle("C:/Users/user/Downloads/rp.pkl")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

