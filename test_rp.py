import pandas as pd

# 換成你下載的檔案路徑
rp = pd.read_pickle("C:/Users/user/Downloads/rp.pkl")



rp.to_csv("rp_preview.csv")

print(rp.shape)
print(rp.head(30))

