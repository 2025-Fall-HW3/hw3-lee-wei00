from Markowitz import RiskParityPortfolio, df

rp_w, rp_ret = RiskParityPortfolio("SPY").get_results()

print("=== RP Weights Shape ===")
print(rp_w.shape)

print("\n=== Columns ===")
print(rp_w.columns.tolist())

print("\n=== First 5 rows of RP Weights ===")
print(rp_w.head())

print("\n=== Last 5 rows of RP Weights ===")
print(rp_w.tail())
