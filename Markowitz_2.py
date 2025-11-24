"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # 1. 選出不用 SPY 的資產
        assets = self.price.columns[self.price.columns != self.exclude]

        # 2. 初始化權重表
        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns
        )

        # 3. 用全部歷史資料（2019-2024）算報酬的 mean & cov
        ret_hist = self.returns[assets]          # shape: (T, N)
        mu = ret_hist.mean()                     # 平均報酬 (N,)
        Sigma = ret_hist.cov()                   # 共變異數矩陣 (N,N)

        # 4. 解一個類似 maximum Sharpe 的權重: w ∝ Σ^{-1} μ
        #    用 pinv 避免不可逆
        try:
            Sigma_inv = np.linalg.pinv(Sigma.values)
            w_raw = Sigma_inv @ mu.values        # shape: (N,)
        except Exception:
            # 如果真的壞掉，就退回等權重
            w_raw = np.ones(len(assets))

        # 5. long-only：把負的砍掉
        w_raw = np.where(w_raw < 0, 0, w_raw)

        # 6. normalize 成 sum=1
        if w_raw.sum() == 0:
            w_final = np.ones(len(assets)) / len(assets)
        else:
            w_final = w_raw / w_raw.sum()

        # 7. 把這組固定權重套用到所有日期（SPY 欄位維持 0）
        for date in self.price.index:
            self.portfolio_weights.loc[date, assets] = w_final

        # 8. 確保沒有 NaN
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0.0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
