import pandas as pd

pd.read_csv("/Volumes/data_pew/numeric/merchant/train.csv")
# pd.read_csv("/Volumes/data_pew/numeric/merchant/test.csv")

transactions = pd.read_csv("/Volumes/data_pew/numeric/merchant/historical_transactions.csv")
merchants = pd.read_csv("/Volumes/data_pew/numeric/merchant/merchants.csv")


transactions.info()
merchants.info()
