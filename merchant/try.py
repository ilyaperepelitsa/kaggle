import pandas as pd

pd.read_csv("/Volumes/data_pew/numeric/merchant/train.csv")
# pd.read_csv("/Volumes/data_pew/numeric/merchant/test.csv")

transactions = pd.read_csv("/Volumes/data_pew/numeric/merchant/historical_transactions.csv")
new_transactions = pd.read_csv("/Volumes/data_pew/numeric/merchant/new_merchant_transactions.csv")
merchants = pd.read_csv("/Volumes/data_pew/numeric/merchant/merchants.csv")


transactions.info()
transactions.info()

merchants.info()
merchants.loc[:,merchants.columns.str.contains("_id")].info()
transactions.loc[:,transactions.columns.str.contains("_id")].info()
