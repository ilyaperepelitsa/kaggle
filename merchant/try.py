import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)


pd.read_csv("/Volumes/data_pew/numeric/merchant/train.csv")
pd.read_csv("/Volumes/data_pew/numeric/merchant/test.csv")

transactions = pd.read_csv("/Volumes/data_pew/numeric/merchant/historical_transactions.csv")
new_transactions = pd.read_csv("/Volumes/data_pew/numeric/merchant/new_merchant_transactions.csv")
merchants = pd.read_csv("/Volumes/data_pew/numeric/merchant/merchants.csv")


transactions.info()
new_transactions.info()
merchants.info()


merchants.loc[:,merchants.columns.str.contains("_id")].info()
transactions.loc[:,transactions.columns.str.contains("_id")].info()



merchants.loc[:,~merchants.columns.str.contains("_id")].describe()
transactions.loc[:,~transactions.columns.str.contains("_id")].describe()
transactions.loc[:,transactions.columns.str.contains("_id")].describe()
transactions.describe()

# transactions.dropna().shape
# transactions.shape
# 26169697 / 29112361
merchants.head()
transactions.head()
