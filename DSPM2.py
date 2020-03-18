import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

lastPurchaseDate = datetime.datetime(2015, 5, 15)

sales_data = pd.read_csv("/home/koen/Documents/DSPM/Assignment 2/sales.csv",names=['SaleID','saleDateTime','accountName','coins','currency','priceInCurrency','priceInEUR','methodId','ip','ipCountry'])
sales_data.saleDateTime = pd.to_datetime(sales_data['saleDateTime'])
data_2011 = sales_data[sales_data['saleDateTime'].dt.year == 2015]
data_2012 = sales_data[sales_data['saleDateTime'].dt.year != 2015]

train_data = pd.DataFrame()
train_data = pd.DataFrame()
test_data = pd.DataFrame()
test_data = pd.DataFrame()

train_data["MoneySpentEUR"] = data_2011.groupby('accountName')['priceInEUR'].sum().astype('int')
train_data["AverageTransactionSize"] = data_2011.groupby('accountName')['priceInEUR'].mean().round().astype('int')
train_data['CountryCode'] = data_2011.groupby('accountName')['ipCountry'].nth(0)
train_data['Method'] = data_2011.groupby('accountName')['methodId'].nth(0).astype('category')
train_data['Transactions'] = data_2011.groupby('accountName')['SaleID'].count().astype('int')
train_data['daysActive'] = (data_2011.groupby('accountName')['saleDateTime'].max() - data_2011.groupby('accountName')['saleDateTime'].min()).dt.days.astype('int')
train_data['daysInactive'] = (lastPurchaseDate - data_2011.groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')

test_data["MoneySpentEUR"] = data_2012.groupby('accountName')['priceInEUR'].sum().astype('int')
test_data["AverageTransactionSize"] = data_2012.groupby('accountName')['priceInEUR'].mean().round().astype('int')
test_data['CountryCode'] = data_2012.groupby('accountName')['ipCountry'].nth(0)
test_data['Method'] = data_2012.groupby('accountName')['methodId'].nth(0).astype('category')
test_data['Transactions'] = data_2012.groupby('accountName')['SaleID'].count().astype('int')
test_data['daysActive'] = (data_2012.groupby('accountName')['saleDateTime'].max() - data_2012.groupby('accountName')['saleDateTime'].min()).dt.days.astype('int')
test_data['daysInactive'] = (lastPurchaseDate - data_2012.groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')

# avgTimeBetweenTransactions = []
# for name, dates in sales_data.groupby('accountName')['saleDateTime']:
#     datePrevious = datetime.datetime.now()
#     interval = []
#     for dateCount, date in enumerate(dates):
#         if (len(dates)== 1):
#             interval.append(date-date)
#         if dateCount != 0:
#             interval.append(date - datePrevious)
#
#         datePrevious = date
#
#     avgTimeBetweenTransactions.append(np.mean(interval))
# train_data['avgTimeBetweenTransactions'] = avgTimeBetweenTransactions

# print(customer_data)
# sns.pairplot(customer_data.head(1000), hue="Method")
# plt.show()

print(train_data)
print(test_data)

clf = tree.DecisionTreeClassifier()
# .OneHotEncoder(categories=[train_data['Method'], train_data['CountryCode']])
# tree.OneHotEncoder(categories=[train_data['Method'], train_data['CountryCode']])
clf = clf.fit(train_data[["AverageTransactionSize",'Transactions','daysActive','daysInactive']], train_data["MoneySpentEUR"])

tree.plot_tree(clf.fit(test_data[["AverageTransactionSize",'Transactions','daysActive','daysInactive']], test_data["MoneySpentEUR"]))
