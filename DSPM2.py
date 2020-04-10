import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

lastPurchaseDate = datetime.datetime(2015, 5, 15)

sales_data = pd.read_csv("/home/koen/Documents/DSPM/Assignment 2/sales.csv",names=['SaleID','saleDateTime','accountName','coins','currency','priceInCurrency','priceInEUR','methodId','ip','ipCountry'])
sales_data.saleDateTime = pd.to_datetime(sales_data['saleDateTime'])
data_training = sales_data[sales_data['saleDateTime'].dt.year != 2015]
data_training = data_training[data_training['saleDateTime'].dt.year != 2014]
data_training = data_training[data_training['saleDateTime'].dt.year != 2013]
data_testing = sales_data[(sales_data['saleDateTime'].dt.year == 2012) ]

train_data = pd.DataFrame()
test_data = pd.DataFrame()

train_data["MoneySpentEUR"] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['priceInEUR'].sum().astype('int')
train_data["AverageTransactionSize"] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['priceInEUR'].mean().round().astype('int')
train_data['CountryCode'] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['ipCountry'].nth(0)
train_data['Method'] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['methodId'].nth(0).astype('category')
train_data['Transactions'] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['SaleID'].count().astype('int')
train_data['daysActive'] = (sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['saleDateTime'].max() - sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['saleDateTime'].min()).dt.days.fillna(0).astype('int')
train_data['daysInactive'] = (lastPurchaseDate - sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')

train_data['purchaseIn2015'] = sales_data[sales_data["saleDateTime"].dt.year ==2014].groupby('accountName')['priceInEUR'].sum().astype('int')
train_data['purchaseIn2015'].fillna(0)

test_data["MoneySpentEUR"] = data_testing.groupby('accountName')['priceInEUR'].sum().astype('int')
test_data["AverageTransactionSize"] = data_testing.groupby('accountName')['priceInEUR'].mean().round().astype('int')
test_data['CountryCode'] = data_testing.groupby('accountName')['ipCountry'].nth(0)
test_data['Method'] = data_testing.groupby('accountName')['methodId'].nth(0).astype('category')
test_data['Transactions'] = data_testing.groupby('accountName')['SaleID'].count().astype('int')
test_data['daysActive'] = (data_testing.groupby('accountName')['saleDateTime'].max() - data_testing.groupby('accountName')['saleDateTime'].min()).dt.days.astype('int')
test_data['daysInactive'] = (lastPurchaseDate - data_testing.groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')

test_data['purchaseIn2015'] = sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['priceInEUR'].sum().astype('int')
test_data['purchaseIn2015'].fillna(0)

avgTimeBetweenTransactions = []
for name, dates in sales_data[sales_data["saleDateTime"].dt.year ==2013].groupby('accountName')['saleDateTime']:
    datePrevious = datetime.datetime.now()
    interval = []
    for dateCount, date in enumerate(dates):
        if dateCount != 0:
            interval.append((date - datePrevious).total_seconds()/ (24 * 60 * 60))

        datePrevious = date
    avgTimeBetweenTransactions.append(np.mean(interval))
np.nan_to_num(avgTimeBetweenTransactions)
train_data['avgTimeBetweenTransactions'] = avgTimeBetweenTransactions
avgTimeBetweenTransactions = []
for name, dates in data_testing.groupby('accountName')['saleDateTime']:
    datePrevious = datetime.datetime.now()
    interval = []
    for dateCount, date in enumerate(dates):
        if dateCount != 0:
            interval.append((date - datePrevious).total_seconds()/ (24 * 60 * 60))

        datePrevious = date
    avgTimeBetweenTransactions.append(np.mean(interval))
np.nan_to_num(avgTimeBetweenTransactions)
test_data['avgTimeBetweenTransactions'] = avgTimeBetweenTransactions

print(train_data)
print(test_data)

results = pd.DataFrame()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data[["AverageTransactionSize",'Transactions','daysActive','daysInactive','avgTimeBetweenTransactions']].fillna(0), train_data['purchaseIn2015'].fillna(0))
results["prediction"] = clf.predict(test_data[["AverageTransactionSize",'Transactions','daysActive','daysInactive','avgTimeBetweenTransactions']].fillna(0).astype(int))
results["actual"] = test_data['purchaseIn2015'].fillna(0).astype(int)
print(results["actual"])
np.savetxt("test.txt",accuracy_score(results[1],results[2]))#accuracy_score(clf.predict(pred,actual))
print(accuracy_score(results[1],results[2]))



# labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
# cm = confusion_matrix(clf.predict(pred, test_data['purchaseIn2015'], labels )
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm, vmin=0, vmax=19000)
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
