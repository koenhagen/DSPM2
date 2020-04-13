import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

lastPurchaseDate = datetime.datetime(2015, 5, 15)

sales_data = pd.read_csv("/home/koen/Documents/DSPM/Assignment 2/sales.csv",names=['SaleID','saleDateTime','accountName','coins','currency','priceInCurrency','priceInEUR','methodId','ip','ipCountry'])
sales_data.saleDateTime = pd.to_datetime(sales_data['saleDateTime'])

train_data = pd.DataFrame()
train_data_total = pd.DataFrame()
test_data = pd.DataFrame()

years = [2010,2011,2012]
for year in years:
    print(year)
    train_data.drop(train_data.index, inplace=True)
    train_data["MoneySpentEUR"] = sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['priceInEUR'].sum().astype('int')
    train_data["LastPurchase"] = sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['priceInEUR'].nth(-1).astype('int')
    train_data["AverageTransactionSize"] = sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['priceInEUR'].mean().round().astype('int')
    # train_data['Method'] = sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['methodId'].nth(0).astype('category')
    train_data['Transactions'] = sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['SaleID'].count().astype('int')
    train_data['daysActive'] = (sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['saleDateTime'].max() - sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['saleDateTime'].min()).dt.days.fillna(0).astype('int')
    train_data['daysInactive'] = (lastPurchaseDate - sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')
    countryIsGreece = []
    for name, country in sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['ipCountry']:
        # print(country, "FIRST", country.value_counts().idxmax())
        if country.iloc[0] == 'GR' or country.iloc[0] == 'TR' or country.iloc[0] == 'RO':
            countryIsGreece.append(1)
        else:
            countryIsGreece.append(0)
    train_data['countryIsGreece'] = countryIsGreece
    countryIsGreece.clear()

    avgTimeBetweenTransactions = []
    for name, dates in sales_data[sales_data["saleDateTime"].dt.year ==year].groupby('accountName')['saleDateTime']:
        datePrevious = datetime.datetime.now()
        interval = []
        for dateCount, date in enumerate(dates):
            if dateCount != 0:
                interval.append((date - datePrevious).total_seconds()/ (24 * 60 * 60))

            datePrevious = date
        avgTimeBetweenTransactions.append(np.mean(interval))
    np.nan_to_num(avgTimeBetweenTransactions)
    train_data['avgTimeBetweenTransactions'] = avgTimeBetweenTransactions
    avgTimeBetweenTransactions.clear()

    train_data['prediction'] = sales_data[sales_data["saleDateTime"].dt.year ==year+1].groupby('accountName')['priceInEUR'].sum().fillna(0).astype('bool').astype('int').fillna(0)
    train_data_total = train_data_total.append(train_data, ignore_index = True)


data_testing = sales_data[(sales_data['saleDateTime'].dt.year == 2013) ]
test_data["MoneySpentEUR"] = data_testing.groupby('accountName')['priceInEUR'].sum().astype('int')
test_data["LastPurchase"] = data_testing.groupby('accountName')['priceInEUR'].nth(-1).astype('int')
test_data["AverageTransactionSize"] = data_testing.groupby('accountName')['priceInEUR'].mean().round().astype('int')
# test_data['Method'] = data_testing.groupby('accountName')['methodId'].nth(0).astype('category')
test_data['Transactions'] = data_testing.groupby('accountName')['SaleID'].count().astype('int')
test_data['daysActive'] = (data_testing.groupby('accountName')['saleDateTime'].max() - data_testing.groupby('accountName')['saleDateTime'].min()).dt.days.astype('int')
test_data['daysInactive'] = (lastPurchaseDate - data_testing.groupby('accountName')['saleDateTime'].max()).dt.days.astype('int')

countryIsGreece = []
for name, country in data_testing.groupby('accountName')['ipCountry']:
    if country.iloc[0] == 'GR' or country.iloc[0] == 'TR' or country.iloc[0] == 'RO':
        countryIsGreece.append(1)
    else:
        countryIsGreece.append(0)
test_data['countryIsGreece'] = countryIsGreece

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

test_data['prediction'] = sales_data[sales_data["saleDateTime"].dt.year ==2015].groupby('accountName')['priceInEUR'].sum().astype('bool').astype('int').fillna(0)
test_data['prediction'].fillna(0)

print(train_data_total)
print(test_data)

print(train_data_total[train_data_total.columns[0:]].fillna(0).corr()['prediction'][:])

results = pd.DataFrame()
regression = LogisticRegression()
tree = tree.DecisionTreeClassifier(criterion='gini',min_samples_leaf=20)
vectorMachine = svm.SVC(kernel='poly',degree=3)
neuralNetwork = MLPClassifier(max_fun=15000, max_iter=500)
for method in neuralNetwork,tree:
    results.drop(results.index, inplace=True)
    method = method.fit(train_data_total[["MoneySpentEUR","LastPurchase","AverageTransactionSize",'Transactions','daysActive','daysInactive','avgTimeBetweenTransactions','countryIsGreece']].fillna(0), train_data_total['prediction'].fillna(0))
    results["prediction"] = method.predict(test_data[["MoneySpentEUR","LastPurchase","AverageTransactionSize",'Transactions','daysActive','daysInactive','avgTimeBetweenTransactions','countryIsGreece']].fillna(0).astype(int))
    results["actual"] = test_data['prediction'].fillna(0).astype(int).tolist()
    print (results, "RESULTS")
    print(method,accuracy_score(results["prediction"],results["actual"]))

# np.savetxt("test.txt",results)

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
# cm = confusion_matrix(clf.predict(pred, test_data['prediction'], labels )
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
