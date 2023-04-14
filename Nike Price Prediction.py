'''
Author: Group 11
Description: Nike Price Prediction
'''
'''Various Imports'''
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.model_selection import train_test_split,validation_curve
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import LinearSVC
import sys
import pandas as pd
import numpy as np

'''Reading Dataset'''
print(f'Reading Dataset...')
nike=pd.read_csv("nike.csv",index_col=0)

'''Data cleaning: All data cleaning is performed in the dataset before imported'''
print(f'Cleaning Data...')

'''Creating Train and Test set'''
print(f'Creating Training and Testing Set...')
x_nike,y_nike=nike.iloc[:,[1,2,3,4,5,6,7]],nike['Close_NIKE']
x_train,x_test,y_train,y_test=train_test_split(x_nike,y_nike)

print(f'--------------------------------------------------------------------')

'''Best Parameter Test'''
print(f'Testing Best Parameters...(This will take 2 mins)')

range1=[0.001,0.01,1,10,20,50,100]
range2=[1,3,5,10,40,70,100]
range3=[5,10,30,50,70,100,None]
range4=[100,200,300,400,450,500,600]
#Lasso
tr_sc,ts_sc=validation_curve(Lasso(max_iter=1000000),x_train,y_train,param_name='alpha',param_range=range1,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best alpha for Lasso is: {range1[ind]}')

#Ridge
tr_sc,ts_sc=validation_curve(Ridge(max_iter=1000000),x_train,y_train,param_name='alpha',param_range=range1,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best alpha for Ridge is: {range1[ind]}')

#K_nearest_neighbors
tr_sc,ts_sc=validation_curve(KNeighborsRegressor(),x_train,y_train,param_name='n_neighbors',param_range=range2,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best n_neighbors for K_nearest_neighbors is: {range2[ind]}')

#Decision Tree
tr_sc,ts_sc=validation_curve(DecisionTreeRegressor(),x_train,y_train,param_name='max_leaf_nodes',param_range=range3,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best max_leaf_nodes for Decision Tree is: {range3[ind]}')

tr_sc,ts_sc=validation_curve(DecisionTreeRegressor(),x_train,y_train,param_name='max_depth',param_range=range3,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best max_depth for Decision Tree is: {range3[ind]}')

#Random Forest
tr_sc,ts_sc=validation_curve(RandomForestRegressor(),x_train,y_train,param_name='max_leaf_nodes',param_range=range3,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best max_leaf_nodes for Random Forest is: {range3[ind]}')

tr_sc,ts_sc=validation_curve(RandomForestRegressor(),x_train,y_train,param_name='max_depth',param_range=range3,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best max_depth for Random Forest is: {range3[ind]}')

tr_sc,ts_sc=validation_curve(RandomForestRegressor(),x_train,y_train,param_name='n_estimators',param_range=range4,cv=4)
ind=np.argmax(ts_sc.mean(axis=1))
print(f'The best n_estimators for Random Forest is: {range4[ind]}')

print(f'--------------------------------------------------------------------')


'''Model Accuracy Test'''

print(f'Testing Model Accuracy...(This will take 5 mins)')

#Lasso
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=Lasso(alpha=0.01,max_iter=1000000)
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for Lasso: {min(result):.3%}')
print(f'The max accuracy for Lasso: {max(result):.3%}')
print(f'The average accuracy for Lasso: {np.mean(result):.3%}')
print("\n")

#Ridge
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=Ridge(alpha=20,max_iter=1000000)
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for Ridge: {min(result):.3%}')
print(f'The max accuracy for Ridge: {max(result):.3%}')
print(f'The average accuracy for Ridge: {np.mean(result):.3%}')
print("\n")

#Linear Regression
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=LinearRegression()
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for Linear Regression: {min(result):.3%}')
print(f'The max accuracy for Linear Regression: {max(result):.3%}')
print(f'The average accuracy for Linear Regression: {np.mean(result):.3%}')
print("\n")

#K_nearest_neighbors
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=KNeighborsRegressor(n_neighbors=10)
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for KNN: {min(result):.3%}')
print(f'The max accuracy for KNN: {max(result):.3%}')
print(f'The average accuracy for KNN: {np.mean(result):.3%}')
print("\n")

#Decision Tree
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=DecisionTreeRegressor(max_depth=5,max_leaf_nodes=50)
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for Decision Tree: {min(result):.3%}')
print(f'The max accuracy for Decision Tree: {max(result):.3%}')
print(f'The average accuracy for Decision Tree: {np.mean(result):.3%}')
print("\n")

#Random Forest
result=[]
for _ in range(50):
   x_train, x_test, y_train, y_test = train_test_split(x_nike, y_nike)
   ls=RandomForestRegressor(n_estimators=200)
   ls.fit(x_nike,y_nike)
   result.append(ls.score(x_test,y_test))
print(f'The min accuracy for Random Forest: {min(result):.3%}')
print(f'The max accuracy for Random Forest: {max(result):.3%}')
print(f'The average accuracy for Random Forest: {np.mean(result):.3%}')

print(f'--------------------------------------------------------------------')

'''Feature Importance & Coefficient'''

print(f'Calculating Feature Importance & Coefficient...(This will take less than 1 min)')

#Lasso
print(f'Most Important Feature for Lasso is:')
ls=Lasso(alpha=0.01,max_iter=1000000)
ls.fit(x_nike,y_nike)
ls_imp = pd.DataFrame(data=ls.coef_, index=x_nike.columns, columns=['Coefficient'])
ls_imp = ls_imp.sort_values(by=['Coefficient'], ascending=False)
print(ls_imp)
print("\n")

#Ridge
print(f'Most Important Feature for Ridge is:')
ls=Ridge(alpha=20,max_iter=1000000)
ls.fit(x_nike,y_nike)
ls_imp = pd.DataFrame(data=ls.coef_, index=x_nike.columns, columns=['Coefficient'])
ls_imp = ls_imp.sort_values(by=['Coefficient'], ascending=False)
print(ls_imp)
print("\n")

#Linear Regression
print(f'Most Important Feature for Linear Regression is:')
ls=LinearRegression()
ls.fit(x_nike,y_nike)
ls_imp = pd.DataFrame(data=ls.coef_, index=x_nike.columns, columns=['Coefficient'])
ls_imp = ls_imp.sort_values(by=['Coefficient'], ascending=False)
print(ls_imp)
print("\n")

#Decision Tree
print(f'Most Important Feature for Decision Tree is:')
ls=DecisionTreeRegressor(max_depth=5,max_leaf_nodes=50)
ls.fit(x_nike,y_nike)
ls_imp = pd.DataFrame(data=ls.feature_importances_, index=x_nike.columns, columns=['Importance'])
ls_imp = ls_imp.sort_values(by=['Importance'], ascending=False)
print(ls_imp)
print("\n")

#Random Forest
print(f'Most Important Feature for Random Forest is:')
ls=RandomForestRegressor(n_estimators=200)
ls.fit(x_nike,y_nike)
ls_imp = pd.DataFrame(data=ls.feature_importances_, index=x_nike.columns, columns=['Importance'])
ls_imp = ls_imp.sort_values(by=['Importance'], ascending=False)
print(ls_imp)

print(f'--------------------------------------------------------------------')


'''Graphs'''

print(f'Caculating the Graph for Decision Tree...(This will take less than 1 min)')
print(f'The Code for Decision Tree Graph is:')
print("\n")
ls=DecisionTreeRegressor(max_depth=5,max_leaf_nodes=14)
ls.fit(x_nike,y_nike)
export_graphviz(ls, out_file=sys.stdout, feature_names=x_nike.columns,  filled=True,impurity=True)
print("\n")
print(f'--------------------------------------------------------------------')

'''Prediction'''

print(f'Predicting the Nike Stock Price using Ridge Regression...(This will take less than 1 min)')
ls=Ridge(alpha=20,max_iter=1000000)
ls.fit(x_nike,y_nike)
print(f'The predicted price on 5/6/2021 is: {ls.predict([[133.37,134.45,132.59,34113.23,33904.89,34221.06,33904.89]])} Actual Price is: 133.49')
print(f'The predicted price on 5/10/2021 is: {ls.predict([[135.00,139.36,134.72,34777.76,34578.27,34811.39,34464.31]])} Actual Price is: 136.40')
print(f'The predicted price on 5/17/2021 is: {ls.predict([[134.34,136.68,134.05,34382.13,34050.86,34454.05,34050.86]])} Actual Price is: 136.41')
print(f'The predicted price on 5/21/2021 is: {ls.predict([[136.71,137.43,135.54,34060.66,34351.18,34408.99,34044.10]])} Actual Price is: 132.66')
print(f'--------------------------------------------------------------------')

'''Cleaning'''
print(f'All The Testing Are Completed!')

