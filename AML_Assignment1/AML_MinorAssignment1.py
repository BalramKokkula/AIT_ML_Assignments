import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix,accuracy_score
from sklearn.datasets import load_wine

wine = load_wine()

x = wine.data
y = wine.target
w = pd.DataFrame(wine.data)
w.columns = wine.feature_names
print(w.head())

corr_matrix = w.corr()
print(corr_matrix["alcohol"].sort_values(ascending=False))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33, random_state=32)

####Linear Regression######
lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
y_Pred = lr_reg.predict(x_test)
lr_cv_model = cross_val_score(lr_reg, x_train, y_train, cv=10)
y_pred = lr_reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_Pred))
r2 = r2_score(y_test,y_Pred)
lr_reg_score = lr_reg.score(x_test, y_test)

print ('rmse = ', rmse)
print ('r2 = ', r2)
print ("Linear Reg score: ", lr_reg_score)
print ("Linear Reg Cross Validation: ",np.mean(lr_cv_model))
print()

#### KNN Regression ######
knn_reg = KNeighborsRegressor(n_neighbors=7)
knn_reg.fit(x_train, y_train)
knn_reg_cv_model = cross_val_score(knn_reg, x_train, y_train, cv=10)
y_pred = knn_reg.predict(x_test)
knn_reg_score = knn_reg.score(x_test, y_test)

print ("KNN Reg score:", knn_reg_score)
print ("KNN Reg Cross Validation: ",np.mean(knn_reg_cv_model))
print()


#### KNN Model ######
knn_clf = KNeighborsClassifier(n_neighbors=7)
knn_clf.fit(x_train, y_train)
knn_score = knn_clf.score(x_test, y_test)
knn_clf_cv_model = cross_val_score(knn_clf, x_train, y_train, cv=10)
y_pred = knn_clf.predict(x_test)
knn_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))

print("KNN Model Confusion matrix: ")
print(knn_conf_matrix)
print ("KNN Model score: ", knn_score)
print ("KNN Model Cross Validation: ",np.mean(knn_clf_cv_model))
print()

#### SVC Model ######
svc_clf = SVC(kernel='linear')
svc_clf_cv_model = cross_val_score(svc_clf, x_train, y_train, cv=10)
svc_clf.fit(x_train, y_train)
y_pred = svc_clf.predict(x_test)
svc_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
print("SVC Model Confusion matrix:")
print(svc_conf_matrix)
svc_score = svc_clf.score(x_test, y_test)
print("SVC Model score: ", svc_score )
print ("SVC ModelCross Validation: ",np.mean(svc_clf_cv_model ))
print()

#### Ensemble RF Model ######
rf_clf = RandomForestClassifier()
rf_clf_cv_model  = cross_val_score(rf_clf, x_train, y_train, cv=10)
rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_test)
rf_score = rf_clf.score(x_test, y_test)
rf_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))

print("Ensemble RF Model Confusion matrix:")
print(rf_conf_matrix)
print ("Ensemble RF Model score:", rf_score )
print ("Ensemble RF Model Cross Validation:",np.mean(rf_clf_cv_model))
print()

#### Neural Network Model ######
nn_clf = MLPClassifier(hidden_layer_sizes=(200,100), max_iter=1000)
nn_clf_cv_model  = cross_val_score(nn_clf, x_train, y_train, cv=10)
nn_clf.fit(x_train, y_train)
y_pred = nn_clf.predict(x_test)
nn_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
nn_score = nn_clf.score(x_test, y_test)

print("Neural Network Model Confusion matrix:")
print(nn_conf_matrix)
print ("Neural Network Model score:", nn_score )
print ("Neural Network Model Cross Validation: ",np.mean(nn_clf_cv_model))
print()

data_output = {'Model':['Linear Reg Model:', 'KNN Reg Model:', 'KNN Model:', 'SVC Model:', 'Ensemble RF Model:', 'Neural Network Model:'], 'CV_value':[np.mean(lr_cv_model),np.mean(knn_reg_cv_model), np.mean(knn_clf_cv_model), np.mean(svc_clf_cv_model), np.mean(rf_clf_cv_model),np.mean(nn_clf_cv_model)],'Accuracy':[lr_reg_score,knn_reg_score,knn_score, svc_score, rf_score,nn_score]}
df = pd.DataFrame(data_output)
print("The accuracy of each model using cross validation and a conclusion which  outline the best model for the chosen dataset:")
print()
print(df.sort_values(by='Accuracy',ascending=False))



