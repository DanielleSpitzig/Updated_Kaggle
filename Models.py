# Load in necessary libraries
import pandas as pd
import numpy as np
import random
import math
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


random.seed(20602629)
#----- Classification Models -----
##Helper functions Base Models##
#Logistic regression returns AUC score
def log_reg(train, test, y_train, y_test, reg='l1', opt = "saga"):
    #If using elasticnet need a l1 ratio - setting to 0.5
    if reg == "elasticnet":
        logisticRegr = LogisticRegression(penalty = reg, solver = opt, max_iter = 100, l1_ratio = 0.5)
    else:
        logisticRegr = LogisticRegression(penalty = reg, solver = opt, max_iter = 100)
    logisticRegr.fit(train, y_train)
    prob_y = logisticRegr.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)

#SVM function returns AUC score
def linSVM(train, test, y_train, y_test, kern = 'rbf'):
    classifier = SVC(kernel = kern, probability = True, max_iter=100)
    classifier.fit(train, y_train)
    prob_y = classifier.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)

#Naive Bayes
def nbayes(train, test, y_train, y_test, laplace = 1e-9):
    classifier = GaussianNB(var_smoothing = laplace)
    classifier.fit(train, y_train)
    prob_y = classifier.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)

#KNN
def knn(train, test, y_train, y_test, num_n = 10):
    classifier = KNeighborsClassifier(n_neighbors=num_n)
    classifier.fit(train, y_train)
    prob_y = classifier.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)

#Decision Trees
def dtree(train, test, y_train, y_test, depth = 5):
    classifier = DecisionTreeClassifier(max_depth = depth)
    classifier.fit(train, y_train)
    prob_y = classifier.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)
  
## Helper functions for base model comparison##
#Run each model with dif parameters on input data - return best validated AUC and index
def comp_LR(X_train, X_test, y_train, y_test):
    #Logistic regression with L1, L2, and elasticnet
    AUC_LR = []
    AUC_LR.append(log_reg(X_train, X_test, y_train, y_test))
    AUC_LR.append(log_reg(X_train, X_test, y_train, y_test, reg="l2"))
    AUC_LR.append(log_reg(X_train, X_test, y_train, y_test, reg="elasticnet"))
    #L1 can use liblinear or saga
    AUC_LR.append(log_reg(X_train, X_test, y_train, y_test, opt="liblinear"))
    #Get max and position of max to find best logistic regression model
    best = np.max(AUC_LR)
    best_index = np.argmax(AUC_LR)
    print("Best LR model Index: {}, AUC: {}".format(best_index, best))

def comp_SVM(X_train, X_test, y_train, y_test):
    #SVM with different kernels
    AUC_SVM = []
    AUC_SVM.append(linSVM(X_train, X_test, y_train, y_test, kern="linear"))
    AUC_SVM.append(linSVM(X_train, X_test, y_train, y_test))
    AUC_SVM.append(linSVM(X_train, X_test, y_train, y_test, kern="poly"))
    AUC_SVM.append(linSVM(X_train, X_test, y_train, y_test, kern="sigmoidal"))
    #Get max and position of max to find best model
    best = np.max(AUC_SVM)
    best_index = np.argmax(AUC_SVM)
    print("Best SVM model Index: {}, AUC: {}".format(best_index, best))

def comp_NB(X_train, X_test, y_train, y_test):
    #Naive Bayes with different smoothing parameters
    AUC_NB = []
    AUC_NB.append(nbayes(X_train, X_test, y_train, y_test))
    AUC_NB.append(nbayes(X_train, X_test, y_train, y_test, laplace=1e-5))
    AUC_NB.append(nbayes(X_train, X_test, y_train, y_test, laplace=1e-1))
    AUC_NB.append(nbayes(X_train, X_test, y_train, y_test, laplace=1))
    #Get max and position of max to find best model
    best = np.max(AUC_NB)
    best_index = np.argmax(AUC_NB)
    print("Best NB model Index: {}, AUC: {}".format(best_index, best))

def comp_KNN(X_train, X_test, y_train, y_test):
    #KNN with different num of neighbours
    AUC_KNN = []
    AUC_KNN.append(knn(X_train, X_test, y_train, y_test, num_n=5))
    AUC_KNN.append(knn(X_train, X_test, y_train, y_test, num_n=10))
    AUC_KNN.append(knn(X_train, X_test, y_train, y_test, num_n=15))
    AUC_KNN.append(knn(X_train, X_test, y_train, y_test, num_n=20))
    #Get max and position of max to find best model
    best = np.max(AUC_KNN)
    best_index = np.argmax(AUC_KNN)
    print("Best KNN model Index: {}, AUC: {}".format(best_index, best))

def comp_dtree(X_train, X_test, y_train, y_test):
    #Decision trees with different depths
    AUC_DT = []
    AUC_DT.append(dtree(X_train, X_test, y_train, y_test))
    AUC_DT.append(dtree(X_train, X_test, y_train, y_test, depth=10))
    AUC_DT.append(dtree(X_train, X_test, y_train, y_test, depth=15))
    AUC_DT.append(dtree(X_train, X_test, y_train, y_test, depth=20))
    #Get max and position of max to find best model
    best = np.max(AUC_DT)
    best_index = np.argmax(AUC_DT)
    print("Best DTree model Index: {}, AUC: {}".format(best_index, best))

## Helper functions for Ensemble ##
#Random forest with num estimators and depth
def rforest(train, test, y_train, y_test, n_est = 500, depth=5):
    classifier = RandomForestClassifier(n_estimators = n_est, max_depth=depth)
    classifier.fit(train, y_train)
    prob_y = classifier.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)
#Adaboost with default decision trees and max_depth 1
def ada(train, test, y_train, y_test, n_est = 100):
    ada_class = AdaBoostClassifier(n_estimators=n_est)
    ada_class.fit(train, y_train)
    prob_y = ada_class.predict_proba(test)
    pred_y = prob_y[:,1]
    return roc_auc_score(y_test, pred_y)
#Stacking method with LR used to aggregrate final predictions
def stack(train, test, y_train, y_test, ests = 0):
    if ests == 0:
        ests = [('rf', RandomForestClassifier(n_estimators=100, max_depth = 10)),
        ('ada', AdaBoostClassifier(n_estimators=100)), 
        ('lg', LogisticRegression(solver='liblinear'))]
    stack_class = StackingClassifier(estimators = estimators)
    stack_class.fit(train, y_train)
    prob_y = stack_class.predict_proba(test)
    pred_y = prob_y[:,1]
    print("AUC of stacked: {}".format(roc_auc_score(y_test, pred_y)))

#Random Forest tuning
def comp_rf(X_train, X_test, y_train, y_test):
    #RF with different number of estimators - 10-200 and depth
    num_ests = list(range(10, 201, 10))
    depths = list(range(1, 10, 1))
    max_AUC = 0
    for ii in num_ests:
        for dep in depths:
            rf_AUC = rforest(X_train, X_test, y_train, y_test, n_est=ii, depth=dep)
            if rf_AUC > max_AUC:
                max_AUC = rf_AUC
                (depth, est) = (dep, ii)
    print("Best RF Num_Est: {}, depth:{},  AUC: {}".format(est, dep, max_AUC))

#Adaboost tuning
def comp_ada(X_train, X_test, y_train, y_test):
    #Adaboost with different number of estimators - 10-200
    max_AUC = 0
    max_i = 0
    num_ests = list(range(10, 201, 10))
    for ii in num_ests:
        ada_AUC = ada(X_train, X_test, y_train, y_test, n_est=ii)
        if ada_AUC > max_AUC:
            max_AUC = ada_AUC
            max_i = ii
    print("Best Adaboost Num_Est: {}, AUC: {}".format(num_ests.index(max_i), max_AUC))


#Run simple one layer with different optimization functions to find best
def NN_norm_opt(train, test, y_train, y_test, hidden = 10):
    opts = ["sgd", "adam", "adadelta", "rmsprop"]
    best = 0
    (_,dim) = train.shape
    for opt in opts:
        model = Sequential()
        model.add(Dense(hidden, input_dim=dim, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        #Compile model and fit it
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(train, y_train, epochs=5, verbose=0, validation_split=0.2, shuffle=True)
        pred_y = model.predict(test)
        NN_AUC = roc_auc_score(y_test, pred_y)
        if best < NN_AUC:
            best=NN_AUC
            best_opt=opt
    #At end return best AUC and best opts
    return(best, best_opt)

#Functions for 2 and 3 layer NN
def layer2(train, test, n=50, opt="sgd", epoch=50, batch=100):
    # create model
    (_,dim) = train.shape
    model = Sequential()
    model.add(Dense(n, input_dim=dim, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #Fit model and run cross-validation
    model.fit(train, test, epochs = epoch, verbose = 0, validation_split = 0.2, shuffle = True, batch_size = batch)
    return model

def layer3(train, test, n=50, opt="sgd", epoch=50, batch=100):
    # create model
    (_,dim) = train.shape
    model = Sequential()
    model.add(Dense(n, input_dim=dim, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #Fit model and run cross-validation
    model.fit(train, test, epochs = epoch, verbose = 0, validation_split = 0.2, shuffle = True, batch_size = batch)
    return model

#NN using above functions with built-in grid search for batch and num neurons
def NN_valid(train, test, y_train, y_test, opt="sgd"):
    #Check 30 - 150 for num neurons by layer and batches 10 - 25
    num_neur = list(range(30, 180, 30))
    batches = [10, 15, 20, 25]
    best_2 = 0
    best_3 = 0
    for batch in batches:
        for n in num_neur:
            #Using batch and neurons get 2 and 3 layer
            model1 = layer2(train, y_train, n=n, batch=batch, opt=opt)
            model2 = layer3(train, y_train, n=n, batch=batch, opt=opt)
            pred_2 = model1.predict(test, batch_size = batch)
            pred_3 = model2.predict(test, batch_size = batch)
            lay2_AUC = roc_auc_score(y_test, pred_2)
            lay3_AUC = roc_auc_score(y_test, pred_3)
            #Update best for each layer with the neuron and batch
            if lay2_AUC > best_2:
                best_2 = lay2_AUC
                (neur_2, batch_2) = (n, batch)
            if lay3_AUC > best_3:
                best_3 = lay3_AUC
                (neur_3, batch_3) = (n, batch)
    #At end check which best is better and return
    if best_2 >= best_3:
      print("Best NN Batch: {}, Neurons: {}, Layers: {}, AUC: {}".format(batch_2, neur_2, "2", best_2))
    else:
      print("Best NN Batch: {}, Neurons: {}, Layers: {}, AUC: {}".format(batch_3, neur_3, "3", best_3))

## Read in data and get train test splits##
train = pd.read_csv('cleaned_train.csv')
test = pd.read_csv('cleaned_test.csv')
train_labels = pd.read_csv('Train_Labels.csv')
test_id = pd.read_csv('test_id.csv')

train.drop('Unnamed: 0', axis = 1, inplace=True)
test.drop('Unnamed: 0', axis = 1, inplace=True)
train_labels.drop('Unnamed: 0', axis = 1, inplace=True)
test_id.drop('Unnamed: 0', axis = 1, inplace=True)

#Apply normalization before split so it carries through
col_names = list(train.columns)

#Apply scaling before split so it carries through
mm_scaler = MinMaxScaler()
train_mm = mm_scaler.fit_transform(train)
train = pd.DataFrame(train_mm, columns=col_names)

test_mm = mm_scaler.fit_transform(test)
test = pd.DataFrame(test_mm, columns=col_names)

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.33)

## Run base models on full data ##
comp_LR(X_train, X_test, y_train, y_test)
comp_SVM(X_train, X_test, y_train, y_test)
comp_NB(X_train, X_test, y_train, y_test)
comp_KNN(X_train, X_test, y_train, y_test)
comp_dtree(X_train, X_test, y_train, y_test)

#Since data so large won't run more complex models on it

## Reducing dimensions - used random forest to get variable importance ##
#Don't use above split here - not prediction just for importance over all vars
clf = RandomForestClassifier(n_estimators = 1000, max_depth=10, random_state=0)
clf.fit(train, train_labels)

feature_importances = list(clf.feature_importances_)
columns = list(train.columns)

## Change n to modify how many features
#Found 100 was good with trade-off of complexity and accuracy
n = 100 #200, 300 overfit, 50 underfit, 100, 150 similar
top_n_idx = np.argsort(feature_importances)[-n:]
top_n_values = [feature_importances[i] for i in top_n_idx]

important_features = []
for i in list(top_n_idx):
    important_features.append(columns[i])

#Use above row splits and subset at columns
rf_train = X_train[important_features]
rf_test = X_test[important_features]

## Use reduced dataset on models ##
# Run base models - will print results automatically
comp_LR(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())
comp_SVM(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())
comp_NB(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())
comp_KNN(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())
comp_dtree(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())

#Now let's run on some ensemble methods
comp_rf(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())
comp_ada(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel())

#Tune NN and run
NN_norm_opt(rf_train, rf_test, y_train, y_test) #Best was rmsprop
NN_valid(rf_train, rf_test, y_train, y_test, opt="rmsprop")

#Do two stacked models with subset data - one with two best models and one with three best models
est_3 = [('rf', RandomForestClassifier(n_estimators=200, max_depth = 9)),
         ('ada', AdaBoostClassifier(n_estimators=17)), 
         ('lg', LogisticRegression(solver='saga', penalty="l1", max_iter=100))]
est_2 = [('rf', RandomForestClassifier(n_estimators=100, max_depth = 10)),
         ('ada', AdaBoostClassifier(n_estimators=100))]

stack(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel(), ests=est_3)
stack(rf_train, rf_test, y_train.values.ravel(), y_test.values.ravel(), ests=est_2)

## PCA to reduce dimenstions then run same models ##
pca = PCA(0.95, svd_solver="full")
#Want to use scaled data here
pca.fit(train)
p_train = pca.transform(train)
p_test = pca.transform(test)

pca_train, pca_test, ytrain, ytest = train_test_split(p_train, train_labels, test_size=0.33)

## Use reduced PCA dataset on models ##
# Run base models - will print results automatically
comp_LR(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())
comp_SVM(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())
comp_NB(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())
comp_KNN(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())
comp_dtree(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())

#Now let's run on some ensemble methods
comp_rf(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())
comp_ada(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel())

#Tune NN and run
NN_norm_opt(pca_train, pca_test, ytrain, ytest) #Best was rmsprop
NN_valid(pca_train, pca_test, ytrain, ytest, opt="rmsprop")

#Do two stacked models with subset data - one with two best models and one with three best models
est_3 = [('rf', RandomForestClassifier(n_estimators=200, max_depth = 9)),
         ('ada', AdaBoostClassifier(n_estimators=17)), 
         ('lg', LogisticRegression(solver='saga', penalty="l1", max_iter=100))]
est_2 = [('rf', RandomForestClassifier(n_estimators=100, max_depth = 10)),
         ('ada', AdaBoostClassifier(n_estimators=100))]

stack(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel(), ests=est_3)
stack(pca_train, pca_test, ytrain.values.ravel(), ytest.values.ravel(), ests=est_2)

## Use top 2 best models to get probability of classification to submit ##
#Want probability because using AUC to compute
#Stacked with reduced variable importance
top1 = StackingClassifier(estimators = est_3)
top1.fit(train[important_features], train_labels.values.ravel())
prob_top1 = top1.predict_proba(test[important_features])
pred_top1 = prob_top1[:,1]
pred_top1 = pd.DataFrame(pred_top1)
pred_top1['id'] = test_id
pred_top1.to_csv("top_prediction.csv")

top2 =  AdaBoostClassifier(n_estimators=17)
top2.fit(train[important_features], train_labels.values.ravel())
prob_top2 = top2.predict_proba(test[important_features])
pred_top2 = prob_top2[:,1]
pred_top2 = pd.DataFrame(pred_top2)
pred_top2['id'] = test_id
pred_top2.to_csv("top2_prediction.csv")
