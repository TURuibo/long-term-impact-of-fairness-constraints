import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tempeh.configurations import datasets
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer
from copy import deepcopy
from eq_odds import *

from fairlearn.postprocessing._constants import (LABEL_KEY, SCORE_KEY, SENSITIVE_FEATURE_KEY, OUTPUT_SEPARATOR,DEMOGRAPHIC_PARITY, EQUALIZED_ODDS)
from fairlearn.postprocessing._threshold_optimizer import _reformat_data_into_dict

def load_compas_X_y_s(datasets):
    # Load compas data X, ground-true label y, and the sensitive attribute
    compas_dataset = datasets['compas']()

    X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
    y_train, y_test = compas_dataset.get_y(format=pd.Series)
    sensitive_features_train, sensitive_features_test = compas_dataset.get_sensitive_features('race', 
                                                                                          format=pd.Series)
    # Combine training set and testing set
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    sensitive_features = pd.concat([sensitive_features_train, sensitive_features_test], ignore_index=True)
    
    return X,y,sensitive_features

def get_lable_group_index(X,y,sensitive_features):
    # Split dataset into 4 sub-groups regarding different ethnicities and qualification states, return the indices
    # Lable 0: not qualified ; 1: qualified 
    # group 0: african-american ; 1: causasian
    index_group0 = sensitive_features[sensitive_features == "African-American"].index
    index_group1 = sensitive_features[sensitive_features == "Caucasian"].index

    index_label0 = y[y == 0.0].index
    index_label1 = y[y == 1.0].index

    index_label0_group0 = set(index_group0).intersection(set(index_label0))
    index_label0_group1 = set(index_group1).intersection(set(index_label0))
    index_label1_group0 = set(index_group0).intersection(set(index_label1))
    index_label1_group1 = set(index_group1).intersection(set(index_label1))

    return list(index_label0_group0),list(index_label0_group1),list(index_label1_group0),list(index_label1_group1)

def transition(alpha,tpr,fpr,T, group):
    g0 = T[0,0,group]*(1-fpr) + T[0,1,group]*fpr
    g1 = T[1,0,group]*(1-tpr) + T[1,1,group]*tpr
    return alpha*g1 + (1-alpha)*g0
#     g0 = T_label0_pred0*(1-fpr) + T_label0_pred1*fpr
#     g1 = T_label1_pred0*(1-tpr) + T_label1_pred1*tpr
#     return alpha*g1 + (1-alpha)*g0

def samplePath(alpha0,alpha1,P0,T,X,y,sensitive_features,subgroups_indices):
    alpha_group0_UN,alpha_group0_EqOpt,alpha_group0_DP = [alpha0],[alpha0],[alpha0]
    alpha_group1_UN,alpha_group1_EqOpt,alpha_group1_DP = [alpha1],[alpha1],[alpha1]
    t = 0
    horizon = 10
    while t < horizon:
        t += 1
        ratio = P0*(1-alpha0),(1-P0)*(1-alpha1),P0*alpha0,(1-P0)*alpha1  # r_label0_group0,r_label0_group1,r_label1_group0,r_label1_group1 
        # sample individuals from 4 sub-groups
        X_train,y_train,sensitive_features_train = dataSelection(X,y,sensitive_features,subgroups_indices,ratio)
        # train (fair) classifiers
        pr_un,acc_un,tpr_un,fpr_un,pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt,pr_dp,acc_dp,tpr_dp,fpr_dp \
        = find_Classifier(X_train,y_train,sensitive_features_train)
        

        # transitions
        alpha_group1_UN.append(transition(alpha_group1_UN[-1],
                                          tpr_un["Caucasian"],fpr_un["Caucasian"],
                                          T, group = 1))
        
        alpha_group1_EqOpt.append(transition(alpha_group1_EqOpt[-1],
                                             tpr_eqopt["Caucasian"],fpr_eqopt["Caucasian"],
                                             T, group = 1))
        
        alpha_group1_DP.append(transition(alpha_group1_DP[-1],
                                          tpr_dp["Caucasian"],fpr_dp["Caucasian"],
                                          T, group = 1))
        
        alpha_group0_UN.append(transition(alpha_group0_UN[-1],
                                          tpr_un["African-American"],fpr_un["African-American"],
                                          T, group = 0))
        
        alpha_group0_EqOpt.append(transition(alpha_group0_EqOpt[-1],
                                             tpr_eqopt["African-American"],fpr_eqopt["African-American"],
                                             T, group = 0))
        
        alpha_group0_DP.append(transition(alpha_group0_DP[-1],
                                          tpr_dp["African-American"],fpr_dp["African-American"],
                                          T, group = 0))
        
    return alpha_group1_UN,alpha_group1_EqOpt,alpha_group1_DP,alpha_group0_UN,alpha_group0_EqOpt,alpha_group0_DP

def eva_classifier(X_train,y_train,sensitive_features_train):
    # (Fair) optimal classifier
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    sensitive_features_train = sensitive_features_train.reset_index(drop = True)


    # ******** UN ********
    estimator = LogisticRegression(solver='liblinear')
    estimator_wrapper = LogisticRegressionAsRegression(estimator).fit(X_train, y_train)
    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)
    pr_un,acc_un,tpr_un,fpr_un = find_proportions(X_train, sensitive_features_train, predictions_train, y_train)
 
    return pr_un,acc_un,tpr_un,fpr_un

def eva_classifier_dp(X_train,y_train,sensitive_features_train):
    # (Fair) optimal classifier
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    sensitive_features_train = sensitive_features_train.reset_index(drop = True)


    # ******** UN ********
    estimator = LogisticRegression(solver='liblinear')
    estimator_wrapper = LogisticRegressionAsRegression(estimator).fit(X_train, y_train)
    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)

    # ******** DP ********
    postprocessed_predictor_DP = ThresholdOptimizer(
        estimator=estimator_wrapper,
        constraints="demographic_parity",
        prefit=True)
    postprocessed_predictor_DP.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    fairness_aware_predictions_DP_train = postprocessed_predictor_DP.predict(X_train, sensitive_features=sensitive_features_train)
    pr_dp,acc_dp,tpr_dp,fpr_dp = find_proportions(X_train, sensitive_features_train, fairness_aware_predictions_DP_train, y_train)
    
    return pr_dp,acc_dp,tpr_dp,fpr_dp

def eva_classifier_eqopt(X_train,y_train,sensitive_features_train):
    # (Fair) optimal classifier
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    sensitive_features_train = sensitive_features_train.reset_index(drop = True)


    # ******** UN ********
    estimator = LogisticRegression(solver='liblinear')
    estimator_wrapper = LogisticRegressionAsRegression(estimator).fit(X_train, y_train)
    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)

    # ******** EqOpt ************

    data_grouped_by_sensitive_feature = _reformat_and_group_data(sensitive_features_train, y_train.astype(int),predictions_train.astype(int))

    group0 = data_grouped_by_sensitive_feature.get_group("African-American")
    group1 = data_grouped_by_sensitive_feature.get_group("Caucasian")
    group0_model = Model(group0['score'].to_numpy(), group0['label'].to_numpy())
    group1_model = Model(group1['score'].to_numpy(), group1['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.eq_odds(group0_model, group1_model)

    # Apply the mixing rates to the test models
    eqopt_group0, eqopt_group1 = Model.eq_odds(group0_model,group1_model,mix_rates)
    pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt = {},{},{},{}
    pr_eqopt["African-American"],acc_eqopt["African-American"],tpr_eqopt["African-American"],fpr_eqopt["African-American"] = eqopt_group0.output()
    pr_eqopt["Caucasian"],acc_eqopt["Caucasian"],tpr_eqopt["Caucasian"],fpr_eqopt["Caucasian"] = eqopt_group1.output()

    return pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt

def eva_classifier_eo(X_train,y_train,sensitive_features_train):
    # (Fair) optimal classifier
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    sensitive_features_train = sensitive_features_train.reset_index(drop = True)


    # ******** UN ********
    estimator = LogisticRegression(solver='liblinear')
    estimator_wrapper = LogisticRegressionAsRegression(estimator).fit(X_train, y_train)
    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)

     # ********EO********
    postprocessed_predictor_EO = ThresholdOptimizer(
        estimator=estimator_wrapper,
        constraints="equalized_odds",
        prefit=True)
    postprocessed_predictor_EO.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    fairness_aware_predictions_EO_train = postprocessed_predictor_EO.predict(X_train, sensitive_features=sensitive_features_train)
    pr,acc,tpr,fpr = find_proportions(X_train, sensitive_features_train, fairness_aware_predictions_EO_train, y_train)

    return pr,acc,tpr,fpr
    

# ********* The following part is not changed *********
def _reformat_and_group_data(sensitive_features, labels, scores,sensitive_feature_names=None):
    """Reformats the data into a new pandas.DataFrame and group by sensitive feature values.
    The data are provided as three arguments (`sensitive_features`, `labels`, `scores`) and
    the new  DataFrame is grouped by sensitive feature values so that subsequently each group
    can be handled separately.
    :param sensitive_features: the sensitive features based on which the grouping is determined;
        currently only a single sensitive feature is supported
    :type sensitive_features: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param labels: the training labels
    :type labels: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param scores: the output from the unconstrained predictor used for training the mitigator
    :type scores: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param sensitive_feature_names: list of names for the sensitive features in case they were
        not implicitly provided (e.g. if `sensitive_features` is of type DataFrame); default
        None
    :type sensitive_feature_names: list of strings
    :return: the training data for the mitigator, grouped by sensitive feature value
    :rtype: pandas.DataFrameGroupBy
    """
    data_dict = {}

    # TODO: extend to multiple columns for additional group data
    # and name columns after original column names if possible
    # or store the original column names
    sensitive_feature_name = SENSITIVE_FEATURE_KEY
    if sensitive_feature_names is not None:
        if sensitive_feature_name in [SCORE_KEY, LABEL_KEY]:
            raise ValueError(SENSITIVE_FEATURE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE)
        sensitive_feature_name = sensitive_feature_names[0]

    _reformat_data_into_dict(sensitive_feature_name, data_dict, sensitive_features)
    _reformat_data_into_dict(SCORE_KEY, data_dict, scores)
    _reformat_data_into_dict(LABEL_KEY, data_dict, labels)

    return pd.DataFrame(data_dict).groupby(sensitive_feature_name)

def find_proportions(X, sensitive_features, y_pred, y=None):
        
    indices = {}
    positive_indices = {}
    negative_indices = {}
    recidivism_count = {}
    pr = {}
    acc = {}
    tpr = {}
    fpr = {}

    groups = np.unique(sensitive_features.values)
    
    for index, group in enumerate(groups):
   
        indices[group] = sensitive_features.index[sensitive_features == group]
        recidivism_count[group] = sum(y_pred[indices[group]])
        pr[group] = recidivism_count[group]/len(indices[group])
        
        #print('PR:\t%.3f' % pr[group])
        
        if y is not None:
            positive_indices[group] = sensitive_features.index[(sensitive_features == group) & (y == 1)]
            negative_indices[group] = sensitive_features.index[(sensitive_features == group) & (y == 0)]
            prob_1 = sum(y_pred[positive_indices[group]])/len(positive_indices[group])
            prob_0 = sum(y_pred[negative_indices[group]])/len(negative_indices[group])
            acc[group] = 1-((1-prob_1)*len(positive_indices[group]) + prob_0*len(negative_indices[group]))/len(indices[group])
            tpr[group] = prob_1
            fpr[group] = prob_0
            #print('TPR:\t%.3f' % prob_1)
            #print('FPR:\t%.3f' % prob_0)
            #print('accuracy:\t%.3f' % acc)

    return pr,acc,tpr,fpr

class LogisticRegressionAsRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, logistic_regression_estimator):
        self.logistic_regression_estimator = logistic_regression_estimator
    
    def fit(self, X, y):
        try:
            check_is_fitted(self.logistic_regression_estimator)
            self.logistic_regression_estimator_ = self.logistic_regression_estimator
        except NotFittedError:
            self.logistic_regression_estimator_ = clone(
                self.logistic_regression_estimator).fit(X, y)
        return self
    
    def predict(self, X):
        # use predict_proba to get real values instead of 0/1, select only prob for 1
        scores = self.logistic_regression_estimator_.predict_proba(X)[:,1]
        return scores

def find_Classifier(X_train,y_train,sensitive_features_train):
    # (Fair) optimal classifier
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    sensitive_features_train = sensitive_features_train.reset_index(drop = True)


    # ******** UN ********
    estimator = LogisticRegression(solver='liblinear')
    estimator_wrapper = LogisticRegressionAsRegression(estimator).fit(X_train, y_train)
    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)

    pr_un,acc_un,tpr_un,fpr_un = find_proportions(X_train, sensitive_features_train, predictions_train, y_train)
    '''
    # ********EO********
    postprocessed_predictor_EO = ThresholdOptimizer(
        estimator=estimator_wrapper,
        constraints="equalized_odds",
        prefit=True)
    postprocessed_predictor_EO.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    fairness_aware_predictions_EO_train = postprocessed_predictor_EO.predict(X_train, sensitive_features=sensitive_features_train)
    pr,acc,tpr,fpr = find_proportions(X_train, sensitive_features_train, fairness_aware_predictions_EO_train, y_train)
    '''

    # ******** DP ********
    postprocessed_predictor_DP = ThresholdOptimizer(
        estimator=estimator_wrapper,
        constraints="demographic_parity",
        prefit=True)
    postprocessed_predictor_DP.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    fairness_aware_predictions_DP_train = postprocessed_predictor_DP.predict(X_train, sensitive_features=sensitive_features_train)
    pr_dp,acc_dp,tpr_dp,fpr_dp = find_proportions(X_train, sensitive_features_train, fairness_aware_predictions_DP_train, y_train)

    # ******** EqOpt ************


    data_grouped_by_sensitive_feature = _reformat_and_group_data(sensitive_features_train, y_train.astype(int),predictions_train.astype(int))

    group0 = data_grouped_by_sensitive_feature.get_group("African-American")
    group1 = data_grouped_by_sensitive_feature.get_group("Caucasian")
    group0_model = Model(group0['score'].to_numpy(), group0['label'].to_numpy())
    group1_model = Model(group1['score'].to_numpy(), group1['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.eq_odds(group0_model, group1_model)

    # Apply the mixing rates to the test models
    eqopt_group0, eqopt_group1 = Model.eq_odds(group0_model,group1_model,mix_rates)
    pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt = {},{},{},{}
    pr_eqopt["African-American"],acc_eqopt["African-American"],tpr_eqopt["African-American"],fpr_eqopt["African-American"] = eqopt_group0.output()
    pr_eqopt["Caucasian"],acc_eqopt["Caucasian"],tpr_eqopt["Caucasian"],fpr_eqopt["Caucasian"] = eqopt_group1.output()

    return pr_un,acc_un,tpr_un,fpr_un,pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt,pr_dp,acc_dp,tpr_dp,fpr_dp
# ---------------------------------------------------------------------------
# ZeroDivisionError                         Traceback (most recent call last)
# <ipython-input-8-32b112974a13> in <module>
#      20         # train (fair) classifiers
#      21         pr_un,acc_un,tpr_un,fpr_un,pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt,pr_dp,acc_dp,tpr_dp,fpr_dp \
# ---> 22         = find_Classifier(X_train,y_train,sensitive_features_train)
#      23 
#      24         # transitions

# ~/work/long-term-impact-of-fairness-constraints/util.py in find_Classifier(X_train, y_train, sensitive_features_train)
#     211     predictions_train = estimator.predict(X_train)
#     212 
# --> 213     pr_un,acc_un,tpr_un,fpr_un = find_proportions(X_train, sensitive_features_train, predictions_train, y_train)
#     214     '''
#     215     # ********EO********

# ~/work/long-term-impact-of-fairness-constraints/util.py in find_proportions(X, sensitive_features, y_pred, y)
#     169             negative_indices[group] = sensitive_features.index[(sensitive_features == group) & (y == 0)]
#     170             prob_1 = sum(y_pred[positive_indices[group]])/len(positive_indices[group])
# --> 171             prob_0 = sum(y_pred[negative_indices[group]])/len(negative_indices[group])
#     172             acc[group] = 1-((1-prob_1)*len(positive_indices[group]) + prob_0*len(negative_indices[group]))/len(indices[group])
#     173             tpr[group] = prob_1

# ZeroDivisionError: division by zero



def dataSelection(X,y,sensitive_features,indices_sub,ratio):
    index_label0_group0,index_label0_group1,index_label1_group0,index_label1_group1 = indices_sub
    r_label0_group0,r_label0_group1,r_label1_group0,r_label1_group1 = ratio
    N = min(len(index_label0_group0)/r_label0_group0,len(index_label0_group1)/r_label0_group1,len(index_label1_group0)/r_label1_group0,len(index_label1_group1)/r_label1_group1)
    
    I_label0_group0 = random.sample(index_label0_group0,int(N*r_label0_group0))
    I_label0_group1 = random.sample(index_label0_group1,int(N*r_label0_group1))
    I_label1_group0 = random.sample(index_label1_group0,int(N*r_label1_group0))
    I_label1_group1 = random.sample(index_label1_group1,int(N*r_label1_group1))
    
    X_train = X.iloc[I_label0_group0+I_label0_group1+I_label1_group0+I_label1_group1,:]
    #X_train = X.iloc[I_label0_group0+I_label0_group1+I_label1_group0+I_label1_group1]
    
    y_train = y.iloc[I_label0_group0+I_label0_group1+I_label1_group0+I_label1_group1]

    sensitive_features_train = sensitive_features.iloc[I_label0_group0+I_label0_group1+I_label1_group0+I_label1_group1]


    return X_train,y_train,sensitive_features_train


def balanceEqn(alpha,tpr,fpr,T_label0_pred0,T_label0_pred1,T_label1_pred0,T_label1_pred1):
    g0 = T_label0_pred0*(1-fpr) + T_label0_pred1*fpr
    g1 = T_label1_pred0*(1-tpr) + T_label1_pred1*tpr
    return np.abs((1/alpha - 1) - (1-g1)/g0)


def balance(P0,alpha0,alpha1,X,y,sensitive_features,subgroups_indices):

    T_label0_pred0_group0 = 0.1
    T_label0_pred1_group0 = 0.5
    T_label1_pred0_group0 = 0.5
    T_label1_pred1_group0 = 0.7

    T_label0_pred0_group1 = 0.3 # "Caucasian"
    T_label0_pred1_group1 = 0.5
    T_label1_pred0_group1 = 0.5
    T_label1_pred1_group1 = 0.9

    ratio = P0*(1-alpha0),(1-P0)*(1-alpha1),P0*alpha0,(1-P0)*alpha1  # r_label0_group0,r_label0_group1,r_label1_group0,r_label1_group1 
    
    # sample individuals from 4 sub-groups
    X_train,y_train,sensitive_features_train = dataSelection(X,y,sensitive_features,subgroups_indices,ratio)
    # train (fair) classifiers
    pr_un,acc_un,tpr_un,fpr_un,pr_eqopt,acc_eqopt,tpr_eqopt,fpr_eqopt,pr_dp,acc_dp,tpr_dp,fpr_dp \
    = find_Classifier(X_train,y_train,sensitive_features_train)

    group1_UN = balanceEqn(alpha1,tpr_un["Caucasian"],fpr_un["Caucasian"],T_label0_pred0_group1,T_label0_pred1_group1,T_label1_pred0_group1,T_label1_pred1_group1)
    group1_EqOpt = balanceEqn(alpha1,tpr_eqopt["Caucasian"],fpr_eqopt["Caucasian"],T_label0_pred0_group1,T_label0_pred1_group1,T_label1_pred0_group1,T_label1_pred1_group1)
    group1_DP = balanceEqn(alpha1,tpr_dp["Caucasian"],fpr_dp["Caucasian"],T_label0_pred0_group1,T_label0_pred1_group1,T_label1_pred0_group1,T_label1_pred1_group1)
    
    group0_UN = balanceEqn(alpha0,tpr_un["African-American"],fpr_un["African-American"],T_label0_pred0_group0,T_label0_pred1_group0,T_label1_pred0_group0,T_label1_pred1_group0)
    group0_EqOpt = balanceEqn(alpha0,tpr_eqopt["African-American"],fpr_eqopt["African-American"],T_label0_pred0_group0,T_label0_pred1_group0,T_label1_pred0_group0,T_label1_pred1_group0)
    group0_DP = balanceEqn(alpha0,tpr_dp["African-American"],fpr_dp["African-American"],T_label0_pred0_group0,T_label0_pred1_group0,T_label1_pred0_group0,T_label1_pred1_group0)
    

    return group1_UN,group1_EqOpt,group1_DP,group0_UN,group0_EqOpt,group0_DP