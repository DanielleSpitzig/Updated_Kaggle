#Load in necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ----- Imputation and One Hot Encoding-----

## Helper functions ##
#in country column 
def set_country_codes(df):
    country_cols = ['v17', 'v25', 'v78', 'v161']
    for col in country_cols:
        df[col] = np.where(df[col] == '66', df['cntry'], df[col])
        df[col] = np.where(df[col].isin(['2', '3', '4', '6', '77', '88', '99']), 'AAA', df[col])
        df[col] = df[col].astype('category')
#one_hot categorical variables
def onehot_cat(train, test, vars):
  enc = OneHotEncoder(handle_unknown = "ignore")
  X = train[vars]
  Y = test[vars]
  enc.fit(X)
  df_train = enc.transform(X).toarray()
  df_test = enc.transform(Y).toarray()
  return df_train, df_test
#Just used to check whether all variables are in data still before we try anything
def sanity_check(vars, columns):
  #Copy list to iterate so don't remove item from itering list
  to_remove = []
  for var in vars:
    if var not in columns:
      to_remove.append(var)

  for var in to_remove:
    vars.remove(var)
  return vars
  

## Read in data ##
#Read in features separately and append at end
train_e_feats = pd.read_csv('F_train.csv')
test_e_feats = pd.read_csv('F_test.csv')
#Read in stripped vars
train_data = pd.read_csv('strip_train.csv')
test_data = pd.read_csv('strip_test.csv')
#When saved had an appended column 'Unnamed: 0' - remove
train_e_feats.drop('Unnamed: 0', axis = 1, inplace=True)
test_e_feats.drop('Unnamed: 0', axis = 1, inplace=True)
train_data.drop('Unnamed: 0', axis = 1, inplace=True)
test_data.drop('Unnamed: 0', axis = 1, inplace=True)

#Replace missing values with nans - treating all missing as the same
train_data.replace(to_replace=['.a', '.b', '.c', '.d', '.'], value= np.nan, inplace=True)
test_data.replace(to_replace=['.a', '.b', '.c', '.d', '.'], value= np.nan, inplace=True)
    

## Imputations lists - make separate for each type of imputation performed ##
# Ordinal where fill NaNs with Mode
ordinal_vars = ['v1', 'v2', 'v13', 'v19', 'v74', 'v76', 'v79', 'v80', 'v81',
                'v82', 'v84', 'v98', 'v99', 'v100', 'v101', 'v104', 'v109',
                'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117',
                'v118', 'v119', 'v120', 'v121', 'v122', 'v123', 'v124', 'v125',
                'v126', 'v127', 'v128', 'v129',  'v130', 'v131', 'v135', 'v136',
                'v137', 'v138', 'v139', 'v140', 'v141', 'v142', 'v143', 'v144',
                'v145', 'v146', 'v147', 'v148', 'v149', 'v177', 'v181', 'v182',
                'v186', 'v219', 'v220', 'v222', 'v228', 'v229', 'v230', 'v240', 
                'v253']

ordinal_vars = sanity_check(ordinal_vars, train_data.columns)

# Ordinal where fill NaNs with middle value (in this case 5) - basically "middle ground" of these Qs
ordinal_vars2 = ['v156', 'v178', 'v179', 'v180', 'v183', 'v184', 'v189',
                'v223', 'v224', 'v225', 'v226', 'v227', 'v232', 'v233',
                'v234', 'v235', 'v236', 'v237', 'v238', 'v249']

ordinal_vars2 = sanity_check(ordinal_vars2, train_data.columns)

# Categorical where NaNs are treated as its own category - "6" will be new category
cat_vars = ['v4', 'v5', 'v11', 'v22', 'v23', 'v35', 'v56', 'v57',
            'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v65','v66',
            'v67', 'v68', 'v69', 'v70', 'v71', 'v75', 'v77', 'v83',
            'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93',
            'v94', 'v95', 'v96', 'v97', 'v108', 'v150', 'v151', 'v153',
            'v158', 'v160', 'v164', 'v165', 'v167', 'v169', 'v170', 'v173',
            'v174', 'v188', 'v190', 'v191', 'v192', 'v193', 'v194', 'v195',
            'v196', 'v197', 'v198', 'v199', 'v200', 'v201', 'v202', 'v203',
            'v204', 'v205','v206', 'v207', 'v208', 'v209', 'v210', 'v211',
            'v212', 'v213', 'v214', 'v215', 'v216', 'v231', 'v239', 'v241',
            'v243', 'v255']

cat_vars = sanity_check(cat_vars, train_data.columns)

# Categorical where NaNs are filled with mode
cat_vars2 = ['v72', 'v73', 'v102', 'v103', 'v105', 'v133', 
            'v162', 'v163', 'v248']

cat_vars2 = sanity_check(cat_vars2, train_data.columns)

# Binary where NaNs are filled with mode
bin_vars = ['v6', 'v7', 'v8', 'v9', 'v10', 'v12', 'v14', 'v18', 'v21', 'v24', 
          'v43', 'v53', 'v85', 'v152', 'v157', 'v166', 'v172', 'v187', 'v221', 
          'v242', 'v254', 'v256', 'v257']

bin_vars = sanity_check(bin_vars, train_data.columns)

## Fill NaNs with median
cont_vars = ['v3', 'v132', 'v250', 'v251', 'v252', 'v258']

cont_vars = sanity_check(cont_vars, train_data.columns)

#Variables with no NANs - geographical values
no_nans_cat = ['v17', 'v20', 'v25', 'v78', 'v134', 'v154', 'v155', 
              'v161', 'v185', 'cntry']

no_nans_cat = sanity_check(no_nans_cat, train_data.columns)

# Categorical of relationship status - took missing as value 6 - none of the above
cat_rel = ['v159']
#Similar as above with binary variable on living with partner or not - took missing as 2 - not living
bin_liv = ['v171']
# Education level - fill NaNs with 0 as assuming no answer here meant no formal education
cont_edu = ['v64']


## Have all lists, time to format correctly those columns correctly ##
# Set proper column type for training data
train_data[ordinal_vars] = train_data[ordinal_vars].astype('float64')
train_data[ordinal_vars2] = train_data[ordinal_vars2].astype('float64')
train_data[bin_vars] = train_data[bin_vars].astype('float64')
train_data[cat_rel] = train_data[cat_rel].astype('str')
train_data[cat_vars] = train_data[cat_vars].astype('str')
train_data[cat_vars2] = train_data[cat_vars2].astype('str')
train_data[bin_liv] = train_data[bin_liv].astype('float64')
train_data[cont_vars] = train_data[cont_vars].astype('float64')
train_data[cont_edu] = train_data[cont_edu].astype('float64')
train_data[no_nans_cat] = train_data[no_nans_cat].astype('str')

# Set proper colmn types for test data
test_data[ordinal_vars] = test_data[ordinal_vars].astype('float64')
test_data[ordinal_vars2] = test_data[ordinal_vars2].astype('float64')
test_data[bin_vars] = test_data[bin_vars].astype('float64')
test_data[cat_rel] = test_data[cat_rel].astype('str')
test_data[cat_vars] = test_data[cat_vars].astype('str')
test_data[cat_vars2] = test_data[cat_vars2].astype('str')
test_data[bin_liv] = test_data[bin_liv].astype('float64')
test_data[cont_vars] = test_data[cont_vars].astype('float64')
test_data[cont_edu] = test_data[cont_edu].astype('float64')
test_data[no_nans_cat] = test_data[no_nans_cat].astype('str')

#Now time to change the country codes by above function
set_country_codes(train_data)
set_country_codes(test_data)

#Rename dataframes and remove id - keep test_id for later
train_features = train_data.drop(labels=['id'], axis = 1)
test_features = test_data.drop(labels=['id'], axis = 1)
test_id = pd.DataFrame(test_data['id'])

## Imputations Using SkLearn SimpleImputer function##
#Fill ordinals 1 with mode
imp_ord_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_ord_mode.fit(train_features[ordinal_vars])
train_features[ordinal_vars] = imp_ord_mode.transform(train_features[ordinal_vars])
test_features[ordinal_vars] = imp_ord_mode.transform(test_features[ordinal_vars])

# Filling ordinals 2 with constant value 5 - is middle value of possible values
imp_ord_5 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 5.0)
imp_ord_5.fit(train_features[ordinal_vars2])
train_features[ordinal_vars2] = imp_ord_5.transform(train_features[ordinal_vars2])
test_features[ordinal_vars2] = imp_ord_5.transform(test_features[ordinal_vars2])

# Fill Category of relationship with 6
imp_cat_6 = SimpleImputer(missing_values='nan', strategy='constant', fill_value = '6')
imp_cat_6.fit(train_features[cat_rel])
train_features[cat_rel] = imp_cat_6.transform(train_features[cat_rel])
test_features[cat_rel] = imp_cat_6.transform(test_features[cat_rel])

# Fill Category with Mode
imp_cat_mode = SimpleImputer(missing_values='nan', strategy='most_frequent')
imp_cat_mode.fit(train_features[cat_vars2])
train_features[cat_vars2] = imp_cat_mode.transform(train_features[cat_vars2])
test_features[cat_vars2] = imp_cat_mode.transform(test_features[cat_vars2])

# Fill the one binary variate with 2s
imp_2s = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 2)
imp_2s.fit(train_features[bin_liv])
train_features[bin_liv] = imp_2s.transform(train_features[bin_liv])
test_features[bin_liv] = imp_2s.transform(test_features[bin_liv])

# Fill continuous variables with median
imp_cts_med = SimpleImputer(missing_values=np.nan, strategy='median')
imp_cts_med.fit(train_features[cont_vars])
train_features[cont_vars] = imp_cts_med.transform(train_features[cont_vars])
test_features[cont_vars] = imp_cts_med.transform(test_features[cont_vars])

# Education continuous var - fill with 0
imp_cts_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0.)
imp_cts_0.fit(train_features[cont_edu])
train_features[cont_edu] = imp_cts_0.transform(train_features[cont_edu])
test_features[cont_edu] = imp_cts_0.transform(test_features[cont_edu])

# Fill Binary variables with Mode
imp_bin = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_bin.fit(train_features[bin_vars])
train_features[bin_vars] = imp_bin.transform(train_features[bin_vars])
test_features[bin_vars] = imp_bin.transform(test_features[bin_vars])

## One hot encode categorical variables - order doesn't matter
to_onehot = cat_rel + cat_vars + cat_vars2 + no_nans_cat
#Run through the helper function to get one hot encoding
[train_onehot, test_onehot] = onehot_cat(train_features, test_features, to_onehot)
train_onehot = pd.DataFrame(train_onehot)
test_onehot = pd.DataFrame(test_onehot)
#Drop original cat variables from dataframe
train_features.drop(labels = to_onehot, axis=1, inplace=True)
test_features.drop(labels = to_onehot, axis=1, inplace=True)
#Join onehot with features and re-add feature engineered vars
train_features = train_onehot.join(train_features)
test_features = test_onehot.join(test_features)
train_features = train_features.join(train_e_feats)
test_features = test_features.join(test_e_feats)

## Binary variables need to be changed from [1,2] to [0,1] ##
binary_variables = bin_vars + bin_liv
train_features[binary_variables] = train_features[binary_variables] - 1

##Save as .csv so we can just call them and not have to go through this again##
train_features.to_csv('cleaned_train.csv', index='False')
test_features.to_csv('cleaned_test.csv', index='False')