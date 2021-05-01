#Load in necessary libraries needed
import pandas as pd
import numpy as np

# ----- Feature Engineering -----

##Helper functions##
#Count the number of NANs or values in check are in each row
def num_nan(df, check):
    #check are values to be treated as NANs - replace
    df.replace(to_replace = check, value = np.nan, inplace = True)
    #Gets the number of NANs in each row - returns list can append as variable
    lst = df.isnull().sum(axis = 1)
    return lst

#Count number of times certain values appear in a variable
def num_var(df, vars, check):
    #intialize list of 0s for each row
    lst = [0]*len(df)
    #subset to only have needed variables
    df = df[vars]
    #For each val to check, add to current list to update count
    for val in check:
        lst = lst + df.isin([val]).sum(axis=1)
    return lst

#Function to remove variables no longer needed
def strip_columns(df, remove):
    for i in remove:
        df.drop(i, axis='columns', inplace=True)
    pass


##Read in data##
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

## Get necessary info for feature engineering ##
#Variables depicting relations in househould of 2nd to 13th person
col_house_rel = ['v192', 'v193', 'v194', 'v195', 'v196','v197', 'v198', 'v199', 'v200', 'v201', 'v202', 'v203']
#Condensed levels: "1" was partner, "2" child, 3 parent/in-law, 4 other relative, 5 non-relative
child = ['2']
relative = ['2', '3', '4']
non_relative = ['5']

#Variables of the gender of person in household
col_house_gend = ['v86', 'v87', 'v88', 'v89', 'v90','v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97']
male = ['1']
female = ['2']

#Variables of year of birth of members of household
col_house_age = ['v259', 'v260', 'v261', 'v262', 'v263','v264', 'v265', 'v266', 'v267', 'v268', 'v269', 'v270']
#Made lists of values for different importance age categories
under_13 = [str(i) for i in (range(2015,2002,-1))]
teen_13_19 = [str(i) for i in (range(2002, 1996, -1))]
over_70 = [str(i) for i in (range(1945, 1900, -1))]

## Apply functions to create new variables ##
#Relations in household train and test
train_data['Children in Household'] = num_var(train_data, col_house_rel, child)
train_data['Relatives in Household'] = num_var(train_data, col_house_rel, relative)
train_data['Non-Relatives in Household'] = num_var(train_data, col_house_rel, non_relative)
test_data['Children in Household'] = num_var(test_data, col_house_rel, child)
test_data['Relatives in Household'] = num_var(test_data, col_house_rel, relative)
test_data['Non-Relatives in Household'] = num_var(test_data, col_house_rel, non_relative)
#Genders in household train and test
train_data['Males in Household'] = num_var(train_data, col_house_gend, male)
train_data['Females in Household'] = num_var(train_data, col_house_gend, female)
test_data['Males in Household'] = num_var(test_data, col_house_gend, male)
test_data['Females in Household'] = num_var(test_data, col_house_gend, female)
#Numer of age groups in household train and test
train_data['Under 13 in Household'] = num_var(train_data, col_house_age, under_13)
train_data['Teenagers in Household'] = num_var(train_data, col_house_age, teen_13_19)
train_data['Elderly in Household'] = num_var(train_data, col_house_age, over_70)
test_data['Under 13 in Household'] = num_var(test_data, col_house_age, under_13)
test_data['Teenagers in Household'] = num_var(test_data, col_house_age, teen_13_19)
test_data['Elderly in Household'] = num_var(test_data, col_house_age, over_70)

## Removed variables we condensed above ##
#has repeat with relation columns
repeat_col = ['v204', 'v205', 'v206', 'v207', 'v208','v209', 'v210', 'v211', 'v212', 'v213', 'v214', 'v215']
to_remove = col_house_rel + col_house_gend + col_house_age + repeat_col
strip_columns(train_data, to_remove)
strip_columns(test_data, to_remove)

## Remove variables deemed already condensed by dataframe ##
#Remove distinct discrimation as a condensed variable exists in dataframe already
remove_discrim = ['v38', 'v39', 'v40', 'v41', 'v42', 'v44', 'v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51', 'v52']
strip_columns(train_data, remove_discrim)
strip_columns(test_data, remove_discrim)
#Remove distinct 7 day activities - variable exists with complete list of all activities
remove_7day = ['v15', 'v22', 'v26', 'v28', 'v31', 'v33', 'v36', 'v54', 'v106', 'v158', 'v175', 'v217', 'v244', 'v246']
strip_columns(train_data, remove_7day)
strip_columns(test_data, remove_7day)
#Remove partner 7 day activities - same as above
remove_partner7 = ['v16', 'v23', 'v27', 'v29', 'v30', 'v32', 'v34', 'v37', 'v55', 'v107', 'v176', 'v218', 'v245', 'v247']
strip_columns(train_data, remove_partner7)
strip_columns(test_data, remove_partner7)

## Check number of NANs and make a variable ##
#To avoid overcounting NANs represented in multiple variables only counted after removed some redundancy
check = ['.a', '.b', '.c', '.d', '.']
train_data['No Answer'] = num_nan(train_data, check)
test_data['No Answer'] = num_nan(test_data, check)

## Remove useless variables ##
remove_time = ['v124', 'v125', 'v126', 'v127', 'v128', 'v129', 'v130', 'v131', 'v133', 'v134']
strip_columns(train_data, remove_time)
strip_columns(test_data, remove_time)
#
remove_notneed = ['v5', 'v63', 'v69', 'v103', 'v123', 'v150', 'v151', 'v160', 'v168', 'v173', 'v174', 'v241', 'v243', 'v252']
strip_columns(train_data, remove_notneed)
strip_columns(test_data, remove_notneed)

#Get the label from train as a separate list and remove from dataframe
train_label = train_data['satisfied']
strip_columns(train_data,['satisfied'])
#Save necessary files to reduce runtime - can load as needed to skip steps
# train_data.to_csv('Feat_Train.csv', index='False')
# test_data.to_csv('Feat_Test.csv', index='False')
train_label.to_csv('Train_Label.csv', index='False')

#With imputations later wanted to keep original stripped data separate from feat eng variables
feat_var = ['Children in Household', 'Relatives in Household', 'Non-Relatives in Household', 'Males in Household',
'Females in Household', 'Under 13 in Household', 'Teenagers in Household', 'Elderly in Household']
#F_train and F_test are just the new features
F_train = train_data[feat_var]
F_test = test_data[feat_var]
F_train.to_csv('F_train.csv', index='False')
F_test.to_csv('F_test.csv', index='False')
#Remove feature variables and return just stripped train and test - reassigned to not overwrite train_data, test_data if needed later
strip_train = train_data
strip_test = test_data
strip_columns(strip_train, feat_var)
strip_columns(strip_test, feat_var)
strip_train.to_csv('strip_train.csv', index='False')
strip_test.to_csv('strip_test.csv', index='False')
##strip_train and strip_test are dataframes with removed vars but no features added

