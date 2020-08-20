#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:25:10 2019
    
Project:
    Baby's Birth Weight Analysis

Purpose:
    Developing a predictive model of baby's birth weight based on hereditary 
    and preventative factors.
"""


# Importing Standard Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Libraries for Modeling
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading Excel File with Historic Data
bw = pd.read_excel(r"datafiles\birthweight_feature_set-1.xlsx")


#############################################################################
## Basic Exploration of the Birth Weight Dataset
#############################################################################

# Looking at the dimension of the dataframe
bw.shape

# Looking at all variables, amount of observations and type of data per column
bw.info()

# Looking at the basic statistics of each feature
bw.describe().round(2)

print("Exploring the available data from the World Bank, there are {} different observations of birth weights examples and {} features for each one of them.".format(bw.shape[0], bw.shape[1]))


#############################################################################
## Missing Values' Exploration and Imputation
#############################################################################

print(bw
      .isnull()
      .sum())


for col in bw:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if bw[col].isnull().any():
        bw['m_'+col] = bw[col].isnull().astype(int)


# Verifying the size of the dataset after flagging missing values
bw.shape


##################################
# Imputing the missing values with the correspondant median 
##################################

# Feature: Mother's Education
fill = bw['meduc'].median()

bw['meduc'] = bw['meduc'].fillna(fill)


# Feature: Father's Education
fill = bw['feduc'].median()

bw['feduc'] = bw['feduc'].fillna(fill)


# Feature: Number of Prenatal Care Visits
fill = bw['npvis'].median()

bw['npvis'] = bw['npvis'].fillna(fill)


# Checking the overall dataset to see if there are any remaining missing values
print(bw
      .isnull()
      .any()
      .any())


#############################################################################
## Distribution Analysis of Continuous and Categorical Variables
#############################################################################

"""
Data Dictionary

Independent Variables: 
    
    a. Assumed Continous Variables 
        1.  mage   : mother's age                   
        2.  meduc  : mother's educ
        3.  monpre : month prenatal care began 
        4.  npvis  : total number of prenatal visits
        5.  fage   : father's age, years
        6.  feduc  : father's educ, years
        7.  omaps  : one minute apgar score
        8.  fmaps  : five minute apgar score 
        9.  cigs   : avg cigarettes per day
        10. drink  : avg drinks per week
        
    b. Assumed Categorical Binary Variables 
        11. male   : 1 if baby male
        12. mwhte  : 1 if mother white
        13. mblck  : 1 if mother black
        14. moth   : 1 if mother is other
        15. fwhte  : 1 if father white
        16. fblck  : 1 if father black
        17. foth   : 1 if father is other            |
  
Dependant Variable

        1. bwght   : birthweight, grams      

"""

##################################
# Visual EDA for Continous Variables (Histograms)
##################################

################
# Ploting the 1st set of histograms 

# Feature 1
plt.subplot(2, 2, 1)
sns.distplot(bw['mage'],
             bins = 35,
             color = 'g')

plt.xlabel("Mother's Age")


# Feature 2
plt.subplot(2, 2, 2)
sns.distplot(bw['meduc'],
             bins = 35,
             color = 'g')

plt.xlabel("Mother's Education")


# Feature 3
plt.subplot(2, 2, 3)
sns.distplot(bw['monpre'],
             bins = 35,
             color = 'g')

plt.xlabel("Month of First Prenatal Care")


# Feature 4
plt.subplot(2, 2, 4)
sns.distplot(bw['npvis'],
             bins = 35,
             color = 'g')

plt.xlabel("Number or visits")


# Saving the 1st set as a Picture
plt.tight_layout()
plt.savefig('BW Data Histograms 1 of 3.png')

plt.show()


################
# Ploting the 2nd set of histograms 


# Feature 5
plt.subplot(2, 2, 1)
sns.distplot(bw['fage'],
             bins = 35,
             color = 'g')

plt.xlabel("Father's Age")


# Feature 6
plt.subplot(2, 2, 2)
sns.distplot(bw['feduc'],
             bins = 35,
             color = 'g')

plt.xlabel("Father's Education")


# Feature 7
plt.subplot(2, 2, 3)
sns.distplot(bw['omaps'],
             bins = 35,
             color = 'g')

plt.xlabel("One-minute Apgar Score")


# Feature 8
plt.subplot(2, 2, 4)
sns.distplot(bw['fmaps'],
             bins = 35,
             color = 'g')

plt.xlabel("Five-minute Apgar Score")


# Saving the 2nd set as a Picture
plt.tight_layout()
plt.savefig('BW Data Histograms 2 of 3.png')

plt.show()


################
# Ploting the 3rd set of histograms 


# Feature 9
plt.subplot(2, 2, 1)
sns.distplot(bw['cigs'],
             bins = 35,
             color = 'g')

plt.xlabel("Cigarettes per day")


# Feature 10
plt.subplot(2, 2, 2)
sns.distplot(bw['drink'],
             bins = 35,
             color = 'g')

plt.xlabel("Drinks per day")


# Dependant Variable
plt.subplot(2, 2, 3)
sns.distplot(bw['bwght'],
             bins = 35,
             color = 'g')

plt.xlabel("Birth Weight")


# Saving the 2nd set as a Picture
plt.tight_layout()
plt.savefig('BW Data Histograms 3 of 3.png')

plt.show()


##################################
# Visual EDA Qualitative for Categorical Variables (Boxplots)
##################################

# Gender
bw.boxplot(column = ['bwght'],
                by = ['male'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("Baby Weight by Gender")
plt.suptitle("")

plt.savefig("Baby Weight by Gender.png")

plt.show()


################
# Mother White
bw.boxplot(column = ['bwght'],
                by = ['mwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("Mom's Race - White")
plt.suptitle("")

plt.savefig("Baby Weight by Mom's Race - White.png")

plt.show()


################
# Mother Black
bw.boxplot(column = ['bwght'],
                by = ['mblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("Mom's Race - Black ")
plt.suptitle("")

plt.savefig("Baby Weight by Mom's Race - Black.png")

plt.show()


################
# Mother Other
bw.boxplot(column = ['bwght'],
                by = ['moth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("Mom's Race - Other ")
plt.suptitle("")

plt.savefig("Baby Weight by Mom's Race - Other.png")

plt.show()


###############################################################################
# Correlation Analysis with Dependent Variable - Based on Pearson's Correlation
###############################################################################

df_corr = bw.corr().round(2)
df_corr.loc['bwght'].sort_values(ascending = False)


##################################
# Correlation Heatmap between all variables 
##################################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Housing Correlation Heatmap.png')
plt.show()


###############################################################################
# Outlier Analysis
###############################################################################

"""
Now, by researching and understanding the features that might influece
the health of a baby during the mother's pregnancy, it becomes necessary to 
define variables to flag the data that is considered a risk (outliers).  
"""

# Looking at the quantiles
bw_quantiles = bw.loc[:, :].quantile([0.05,
                                      0.40,
                                      0.60,
                                      0.80,
                                      0.95])

print(bw_quantiles)


##################################
# Defining limits for each risky feautre 
##################################


# Definition factor: Mother's low risk pregnancy age range 
mage_lo = 15
mage_hi = 35

# Definition factor: Father's low risk pregnancy age range 
fage_lo = 15
fage_hi = 50 

# Definition factor: 12 as the minimun to guarantee a Bachelor's degree
meduc_lo = 12
feduc_lo = 12

# Definition factor: Standard visitis are minimum 1 each month of pregnancy
npvis_lo = 10

# Definition factor: Risky number of cigars per day
cigs_hi = 0 

# Definition factor: Risky number of drinks per dat
drinks_hi = 2


##################################
#Flagging the outliers in new columns
##################################

################
# Mother's Age
bw['out_mage'] = 0


for val in enumerate(bw.loc[ : , 'mage']):
    
    if val[1] <= mage_lo:
        bw.loc[val[0], 'out_mage'] = -1


for val in enumerate(bw.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        bw.loc[val[0], 'out_mage'] = 1
        

################
# Mother's Education
bw['out_meduc'] = 0


for val in enumerate(bw.loc[ : , 'meduc']):
    
    if val[1] <= meduc_lo:
        bw.loc[val[0], 'out_meduc'] = -1
   
    
################       
# Father's Age 
bw['out_fage'] = 0

for val in enumerate(bw.loc[ : , 'fage']):
    
    if val[1] <= fage_lo:
        bw.loc[val[0], 'out_fage'] = -1
        
for val in enumerate(bw.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        bw.loc[val[0], 'out_fage'] = 1        


################
# Father's Education
bw['out_feduc'] = 0

for val in enumerate(bw.loc[ : , 'feduc']):
    
    if val[1] <= feduc_lo:
        bw.loc[val[0], 'out_feduc'] = -1


################                              
# Number of Prenatal Visits 
bw['out_npvis'] = 0


for val in enumerate(bw.loc[ : , 'npvis']):
    
    if val[1] <= npvis_lo:
        bw.loc[val[0], 'out_npvis'] = -1


################
# Cigarettes Consumption
bw['out_cigs_hi'] = 0


for val in enumerate(bw.loc[ : , 'cigs']):
    
    if val[1] >= cigs_hi:
        bw.loc[val[0], 'out_cigs_hi'] = 1


################       
# Drinks Consumption
bw['out_drink_hi'] = 0


for val in enumerate(bw.loc[ : , 'drink']):
    
    if val[1] >= drinks_hi:
        bw.loc[val[0], 'out_drink_hi'] = 1
        

###############################################################################
# Linear Model Regression Creation
###############################################################################

##################################
# Full Model
##################################
   
# Creating the model withe all variables or features to look at significance   
lm_full = smf.ols(formula = """bwght ~ bw2['mage'] +
                                       bw2['meduc'] +
                                       bw2['monpre'] +
                                       bw2['npvis'] +
                                       bw2['fage'] +
                                       bw2['feduc'] +
                                       bw2['omaps'] +
                                       bw2['fmaps'] +
                                       bw2['cigs'] +
                                       bw2['drink'] +
                                       bw2['male'] +
                                       bw2['mwhte'] +
                                       bw2['mblck'] +
                                       bw2['moth'] +
                                       bw2['fwhte'] +
                                       bw2['fblck'] +
                                       bw2['foth'] +
                                       bw2['mfwhte'] +
                                       bw2['mfblck'] +
                                       bw2['mfother'] +
                                       bw2['out_mage'] +
                                       bw2['out_meduc'] +
                                       bw2['out_fage'] +
                                       bw2['out_feduc'] +
                                       bw2['out_npvis'] +
                                       bw2['out_cigs_hi'] +
                                       bw2['out_drink_hi']
                                       """,
                                       data = bw)

# Fitting Results
results = lm_full.fit()


# Printing Summary Statistics
print(results.summary())

    
##################################
# Significant Model
##################################

################
# Statsmodels Version

# Creating the model with just the  significant variables or feature
lm_full = smf.ols(formula = """bwght ~ bw2['mage'] +
                                       bw2['meduc'] +                      
                                       bw2['feduc'] +
                                       bw2['cigs'] +
                                       bw2['drink'] +
                                       bw2['out_mage'] +
                                       bw2['out_fage'] +
                                       bw2['out_npvis'] +
                                       bw2['out_drink_hi']
                                       """,
                                       data = bw)


# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.summary())



################
# Scikit.learn Version

# Preparing a DataFrame based the the analysis above of significant features
bw_data2  =  bw.loc[:,['mage',
                       'meduc',
                       'feduc',
                       'cigs',
                       'drink',
                       'out_mage',
                       'out_fage',
                       'out_npvis',
                       'out_drink_hi']]


bw_target2 = bw.loc[:,'bwght']


# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(
            bw_data2,
            bw_target2,
            test_size = 0.10,
            random_state = 508)

# Prepping the Model
lr = LinearRegression()

# Fitting the model
lr_fit = lr.fit(X_train, y_train)

# Predictions
lr_pred = lr_fit.predict(X_test)

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

final_score = y_score_ols_optimal.round(4)


# Looking at the coefficients of the final model
coeff= pd.DataFrame(lr.coef_,X_test.columns, columns= ['Coefficient'])
print(coeff)


# Comparing the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))


# Seen the final score based on variables
print(f"""     
As a result, the previous model would be able to predict the Birth Weight 
of a baby with a {final_score*100}% accuracy based on the following features:
    
    1. Mother's Age
    2. Mother's Education (years)
    3. Father's Education (years)
    4. Average Cigarettes per Day
    5. Average Drinks per Day
    6. Number of Prenatal Visits
        """)

    
# Sending the Predictions to an Excel file
pred = pd.DataFrame(lr_pred) 

pred.to_excel('Birth Weight Predictions.xlsx')

    
###############################################################################
# Appendix
###############################################################################

##################################
# Pair Plots of main features impacting babies birth weight
##################################

# Plotting number of cigarrettes, number of drinks and mother's age
sns_plot = sns.pairplot(bw, 
                        x_vars=['cigs','drink','mage'], 
                        y_vars='bwght', 
                        size=5, 
                        aspect=0.7, 
                        kind='reg')

# Saving the plots as pictures
sns_plot.savefig('cigs-drink-mage.png')


# Plotting father's age, mother's education and father's education
sns_plot1 = sns.pairplot(bw, 
                         x_vars=['fage','meduc','feduc'], 
                         y_vars='bwght', 
                         size=5, 
                         aspect=0.7, 
                         kind='reg')

# Saving the plots as pictures
sns_plot1.savefig('fage-meduc-feduc.png')



