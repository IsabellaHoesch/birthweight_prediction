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

print(bw.isnull().sum())

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

# Feature: Mother's Education, Father's Education, Number of Prenatal Care Visits
fill_median=['meduc', 'feduc', 'npvis']
for e in fill_median:
    fill = bw[e].median()
    bw[e] = bw[e].fillna(fill)

# Checking the overall dataset to see if there are any remaining missing values
print(bw.isnull().any().any())


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
feat=['mage', 'meduc', 'monpre', 'npvis']
feat_name = ["Mother's Age", "Mother's Education", "Month of First Prenatal Care", "Number or visits"]
n=1
for e, i in zip(feat, feat_name):
    plt.subplot(2, 2, n)
    sns.distplot(bw[e], bins=35, color='g')
    plt.xlabel(i)
    n += 1
plt.tight_layout()
plt.show()


################
# Ploting the 2nd set of histograms

feat=['fage', 'feduc', 'omaps', 'fmaps']
feat_name = ["Father's Age", "Father's Education", "One-minute Apgar Score", "Five-minute Apgar Score"]
n=1
for e, i in zip(feat, feat_name):
    plt.subplot(2, 2, n)
    sns.distplot(bw[e], bins=35, color='g')
    plt.xlabel(i)
    n += 1
plt.tight_layout()
plt.show()

################
# Ploting the 3rd set of histograms 

feat=['cigs', 'drink', 'bwght']
feat_name = ["Cigarettes per day", "Drinks per day", "Birth Weight"]
n=1
for e, i in zip(feat, feat_name):
    plt.subplot(2, 2, n)
    sns.distplot(bw[e], bins=35, color='g')
    plt.xlabel(i)
    n += 1
plt.tight_layout()
plt.show()


##################################
# Visual EDA Qualitative for Categorical Variables (Boxplots)
##################################

# Gender
bw.boxplot(column = ['bwght'], by = ['male'], vert = False, patch_artist = False, meanline = True, showmeans = True)
plt.title("Baby Weight by Gender")
plt.suptitle("")
plt.show()


################
# Mother White
bw.boxplot(column = ['bwght'], by = ['mwhte'], vert = False, patch_artist = False, meanline = True, showmeans = True)
plt.title("Mom's Race - White")
plt.suptitle("")
plt.show()


################
# Mother Black
bw.boxplot(column = ['bwght'], by = ['mblck'], vert = False, patch_artist = False, meanline = True, showmeans = True)
plt.title("Mom's Race - Black ")
plt.suptitle("")
plt.show()


################
# Mother Other
bw.boxplot(column = ['bwght'], by = ['moth'], vert = False, patch_artist = False, meanline = True, showmeans = True)
plt.title("Mom's Race - Other ")
plt.suptitle("")
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
# Defining limits for each risky feature
##################################


# Definition factor: Mother's low risk pregnancy age range 
mage_lo = 15
mage_hi = 35

# Definition factor: Father's low risk pregnancy age range 
fage_lo = 15
fage_hi = 50 

# Definition factor: 12 as the minimum to guarantee a Bachelor's degree
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
        

