# Importing Standard Libraries
import pandas as pd
import seaborn as sns

# Importing Libraries for Modeling
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


###############################################################################
# Linear Model Regression Creation
###############################################################################

##################################
# Full Model
##################################
bw2=bw.copy()
# Creating the model withe all variables or features to look at significance
lm_full = smf.ols(formula="""bwght ~ bw2['mage'] +
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
                  data=bw)

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
lm_full = smf.ols(formula="""bwght ~ bw2['mage'] +
                                       bw2['meduc'] +                      
                                       bw2['feduc'] +
                                       bw2['cigs'] +
                                       bw2['drink'] +
                                       bw2['out_mage'] +
                                       bw2['out_fage'] +
                                       bw2['out_npvis'] +
                                       bw2['out_drink_hi']
                                       """,
                  data=bw)

# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.summary())

################
# Scikit.learn Version

# Preparing a DataFrame based the the analysis above of significant features
bw_data2 = bw.loc[:, ['mage',
                      'meduc',
                      'feduc',
                      'cigs',
                      'drink',
                      'out_mage',
                      'out_fage',
                      'out_npvis',
                      'out_drink_hi']]

bw_target2 = bw.loc[:, 'bwght']

# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(
    bw_data2,
    bw_target2,
    test_size=0.10,
    random_state=508)

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
coeff = pd.DataFrame(lr.coef_, X_test.columns, columns=['Coefficient'])
print(coeff)

# Comparing the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Seen the final score based on variables
print(f"""     
As a result, the previous model would be able to predict the Birth Weight 
of a baby with a {final_score * 100}% accuracy based on the following features:

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
                        x_vars=['cigs', 'drink', 'mage'],
                        y_vars='bwght',
                        size=5,
                        aspect=0.7,
                        kind='reg')

# Plotting father's age, mother's education and father's education
sns_plot1 = sns.pairplot(bw,
                         x_vars=['fage', 'meduc', 'feduc'],
                         y_vars='bwght',
                         size=5,
                         aspect=0.7,
                         kind='reg')



