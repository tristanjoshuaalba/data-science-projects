import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

matplotlib.style.use('ggplot')
#Load up data downloaded from http://data.worldbank.org/indicator/SP.DYN.LE00.IN
X = pd.read_csv('C:\Users\TristanJoshua\Google Drive\!!_Data Science\Data Science\Portfolio\Data Sets/life_expectancy_world_data.csv', 
                error_bad_lines=False, header=2)

#Rename column names for indexing
X = X.rename(columns={'Country Name':'country_name',
                      'Country Code':'country_code'})

#Index Philippines/PHL life expectancy data
X = X[X.country_code == 'PHL']
X = X.drop(labels = ['country_name', 'country_code',
         'Indicator Name', 'Indicator Code'], axis = 1)

#Transpose data
X = X.transpose()
X.columns = ['life_exp']

#Extract year and life expectancy values
year = pd.DataFrame(X.index.values, columns = ['year'], index=X.index)
life_exp = pd.DataFrame(X.life_exp, columns = ['life_exp'])

#Join year and life expectancy in a single dataframe
X = year.join(life_exp, how = 'left')
X.dropna(axis=0, inplace=True)
X.year = pd.to_numeric(X.year, errors = 'coerce')

#Split Training and Testing data where Training data are label values before 1981
X_train = X[['year']][X.year<1981]
y_train = X[['life_exp']][X.year<1981]
X_test = X[['year']][X.year>1980]
y_test = X[['life_exp']][X.year>1980]

#Draw scatter plot of test data and fitted regression line
def DrawRegression(model,X_test, y_test, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(X_test, y_test, c='#40E0D0',marker='o')
  
    ax.plot(X_test, model.predict(X_test),color='orange', linewidth=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('YEAR')
    ax.set_ylabel('AGE')
    ax.legend(['Linear Regression','Actual Data'])
    plt.show()
    print "Est 2014 " + title + " Life Expectancy: %.2f" % model.predict([[2014]])[0]
    print "Est 2030 " + title + " Life Expectancy: %.2f" % model.predict([[2030]])[0]
    print "Est 2045 " + title + " Life Expectancy: %.2f" % model.predict([[2045]])[0]
    
#Train data on linear regression
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
DrawRegression(model,X_test, y_test, 'Philippines Life Expectancy')

#Score and print metrics of model
from sklearn.metrics import r2_score
score = r2_score(y_test, model.predict(X_test))
print "Score: %.3f" % score
print "Mean Squared Error: %.3f" % np.mean((model.predict(X_test) - y_test) ** 2), "years"

## OUTPUT: 
## Est Est 2014 Philippines Life Expectancy Life Expectancy: 69.88
## Est 2030 Philippines Life Expectancy Life Expectancy: 73.32
## Est 2045 Philippines Life Expectancy Life Expectancy: 76.55
## Score: 0.801
## Mean Squared Error: 0.521 years                                        