import pandas as pd
X = pd.read_csv('C:\Users\TristanJoshua\Google Drive\!!_Data Science\Data Science\Portfolio\Data Sets\credit_card_defaults.csv', header=1)
X.head()

#Check dataframe for NAN's: No NAN's! Great!
X[pd.isnull(X).any(axis=1)]

#Rename column for default status label
X = X.rename(columns={'default payment next month':'default'})

#Separate default label column into another data frame
y = X['default'].copy()
X.drop(labels=(['default', 'ID']), axis=1, inplace=True)

#Map out SEX values into male and female
X.SEX = X.SEX.map({1:'Male', 2:'Female'})

#Convert NOMINAL CATEGORICAL variables columns
col = X.columns.values.tolist()[1:4]
for i in col:
    X[str(i)] = X[str(i)].astype('category')
    
#Convert NOMINAL CATEGORICAL variables columns
#PAY_0,2,3,4,5,6 and LIMIT_BAL are ordinal categorical 
col = X.columns.values.tolist()[5:11]
for i in col:
    X[str(i)] = X[str(i)].astype('category', ordered=True) 
    
#Limit balance for CC also comes in tiers of suprisingly 81 levels
X.LIMIT_BAL = X.LIMIT_BAL.astype("category", ordered=True)

#Get dummy variables for SEX
X = pd.get_dummies(X, columns=['SEX'],prefix=['SEX'])

#Split data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=7)

#Fit Decision Tree Classifier using training data 
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=4, criterion="entropy")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print "DTree Score is %.3f"%score

#Get importances of each feature in a data frame
impt = pd.DataFrame(zip(X.columns, model.feature_importances_))

#Export decision tree visual
tree.export_graphviz(model.tree_, out_file = 'CC-Defaults.dot',\
                     proportion= True, feature_names = X.columns,\
                    filled = True, rounded=True)
from subprocess import call
call(['dot', '-T','png', 'CC-Defaults.dot', '-o', 'CC-Defaults.png'])


#Evaluate the Model

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

#Insert prediction and prediction probabilities into separate dataframes
pred_label = model.predict(X_test)
pred_proba = model.predict_proba(X_test)
test_results = X_test.copy()
test_results['status_pred_label'] = pred_label
test_results['status_pred_proba'] = pred_proba[:,1]
test_results['status_true'] = y_test

#Evaluation of the model

#Initiate ROC Curve and AUC score computation
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test,pred_proba[:,1])
rsc = roc_auc_score(y_test,pred_proba[:,1], average = 'samples') 
  
#Plot ROC Curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fpr,tpr, c='#FFA500')
ax.plot([0,1], '--',c='#40E0D0')
ax.set_title('CC DEFAULTS CLASSIFICATION')
ax.set_xlabel('False Positive Rate')    
ax.set_ylabel('True Positive Rate')

ax.grid(which='major', linestyle='-', color='#D3D3D3') 
ax.set_axis_bgcolor('#ffffff')
plt.text(0.55, 0.05, 'AREA UNDER THE CURVE = %.3f' % rsc )
ax.legend(['ROC Curve','Random Guess'])
plt.show()

#Show classification metrics
from sklearn.metrics import confusion_matrix as cm,\
                            classification_report as cr,\
                            accuracy_score as asc
cm = cm(y_test,pred_label)
cr = cr(y_test,pred_label)
asc = asc(y_test,pred_label)

print 'CC DEFAULTS Classification Report\n\n', cr
print 'Accuracy: %.3f' % asc
print "True Negatives  = %s \nFalse Negatives = %s \nTrue Positives"\
      "  = %s \nFalse Positives = %s" % (cm[0,0],cm[1,0],cm[1,1],cm[0,1])
 


