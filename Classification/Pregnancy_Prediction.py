import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.style.use('ggplot')

X = pd.read_csv('C:\Users\TristanJoshua\Google Drive\!!_Data Science\Data Science\Portfolio\Data Sets\pregnancy_detection.csv')
X[pd.isnull(X).any(axis=1)]
X.head()
X.describe()

#Display number of pregnant and not pregnant based on implied gender
print "Not Pregnant \n", X.implied_gender[X.status==0].value_counts()
print "Pregnant \n", X.implied_gender[X.status==1].value_counts()

#Represent implied_gender and address as dummy variables
X = pd.get_dummies(X, columns=['implied_gender','address'],prefix=['gndr','addr'])
X.groupby(by='status').hist(figsize=(20,20), xlabelsize = 0.5)

#Since features are dichotomous, they are converted into categories
for column in X:
    X[column] = X[column].astype("category")
    
#The label 'status' is copied to dataframe y and dropped from X
y = pd.DataFrame(X['status'].copy())
X.drop(labels = ['status'], axis = 1, inplace=True)

#Data is split into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30,
                                                    random_state=9)

#Logistic regression model is trained using training data
from sklearn.linear_model import LogisticRegression as logreg
model = logreg()
model.fit(X_train, y_train)

#Display mean accuracy on testing data and testing labels
score = model.score(X_test, y_test)

#Interpret coefficients: Positive coefficients increases log probability of pregnancy
logreg_coef = pd.DataFrame(zip(X.columns, model.coef_[0]))

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
ax.set_title('PREGNANCY DETECTION PERFORMANCE')
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

print 'Pregnancy Prediction Classification Report\n\n', cr
print 'Accuracy: %.3f' % asc
print "True Negatives  = %s \nFalse Negatives = %s \nTrue Positives"\
      "  = %s \nFalse Positives = %s" % (cm[0,0],cm[1,0],cm[1,1],cm[0,1])
 
