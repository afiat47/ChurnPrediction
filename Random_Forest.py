from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt
import pandas as pd

new_df1 = pd.read_csv('C-Data/modified_data.csv')
del new_df1['Unnamed: 0']

X = new_df1.drop('Churn Label', axis = 1)
y = new_df1['Churn Label']



X_train,X_test,y_train,y_test = tts(X,y,test_size= 0.2, random_state= 42)

rfc = RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
rfc.fit(X_train,y_train)
y_pred_rf = rfc.predict(X_test)

print(rfc.score(X_test, y_pred_rf))
print(accuracy_score(y_test,y_pred_rf))

print(classification_report(y_test, y_pred_rf))

auc_rf = roc_auc_score(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, rfc.predict(X_test))
print(f'Random Forest - AUC: {auc_rf:.5f}, ACC: {acc_rf:.2%}')


fpr1, tpr1, _ = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr1, tpr1)

# Plot the ROC curve
plt.figure()
plt.plot(fpr1, tpr1, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()