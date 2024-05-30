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
import seaborn as sns


new_df1 = pd.read_csv('C-Data/modified_data.csv')
del new_df1['Unnamed: 0']


plt.figure(figsize= (20,14))
sns.heatmap(new_df1.corr(),cmap='PuBuGn', annot = True,annot_kws={'fontsize': 8})
plt.show()


X = new_df1.drop('Churn Label', axis = 1)
y = new_df1['Churn Label']

# sm = SMOTEENN()
# X_resampled1, y_resampled1 = sm.fit_resample(X,y)

# X_train,X_test,y_train,y_test=tts(X_resampled1, y_resampled1,test_size=0.2)


# lg = LogisticRegression()
# lg.fit(X_train,y_train)


# y_pred = lg.predict(X_test)
# lg.score(X_test, y_pred)

# accuracy_score(y_test,y_pred)

# print(classification_report(y_test, y_pred))

# auc_Logistic = roc_auc_score(y_test, y_pred)
# acc_Logistic = accuracy_score(y_test, lg.predict(X_test))
# print(f'Logistic Regression - AUC: {auc_Logistic:.5f}, ACC: {acc_Logistic:.2%}')

# fpr, tpr, _ = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# # Plot the ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
# plt.legend(loc="lower right")
# plt.show()