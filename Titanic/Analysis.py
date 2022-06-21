#Importing libraries and packages
from operator import index
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
 
#Loading data
pd.set_option("display.max_columns", None)
training_data = pd.read_csv(r'E:\Titanic\titanic.csv\train.csv')
testing_data =  pd.read_csv(r'E:\Titanic\titanic.csv\test.csv')
training_data['train_test'] = 1
testing_data['train_test'] = 0
testing_data['Survived'] = np.NaN
all_data = pd.concat([training_data,testing_data])
#print('Testing columns are: ',testing_data.describe().columns)

#Looking at numerical and categorical data separately
df_num = training_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
df_cat = training_data[['Survived', 'Pclass', 'Sex', 'Cabin', 'Embarked']]
print('Training data is: ',training_data.describe().columns)

#Distrinutions for numeric data
for i in df_num.columns:
    plt.hist (df_num[i])
    plt.title(i)
    #plt.show()    

#Correlations
print(df_num.corr())
sns.heatmap(df_num.corr())
#plt.show()
print(pd.pivot_table(training_data, index='Survived', values=['Pclass','Age','SibSp','Parch','Fare']))

# compare survival rate across Age, SibSp, Parch, and Fare 
pd.pivot_table(training_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])

#Distribution for categorical data
for i in df_cat.columns:
   sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
   plt.show()

print(pd.pivot_table(training_data, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(training_data, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(training_data, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))

df_cat.Cabin
training_data['cabin_multiple'] = training_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
# after looking at this, we may want to look at cabin by letter or by number. Let's create some categories for this 
# letters 
# multiple letters 
training_data['cabin_multiple'].value_counts()
pd.pivot_table(training_data, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')
#creates categories based on the cabin letter (n stands for null)
#in this case we will treat null values like it's own category

training_data['cabin_adv'] = training_data.Cabin.apply(lambda x: str(x)[0])
#comparing surivial rate by cabin
print(training_data.cabin_adv.value_counts())
pd.pivot_table(training_data,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')
#understand ticket values better 

#numeric vs non numeric 
training_data['numeric_ticket'] = training_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
training_data['ticket_letters'] = training_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
training_data['numeric_ticket'].value_counts()

#Viewing lettered tickets
#pd.set_option('max_rows', None)
training_data['ticket_letters'].value_counts()
#Survival rate by numeric ticket

pd.pivot_table(training_data,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')
#Survival rate by lettered ticket
pd.pivot_table(training_data,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')

#Ticket type does not play as major a role


#Determining if there is a correlation between Title and survival
training_data.Name.head(50)
training_data['Name_Title'] = training_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
training_data['Name_Title'].value_counts()
pd.pivot_table(training_data,index='Survived',columns='Name_Title', values = 'Name', aggfunc='count')

#Next is modelling pre-processing



#create all categorical variables that we did above for both training and test sets 
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data 
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(training_data.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(training_data.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 
all_data.dropna(subset=['Embarked'],inplace = True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()

# log norm of fare (used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

# converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape

# Scale data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


svc = SVC(probability = True)
cv = cross_val_score(svc,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1, use_label_encoder=False)
cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#Voting classifier takes all of the inputs and averages the results. For a "hard" voting classifier each classifier gets 1 vote "yes" or "no" and the result is just a popular vote. For this, you generally want odd numbers
#A "soft" classifier averages the confidence of each of the models. If a the average confidence is > 50% that it is a 1 it will be counted as such
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())



