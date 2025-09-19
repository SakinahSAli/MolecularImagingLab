#Decision Tree 
#import appropiate libraries
import pandas as pd

#create dataframe
df = pd.read_csv(r"C:\Users\SakinahAli\Downloads\titanic.csv")
#print(df.head())

#drop unnecessary columns
# Only drop columns that still exist in 'inputs'
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
# After initial drop, drop target
inputs = df.drop('Survived', axis='columns')
target = df.Survived
#print(inputs.head())

#Convert categorical data to numerical data
inputs.Sex = inputs.Sex.map({'male':1,'female': 2})
inputs.Age = inputs.Age.fillna(inputs.Age.mean())  #fill NaN values with mean
#print(inputs.head())


#Train model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.7)

print(len(x_train))

print(len(x_test))

#Make predictions
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))