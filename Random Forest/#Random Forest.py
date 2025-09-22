#Random Forest
#importing the libraries
from sklearn.datasets import load_iris

#loading the dataset
iris = load_iris()
#print(dir(iris))

#creating the dataframe
import pandas as pd
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df.head()
df['target'] = iris.target
df.head()   

#splitting the dataset into features and target variable
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)
model.fit(x_train, y_train)

#accuracy 
print(model.score(x_test, y_test))

