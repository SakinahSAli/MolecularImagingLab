#Random Forest
#importing the libraries
import pandas as pd
from sklearn.datasets import load_iris

#loading the dataset
iris = load_iris()

#creating the dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()

#splitting the dataset into features and target variable
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size=0.3)

#adding random  forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier
model.fit(x_train, y_train)

#accuracy 
print(model.score(x_test, y_test))