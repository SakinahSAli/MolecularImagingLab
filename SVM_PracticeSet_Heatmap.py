
#import necessary libraries
from matplotlib import cm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


# Replace the file path with the correct path to your CSV file
file_path = r"c:\Users\SakinahAli\Downloads\recipes_muffins_cupcakes (1).csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

#plot two ingredients using seaborn
sns.lmplot(x='Milk', y='Butter', data=df, hue='Type', markers=['o', 's'], palette='Set1', fit_reg=False, scatter_kws={'s': 70})

# Show the plot due to this IDE 
plt.show()
X = df[['Milk', 'Butter']]
y = df['Type']
  
# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=250)

# Create a SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)    
print("Predicted values:", y_pred)  

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the classifier
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))      
