# DATA COLLECTION 

# Import the Libraries
import pandas as pd                                                        #for data manipulation
import numpy as np                                                         #for numerical operations
from matplotlib import pyplot as plt                                       #for visualization
import seaborn as sns                                                      #for visualization
from sklearn.ensemble import RandomForestClassifier                        #for using Random forest Classifier algorithm
from sklearn.metrics import accuracy_score                                 #for accouracy

# Load the Dataset
df = pd.read_csv("C:/Users/Lenovo/Desktop/ML.Project/diabetes.csv")


# DATA EXPLORATION

print('Total number of records: ',len(df),'\n')                                  # For checking number of records

print("Parameter are: ",df.columns,'\n')                                         #For printing coloumn names

print(df.info(),'\n')                                                             #Get insight about data

print(df.describe(),'\n')                                                         

print(df['Outcome'].describe(),'\n')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# DATA PREPROCESSING
# In this step we will perform task as handling missing values and encode catagorical data

# Handling missing value
print(df.isnull().values.any(),'\n')

true_count= len(df.loc[df['Outcome']==True])
False_count= len(df.loc[df['Outcome']==False])
print(true_count,False_count,'\n')

# Finding Correlation between Attributes
print(df.corr()['Outcome'].sort_values(),'\n')


# DATA SPLITTING
# In this step we will split data in to testing and training data

from sklearn.model_selection import train_test_split
X = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

# MODEL SELECTION and MODEL TRAINING
# Firstly we will select the right algorithm according to our requirement and then we will train data on model

n_estimators = 10
base_models = []
     
for i in range(n_estimators):
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, Y_train)
    base_models.append(rf)
    
# Model Evaluation or Testing

predictions = [model.predict(X_test) for model in base_models]
     
ensemble_predictions = np.round(np.mean(predictions, axis=0))
     
ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {ensemble_accuracy}")


# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(Y_test, ensemble_predictions)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion)

cm = confusion_matrix(Y_test, ensemble_predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# FINAL RESULT
# Get input from the user
user_input = {
    'Glucose': float(input('Enter Glucose level: ')),
    'BloodPressure': float(input('Enter Blood Pressure: ')),
    'SkinThickness': float(input('Enter Skin Thickness: ')),
    'Insulin': float(input('Enter Insulin level: ')),
    'BMI': float(input('Enter BMI: ')),
    'DiabetesPedigreeFunction': float(input('Enter Diabetes Pedigree Function: ')),
    'Age': float(input('Enter Age: '))
}

# Convert the user input into a DataFrame
user_df = pd.DataFrame([user_input])

# Make a prediction using the trained model
user_prediction = rf.predict(user_df)

# Interpret the prediction
if user_prediction[0] == 0:
    result = "No diabetes"
else:
    result = "Diabetes"

print(f"The model predicts: {result}")