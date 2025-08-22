import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
data.head()

# Preprocessing
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
X['Sex'] = X['Sex'].map({'male':0,'female':1})
X = X.fillna(X.mean())
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
preds = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test,preds))
