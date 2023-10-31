import pandas as pd

CRX = pd.read_csv('crxdataReducedClean.csv',header = None)
CRX.shape
CRX.head()
CRX.info()
CRX.describe()
CRXx = CRX.copy()


# Select the Categorical Features
categorical_features = []
for x, xCols in CRXx.iteritems():
    if(CRXx[x].dtype == 'object'):
        categorical_features.append(x)


# Import Library for Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Apply the Encoding on Categorical Features
# Apply the Encoding on Categorical Features
for col in categorical_features:
    CRXx[col] = le.fit_transform(CRXx[col]).astype("int8")
    
CRX.dtypes
CRXx.head()


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=len(CRXx), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)
NBS=gnb.score(X_train, y_train)

#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score

NBAS=accuracy_score(y_test, y_pred)
print("Score:",NBS)
print("Accuracy:",NBAS)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

names = [
    
    "K-Nearest Neighbours",
    "Random Forest Classifier",
    "Decision Tree Classifier",
    "Bagging Classifier",
    "Extra Trees Classifier"
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    DecisionTreeClassifier(),
    BaggingClassifier(n_estimators=10, random_state=0),
    ExtraTreesClassifier(random_state=1)
]


modelScore = []
modelAccuracy = []

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    predict = clf.predict(X_test)
    
    modelScore.append(score)
    modelAccuracy.append(accuracy_score(y_test, predict))

print("\n"*2)
print("Score of Train Models")
print("-"*25)
modelScore

print("Model Accuracy")
print("-"*15)
modelAccuracy

names.insert(0, 'Naive Bayes')
modelScore.insert(0, NBS)
modelAccuracy.insert(0, NBAS)

df_modeAccuracy = pd.DataFrame(columns=['name', 'score', 'accuracy'])
df_modeAccuracy['name'] = names
df_modeAccuracy['score'] = modelScore
df_modeAccuracy['accuracy'] = modelAccuracy

df_modeAccuracy
