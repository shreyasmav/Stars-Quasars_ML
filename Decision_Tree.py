import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y,depth =0): #training the model
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        
    def predict(self, X): #predicting for new examples
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y): #finding the best split at a node 
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0): 
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    
    def score(self,pred,y_test):
        Total_len =  len(y_test)
        actual_true = 0
        y_array = np.array(y_test)
        for i in range(Total_len):
            if pred[i]==y_array[i]:
                actual_true+=1
        return actual_true/Total_len


df = pd.read_csv('catalog3/cat3.csv',index_col=0)
X = df.drop(['pred','galex_objid','sdss_objid','class','spectrometric_redshift'],axis=1)
y = df['class']

#For scaling the data 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# scaler.transform(X)

   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state = 30)
   
#Using SMOTE for generating synthetic samples for stars
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_train = np.array(X_train_res)
y_train = np.array(y_train_res)
 
X_train_res = np.array(X_train)
y_train_res = np.array(y_train)
X_test = np.array(X_test)

#To find the optimal max_depth
# depth_accu =[] #For keeping accuracy of each max_depth value
# for i in range(1,16):
#     clf = DecisionTreeClassifier(max_depth=i)
#     clf.fit(X_train_res, y_train_res) #Training the model
#     predictions = clf.predict(X_test) #Getting the predictions
#     accu = clf.score(predictions,y_test) #Calaculating the accuracy
#     temp = [0,0]
#     temp[0]=i
#     temp[1]=accu
#     depth_accu.append(temp)
#     print("max_dept = " ,i)
#     print(classification_report(y_test, predictions))
#     print('\n')
#     print(confusion_matrix(y_test, predictions))
#     print('---------------------------------------------------------------------------------------')

clf = DecisionTreeClassifier(max_depth=8)
clf.fit(X_train_res, y_train_res) #Training the model
predictions = clf.predict(X_test)
accuracy = clf.score(predictions,y_test)
print('Accuracy :',accu)
print('Classification Report')
print(classification_report(y_test, predictions))
print('Confusion Matrix')
print(confusion_matrix(y_test, predictions))

