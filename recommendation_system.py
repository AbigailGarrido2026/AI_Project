
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("products.csv")
X_train, X_test, Y_train, y_test = train_test_split(data[["feature1", "feature2","feature3"]],data["label"], test_size=0.2)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)
print(f"Accuracy: {model.score(X_test, y_test)*100:.2f} %")