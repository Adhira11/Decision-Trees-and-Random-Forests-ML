import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Visualize Decision Tree
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=feature_names,  
                           class_names=data.target_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree_visualization")

# 2. Analyze Overfitting - Check accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# 3. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# 4. Feature Importances
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=feature_names)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# 5. Cross-validation scores
dt_scores = cross_val_score(dt_model, X, y, cv=5)
rf_scores = cross_val_score(rf_model, X, y, cv=5)
print("Decision Tree CV Accuracy:", dt_scores.mean())
print("Random Forest CV Accuracy:", rf_scores.mean())
