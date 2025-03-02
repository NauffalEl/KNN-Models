# 📌 KNN Models: Classification & Regression

## 🎯 Description
Welcome to **KNN Models**, a repository that contains implementations of the **K-Nearest Neighbors (KNN)** and **Decision Tree** algorithms for **classification** and **regression** on various datasets.

The KNN method is used to **predict categories or values based on their nearest neighbors**, while the Decision Tree helps in making structured decisions.

---
## 📂 Directory Structure

```
KNN-Models/
│── KNN Classification/
│   ├── KNN Classification.ipynb
│── KNN Regression/
│   ├── KNN Regression.ipynb
│── Decision Tree Regression/
│   ├── Decision Tree Regression.ipynb
│── KNN Cancer/
│   ├── KNN Classification Cancer Data.ipynb
│── README.md
```

Each folder contains different model implementations:
- **KNN Classification:** KNN model for general classification tasks.
- **KNN Regression:** KNN model for numerical regression.
- **Decision Tree Regression:** Decision Tree model for regression tasks.
- **KNN Cancer:** Specialized KNN model for cancer classification.
- **datasets/** contains the datasets used in this project.

---
## 🚀 Installation & Usage

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/NauffalEl/KNN-Models.git
cd KNN-Models
```

### 2️⃣ **Install Dependencies**
Make sure you have **Python 3.x** and run the following command:
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run Jupyter Notebook**
```bash
jupyter notebook
```
Then, open the **`.ipynb`** files in Jupyter Notebook to run the experiments.

---
## 📊 Models Used

| Model                      | Description |
|----------------------------|------------------------------------------------|
| **KNN Classification**      | Predicts categories based on k-nearest neighbors |
| **KNN Regression**         | Predicts numerical values using neighbor interpolation |
| **Decision Tree Regression** | Creates a decision tree for regression tasks |
| **KNN Cancer Classification** | Model for detecting cancer based on dataset |

---
## 📖 Example Usage

### 🔹 **KNN Classification**
```python
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load dataset
data = pd.read_csv('datasets/knn_data.csv')
X = data.drop('label', axis=1)
y = data['label']

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Prediction
prediction = knn.predict([[5.1, 3.5, 1.4, 0.2]])
print("Predicted Class:", prediction)
```

---
## 🎯 Contributor
💡 Created by **[Nauffal El](https://github.com/NauffalEl)**

If you have suggestions or would like to contribute, feel free to make a **Pull Request** or submit an **Issue**! 🚀

---
## 📜 License
This project is released under the **MIT License**. You are free to use, modify, and distribute this project with proper attribution.

---
🔥 **Happy experimenting with KNN & Decision Tree!** 🚀

