# Customer Churn Prediction using Deep Learning

## 📌 Project Overview
This project predicts customer churn in a bank using both machine learning and deep learning approaches. The dataset includes customer demographics, account information, and activity status. We build and compare models using **Deep Neural Networks (TensorFlow/Keras)** and **XGBoost** to improve accuracy and performance.

## 🚀 Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, TensorFlow/Keras, XGBoost
- **Tools:** Jupyter Notebook, Google Colab, GitHub

## 📂 Dataset Description
The dataset consists of customer details and includes the following features:

| Feature           | Description |
|------------------|-------------|
| Customer ID      | Unique identifier for each customer |
| Surname         | Customer's last name |
| Credit Score    | Numerical credit score |
| Geography       | Customer's country |
| Gender          | Customer's gender |
| Age             | Customer's age |
| Tenure          | Years with the bank |
| Balance         | Account balance |
| NumOfProducts   | Number of bank products used |
| HasCrCard       | If the customer owns a credit card (0/1) |
| IsActiveMember  | If the customer is an active member (0/1) |
| EstimatedSalary | Customer's estimated salary |
| Exited (Target) | 1 = Customer churned, 0 = Customer stayed |

## 🔍 Exploratory Data Analysis (EDA)
- Checked for missing values
- Analyzed distributions of numerical and categorical features
- Plotted correlation heatmaps and histograms

## ⚙️ Data Preprocessing
- Encoded categorical features using **Label Encoding** (Geography, Gender)
- Removed unnecessary columns (Customer ID, Surname)
- Normalized numerical features
- Split data into **80% training** and **20% testing**

## 🏗️ Model Building
### 1️⃣ Deep Learning Model (TensorFlow/Keras)
A **5-layer Neural Network** was implemented:
- **Input Layer**: 64 neurons (ReLU activation)
- **Hidden Layers**: 32, 16, and 8 neurons (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation for binary classification)
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Early Stopping:** To prevent overfitting

### 2️⃣ XGBoost Model
Implemented **XGBoost Classifier** to compare performance with the deep learning model.

## 📊 Model Evaluation
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### 🔹 Deep Learning Model Results:
- Training Accuracy: **~78.9%**
- Validation Accuracy: **~78.9%**

### 🔹 XGBoost Model Results:
- Accuracy: **86.6%**
- Precision: **86.6%**
- Recall: **86.6%**
- F1 Score: **86.6%**

## 📌 Key Findings
- XGBoost outperformed the deep learning model in this case.
- The model could benefit from hyperparameter tuning and additional feature engineering.
- Class imbalance may be affecting model performance; techniques like **SMOTE** can be explored.

## 🔧 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction-Deep-Learning.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
   ```
3. Run the Jupyter Notebook or Python script.

## 📌 Future Improvements
- Hyperparameter tuning for better model performance
- Implementing **SMOTE** to handle class imbalance
- Exploring other ML models like **Random Forest** and **Gradient Boosting**
- Deploying the model as an API for real-world use

## ✨ Author
👨‍💻 **Muhammad Fakhar ul Hasnain**

Feel free to contribute, raise issues, or fork the repository! 🚀

