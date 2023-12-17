# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# %%
df = pd.read_csv("./dataset/data.csv")
df.head()

# %%
df.y.value_counts()

# %%
all_columns = df.columns
all_columns

# %% [markdown]
# ## `EDA: ` Exploratory Data Analysis

# %% [markdown]
# ### Drop Id Column

# %%
# Drop the id column as it is not required for now
print(f"Shape of dataset befor id removal: {df.shape}")
del df['Id']
print(f"Shape of dataset After id removal: {df.shape}")

# %% [markdown]
# ### Drop Null Values

# %%
df.isnull().sum()
print(f"Shape of dataset befor drop null values: {df.shape}")
total_rows = len(df.y)
df = df.dropna()
print(f"Shape of dataset after drop null values: {df.shape}")
print(f"Total number of rows contains null values: {total_rows - len(df.y)}")

# %% [markdown]
# ### Make feature list

# %%
target = 'y'
# create the list of categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"categorical_features : {categorical_features}")
# create the list of numerical features
numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
print(f"numerical_features : {numerical_features}")

# %%
sns.displot(df.age, kde=True)

# %%
df.age.describe()

# %% [markdown]
# we can see `-1` and `999.0` are the anomalies in the data set. lets drop it

# %%
# This is illogical or not possible value
print(f"number of records for 999.0 {df.age[df.age > 100].value_counts()}")
print(f"number of records for -1 {df.age[df.age < 0].value_counts()}")

# %%
# Let's drop the illogical age value
df = df[df.age != 999.0]
df = df[df.age != -1]

# %%
# After dropping the illogical Age Value
sns.displot(df.age, kde=True)

df.age.describe()

# %%
import matplotlib.pyplot as plt
job_category_count = df['job'].value_counts()

plt.figure(figsize=(8,6))
plt.bar(job_category_count.index, job_category_count.values)

plt.xlabel('Job')
plt.ylabel('count')
plt.title('Job Distribution in Dataset')
plt.xticks(rotation=45)
plt.show()

print(f"{df['job'].value_counts()}")
print("----------------------------------")
df['job'].describe()

# %% [markdown]
# ## Analyse `df['marital']`

# %%
marital_status_count = df['marital'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(marital_status_count, labels=marital_status_count.index, startangle=90, autopct='%1.1f%%',)

plt.title('Marital Distribution in Dataset')
plt.show()

print(f"{df['marital'].value_counts()}")
print("----------------------------------")
df['marital'].describe()

# %% [markdown]
# ## Analyse `df['education']`

# %%
education_count = df['education'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(education_count, labels=education_count.index, startangle=90, autopct='%1.1f%%',)

plt.title('education Distribution in Dataset')
plt.show()

print(f"{df['education'].value_counts()}")
print("----------------------------------")
df['education'].describe()

# %% [markdown]
# ## Relationship between `age` and `balance`
# lets visualize it using scatter plot

# %%
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['balance'], alpha=0.5)
plt.xlim(0, 100)
plt.title('Scatter Plot of Age vs Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()

# %% [markdown]
# ## Relationship between `age` and `loan`
# lets visualize it using scatter plot

# %%
plt.figure(figsize=(8, 4))
plt.scatter(df['age'], df['loan'], alpha=0.5)
plt.xlim(0, 100)
plt.title('Scatter Plot of Age vs loan')
plt.xlabel('Age')
plt.ylabel('loan')
plt.show()

# %% [markdown]
# ### `Calculate mutual information of categorical variables`

# %%
from sklearn.metrics import mutual_info_score
scores = []
for col in categorical_features:
    score = mutual_info_score(df[target], df[col])
    scores.append(round(score, 4))
    print(f"{col} -> {round(score, 4)}")
print(f"Scores : {scores}")

# %%
import numpy as np

categorical_data = np.array(list(zip(categorical_features, scores)), dtype=[('category', 'U10'), ('value', float)])

# Sort the NumPy array by the 'value' field
sorted_data = np.sort(categorical_data, order='value')

# Convert the sorted NumPy array back to a list of tuples (optional)
sorted_list = sorted_data.tolist()

sorted_list

# %% [markdown]
# ### Here `default` feature looks like have very less value as compare to others
# so lets drop it

# %%
del df['default']
categorical_features.remove('default')

print(f"categorical features : {categorical_features}")

# %% [markdown]
# ### Convert target feature into numerical from categorical

# %%
df[target] = df[target].replace({'yes': 1, 'no': 0})
numerical_features = numerical_features + [target]
print(f"numerical_features : {numerical_features}")

# %% [markdown]
# ### Find corelation between numerical features:

# %%
corr_matrix = df[numerical_features].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot= True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")

# %% [markdown]
# ### Split Data into Tran/Test/Val

# %%
from sklearn.model_selection import train_test_split

def split_data(df):    
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    print(f"train data size: {len(df_train)}, Test data size: {len(df_test)}, Validation data size: {len(df_val)}")

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    del df_train[target]
    del df_val[target]
    del df_test[target]
    
    return df_train, y_train, df_val, y_val, df_test, y_test

# %%
df_train, y_train, df_val, y_val, df_test, y_test = split_data(df)

# %%
y_train.sum(), y_val.sum(), y_test.sum()

# %%
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
def pre_process_data(df_train, df_val, df_test):
  """
  Preprocess the data including one-hot encoding and standardization.

  Args:
    df_train (pd.DataFrame): Training data.
    df_val (pd.DataFrame): Validation data.
    df_test (pd.DataFrame): Test data.

  Returns:
    Tuple: Tuple containing the DictVectorizer, StandardScaler, and preprocessed data.
  """
  dict_train = df_train.to_dict(orient='records')
  dict_val = df_val.to_dict(orient='records')
  dict_test = df_test.to_dict(orient='records')

  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dict_train)
  X_val = dv.transform(dict_val)
  X_test = dv.transform(dict_test)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  return dv, scaler, X_train, X_val, X_test

# %%
dv, scaler, X_train, X_val, X_test = pre_process_data(df_train, df_val, df_test)

# %% [markdown]
# ### Add One Hot Encoding for categorical features

# %%
def prepare_dictionaries(df: pd.DataFrame):
    dicts = df.to_dict(orient='records')
    return dicts

dict_train = prepare_dictionaries(df_train)
dict_val = prepare_dictionaries(df_val)
dict_test = prepare_dictionaries(df_test)

# %%
pd.DataFrame(X_train).to_csv("./dataset/train.csv")
pd.DataFrame(y_train).to_csv("./dataset/train_label.csv")
pd.DataFrame(X_test).to_csv("./dataset/test.csv")
pd.DataFrame(y_test).to_csv("./dataset/test_label.csv")
pd.DataFrame(X_val).to_csv("./dataset/val.csv")
pd.DataFrame(y_val).to_csv("./dataset/val_label.csv")

# %%
X_train[0]

# %% [markdown]
# ## Import required libraries

# %%
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import xgboost as xgb

# %%
# List of models with their configurations
model_des_list = [
  {
    "name": "LR",
    "model": LogisticRegression(),
    "params": {
      "solver": ['liblinear'],
      "C": [1.0],
      "max_iter": [10, 50, 100]
    }
  },
  {
    "name": "DecisionTreeClassifier",
    "model": DecisionTreeClassifier(),
  },
  {
    "name": "RandomForestClassifier",
    "model": RandomForestClassifier(),
  },
  {
    "name": "XGBClassifier",
    "model": xgb.XGBClassifier(
      objective="binary:logistic",  # For binary classification
      random_state=42,              # Random seed for reproducibility
      n_jobs=-1,                    # Used for parallel processing, It will use all available CPU's
      use_label_encoder=False
    ),
    "params": {
      "n_estimators": [200],         # Number of boosting rounds
      "learning_rate": [0.1],    # Learning rate
      "max_depth": [5],                 # Maximum depth of each tree
    }
  },
  {
    "name": "LDA",
    "model": LinearDiscriminantAnalysis()
  },
  {
    "name": "NB",
    "model": GaussianNB()
  }
]

# %%
from sklearn.model_selection import GridSearchCV
def get_best_params_and_estimator(model_des_list, X_train, y_train, X_val, y_val):
  """
  Get best hyperparameters and estimator by using GridSearchCV

  Args:
    model_des_list: Model information containing model name, model object, set of parameters
    X_train: preprocessed training data set
    y_train: training label
    X_val: preprocessed validation data set
    y_val: validation label

  Returns:
    Tuple: Tuple containing best hyper parameters and best estimators
  """
  model = model_des_list['model']
  model_name = model_des_list['name']
  params = model_des_list.get('params', {})

  grid_search = GridSearchCV(model, params, cv=5)
  grid_search.fit(X_train, y_train)
  best_params = grid_search.best_params_
  best_estimator = grid_search.best_estimator_
  return best_params, best_estimator

# %%
def get_model_evaluation(model, X, y):
  """
  Get model evaluation report, It helps us to understand how our model is behaving

  Args:
    model: Model on which we have to evaluate model accuracy
    X: Dataset on which we will evaluate model accuracy
    y: Actual label (Truth label values)
  """
  y_pred = model.predict(X)
  acc = roc_auc_score(y, y_pred)
  return round(acc, 4)

# %%
from sklearn.metrics import accuracy_score, roc_auc_score
def get_model_reports(model_des_list, X_train, y_train, X_val, y_val):
  """
  Get the model report based on provided multiple model information

  Args:
    model_des_list: Multiple model information
    X_train, y_train: Training dataset
    X_val, y_val: Validatioin dataset

  Returns:
    List: List of model reports containing, model name, model object, accuracy, hyperparameters
  """
  model_reports = []
  for model_des in model_des_list:
    best_params, best_estimator = get_best_params_and_estimator(model_des, X_train, y_train, X_val, y_val)
    train_accuracy = get_model_evaluation(best_estimator, X_train, y_train)
    val_accuracy = get_model_evaluation(best_estimator, X_val, y_val)
    
    print(f"Model: {model_des['name']}, train_accuracy: {train_accuracy}, validation accuracy: {val_accuracy}")
    
    model_report = {
      "name": model_des["name"],
      "model": best_estimator,
      "best_params": best_params,
      "train_accuracy": train_accuracy,
      "val_accuracy": val_accuracy
    }
    model_reports.append(model_report)

  return model_reports

# %%
model_report = get_model_reports(model_des_list, X_train, y_train, X_val, y_val)

# %%
def get_best_model_info(model_reports):
  """
  Get best model based on accuracy

  Args:
    model_report: List of model reports

  Returns:
    best_model: Best model based on model accuracy
  """
  sorted_report = sorted(model_reports, key=lambda x: x['val_accuracy'], reverse=True)
  return sorted_report[0]

# %%
best_model_info = get_best_model_info(model_report)

# %%
def save_artifacts(dictVectorizer, standardScaler, model, model_file):
  """
  Save artifacts, that can be used in web server to generate prediction

  Args:
    dictVectorizer: One hot encoder
    standardScaler: Standard Scaler
    model: Best model
    model_file: File name, in which model should be stored
  """
  pipeline = make_pipeline(dictVectorizer, standardScaler, model)
  with open(model_file,'wb') as f_out: 
    pickle.dump(pipeline, f_out)

# %%
from sklearn.pipeline import make_pipeline
import pickle
best_model = best_model_info["model"]
save_artifacts(dv, scaler, best_model, "./artifacts/model.bin")

# %%
best_model

# %%



