# Libraries:
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    print('***Initializing the Experiment***')
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment(experiment_name='Titanic mlflow test_DecisionTreeClassifier')
    mlflow.set_experiment(experiment_name='Titanic_mlflow_test')

    mlflow.autolog()  ##record automatically

    print('**Loading the data**')

    titanic_df = pd.read_csv("Titanic.csv")

    print(titanic_df.isnull().sum()) # Checking whether any null values in the dataset.

    # As the titanic_df['Age'], titanic_df['cabin'] & titanic_df['Embarked'] has null values.
    # replacing null values of titanic_df['Age'] & titanic_df['Embarked'] with median and mode values
    titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df.Age.median())

    print(titanic_df['Embarked'].value_counts())

    #  imputing the null values with the most frequent values in the Embarked column.

    titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
    print(titanic_df.isnull().sum())

    fig = sns.barplot(titanic_df['Pclass'], titanic_df['Survived'], data=titanic_df)
    # Saving the Seaborn Figure:
    fig.figure.savefig("Pclass_Survived_barplot.png")
    mlflow.log_artifact("Pclass_Survived_barplot.png")

    fig1 = sns.barplot(titanic_df['Embarked'], titanic_df['Survived'], data=titanic_df)
    # Saving the Seaborn Figure:
    fig1.figure.savefig("Embarked_Survived_barplot.png")
    mlflow.log_artifact("Embarked_Survived_barplot.png")

    # relationship between Age and Survival
    g = sns.FacetGrid(titanic_df, col="Survived")
    g.map_dataframe(sns.histplot, x='Age')
    g.savefig("Age_Survival.png")
    mlflow.log_artifact("Age_Survival.png")

    print(titanic_df['Survived'].value_counts())
    log_param("Value counts", titanic_df['Survived'].value_counts())

    # Converting Sex column & Embarked column value to Catagorical Value:
    le = LabelEncoder()
    titanic_df['Embarked'] = le.fit_transform(titanic_df['Embarked'])
    print(titanic_df['Embarked'].value_counts())
    titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])
    print(titanic_df['Sex'].value_counts())

    # As a part of data cleaning we have to drop the column like PassengerId, Name, Ticket, Cabin
    # inorder to increase the overall accuracy and efficiency of the data.

    titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    print(titanic_df.head())
    X_train, X_test, y_train, y_test = train_test_split(titanic_df.drop('Survived', axis=1), titanic_df['Survived'],
                                                        test_size=.2,
                                                        random_state=20)
    print(X_train.shape, X_test.shape)
    log_param("Train Shape", X_train.shape)

    # Dt_model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 100, min_samples_leaf=20)
    # Dt_model.fit(X_train, y_train)


    rf_model = RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=200,
                                      min_samples_leaf=20, random_state=20, )

    rf_model.fit(X_train, y_train)

    print("*****Model trained*****")
    y_pred = rf_model.predict(X_test)

    train_accuracy = rf_model.score(X_train,y_train) #performence on training set
    print("train accuracy: ", train_accuracy)
    test_accuracy = rf_model.score(X_test, y_test) #performence on test set
    print("test accuracy: ", test_accuracy)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred) #Accuracy Score
    p_score = precision_score(y_true=y_test, y_pred=y_pred) #Precision Score
    recall = recall_score(y_true=y_test, y_pred=y_pred) #Recall Score
    f1 = f1_score(y_true=y_test, y_pred=y_pred) #f1 Score

    print('Precision: %.3f' % p_score)
    print('Recall: %.3f' % recall)
    print('Accuracy: %.3f' % accuracy)
    print('F1 Score: %.3f' % f1)
    # log_metric("Accuracy for this run", test_accuracy)

    # Log in mlflow (metrics)
    log_metric("Precision Score", p_score)
    log_metric("Recall Score", recall)
    log_metric("Accuracy Score", accuracy)
    log_metric("F1 Score", f1)


    mlflow.sklearn.log_model(rf_model, "RF Model1")
    print(mlflow.active_run().info.run_uuid)

















