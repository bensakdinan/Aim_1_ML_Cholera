import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # File paths
    cholera_metadata_path = "/Users/bensakdinan/Desktop/prophage_induction/00_metadata.csv"
    cholera_quantitative_data_path = "/Users/bensakdinan/Desktop/prophage_induction/MG_data_NM.csv"
    cholera_propagate_data_path = "/Users/bensakdinan/Desktop/prophage_induction/propagate_results.csv"

    # Reading .csv files
    cholera_metadata = pd.read_csv(cholera_metadata_path)
    cholera_quantitative_data = pd.read_csv(cholera_quantitative_data_path)
    cholera_propagate_data = pd.read_csv(cholera_propagate_data_path)

    # Merging .csv files into a single dataframe
    merged_cholera_data = cholera_metadata.merge(
        cholera_quantitative_data, on="Sample", how="outer").merge(
        cholera_propagate_data, on="Sample", how="outer")

    # Calculating ICP1/2/3:Vc ratios (and handling potential divide by 0)
    merged_cholera_data["ICP1_Vc_ratio"] = np.where(
        merged_cholera_data["Vibrio.cholerae"] == (0 or 0.0), 0,
        merged_cholera_data["ICP1"] / merged_cholera_data["Vibrio.cholerae"]
    )
    merged_cholera_data["ICP2_Vc_ratio"] = np.where(
        merged_cholera_data["Vibrio.cholerae"] == (0 or 0.0), 0,
        merged_cholera_data["ICP2"] / merged_cholera_data["Vibrio.cholerae"]
    )
    merged_cholera_data["ICP3_Vc_ratio"] = np.where(
        merged_cholera_data["Vibrio.cholerae"] == (0 or 0.0), 0,
        merged_cholera_data["ICP3"] / merged_cholera_data["Vibrio.cholerae"]
    )

    # In the original datatable, Mild = 1, Moderate = 2, Moderate = 3
    # Binning Dehydration_Status into Mild (= 0) or Non-Mild (= 1)
    merged_cholera_data['Dehydration_Status_Mild_NonMild'] = 1
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 3]), 'Dehydration_Status_Mild_NonMild'] = 0

    # Binning Dehydration_Status into Severe or Non-Severe
    # Binning Dehydration_Status into Severe (= 2.0) or Non-Severe (= 1.0)
    merged_cholera_data['Dehydration_Status_Severe_NonSevere'] = 1
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 1]), 'Dehydration_Status_Severe_NonSevere'] = 0

    # Replacing 'Nature_of_Stool' string values with float values
    # DataOrganization.convert_nature_of_stool_to_numeric(merged_cholera_data)
    # DataOrganization.convert_area_code_to_numeric(merged_cholera_data)

    # patient_features = ['ICP1_Vc_ratio', 'Area_Code', 'Age_in_Years', 'Nature_of_Stool', 'ICP1', 'Vibrio.cholerae']
    relevant_columns = [
        'Dehydration_Status', 'Dehydration_Status_Mild_NonMild', 'Dehydration_Status_Severe_NonSevere',
        'Age_in_Years', 'Area_Code', 'Vibrio.cholerae',
        'Duration_of_Dirrhoea_in_Hrs', 'ICP1', 'ICP1_Vc_ratio',
        'Nature_of_Stool', 'AZI', 'CIP', 'DOX', 'Active_Prophages'
    ]

    merged_cholera_data = merged_cholera_data[relevant_columns]  # Keep only relevant columns

    # Replace whitespaces in 'Nature_of_Stool' with '_'
    merged_cholera_data['Nature_of_Stool'] = merged_cholera_data['Nature_of_Stool'].replace(' ', '_', regex=True)

    # Filling in missing data points with 0.0
    merged_cholera_data["AZI"] = merged_cholera_data["AZI"].fillna(0.0)
    merged_cholera_data["CIP"] = merged_cholera_data["CIP"].fillna(0.0)
    merged_cholera_data["DOX"] = merged_cholera_data["DOX"].fillna(0.0)

    # Writing to new .csv file for visualization
    merged_cholera_data.to_csv("merged_cholera_data.csv", index=False)

    # Features and prediction target dataframes
    X = merged_cholera_data.drop(columns=['Dehydration_Status',
                                          'Dehydration_Status_Mild_NonMild',
                                          'Dehydration_Status_Severe_NonSevere'])

    # y = merged_cholera_data['Dehydration_Status'].copy()
    y = merged_cholera_data['Dehydration_Status_Mild_NonMild'].copy()  # Uncomment the desired prediction target
    # y = merged_cholera_data['Dehydration_Status_Severe_NonSevere'].copy()
    # print(X.head())
    # print(y.head())

    print(X.dtypes)

    # One-hot encoding my categorical data columns, 'Area_Code' and 'Nature_of_Stool'. Numerical values of
    # 'Area_Code' and 'Nature_of_Stool' have no continuous value, so values must be categorized
    X_encoded = pd.get_dummies(X, columns=['Area_Code', 'Nature_of_Stool'])
    X_encoded.to_csv("X_encoded.csv", index=False)

    # Check that there are only 2 possible classifications, the below array should be [1 0]
    print(y.unique())

    # Splitting data into test/training subsets
    print(sum(y) / len(y))  # 0.110 of y is mild, the training and test subsets should be stratified to this proportion
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=37, stratify=y)
    print(sum(y_train) / len(y_train))  # 0.112 -> Stratification is good
    print(sum(y_test) / len(y_test))  # 0.105 -> Stratification is good

    # Train the xgboost model
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=37)
    clf_xgb.set_params(eval_metric='aucpr', early_stopping_rounds=10)
    clf_xgb.fit(X_train,
                y_train,
                verbose=True,
                eval_set=[(X_test, y_test)])

    # Predict
    y_predictions = clf_xgb.predict(X_test)
    acc_score = accuracy_score(y_test, y_predictions)
    print(f"Percentage of correct predictions: {acc_score}")

    # Plot confusion matrix
    cmatrix = confusion_matrix(y_test, y_predictions)
    plt.figure(figsize=(9,7))
    sns.heatmap(cmatrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Non-Mild", "Mild"],
                yticklabels=["Non-Mild", "Mild"])
    # sns.heatmap(cmatrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Severe", "Non_Severe"],
    #            yticklabels=["Severe", "Non_Severe"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate the weights of each feature to prediction -> Especially ICP1:Vc ratio
