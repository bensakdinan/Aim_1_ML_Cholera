import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import DataOrganization

if __name__ == "__main__":

    # Path names
    cholera_metadata_path = "/Users/bensakdinan/Desktop/prophage_induction/00_metadata.csv"
    cholera_quantitative_data_path = "/Users/bensakdinan/Desktop/prophage_induction/MG_data_NM.csv"
    cholera_propagate_data_path = "/Users/bensakdinan/Desktop/prophage_induction/propagate_results.csv"

    # Reading .csv files
    cholera_metadata = pd.read_csv(cholera_metadata_path)
    cholera_quantitative_data = pd.read_csv(cholera_quantitative_data_path)
    cholera_propagate_data = pd.read_csv(cholera_propagate_data_path)

    # print(cholera_metadata)
    # print(cholera_metadata.describe())
    # print(cholera_quantitative_data)
    # print(cholera_quantitative_data.describe())
    # print(cholera_propagate_data)
    # print(cholera_propagate_data.describe())

    # Merging .csv files into a single dataframe
    merged_cholera_data = cholera_metadata.merge(
        cholera_quantitative_data, on="Sample", how="outer").merge(
        cholera_propagate_data, on="Sample", how="outer")
    # print(merged_cholera_data)
    merged_cholera_data = merged_cholera_data.fillna(0)

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
    # Binning Dehydration_Status into Mild (= 1.0) or Non-Mild (= 2.0)
    merged_cholera_data['Dehydration_Status_Mild_NonMild'] = 1.0
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 3]), 'Dehydration_Status_Mild_NonMild'] = 2.0

    # Binning Dehydration_Status into Severe or Non-Severe
    # Binning Dehydration_Status into Severe (= 2.0) or Non-Severe (= 1.0)
    merged_cholera_data['Dehydration_Status_Severe_NonSevere'] = 2.0
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 1]), 'Dehydration_Status_Severe_NonSevere'] = 1.0

    # Replacing 'Nature_of_Stool' string values with float values
    DataOrganization.convert_str_to_numeric(merged_cholera_data)

    merged_cholera_data.to_csv("merged_cholera_data.csv", index=False)

    # Prediction target data is Dehydration_Status
    # But I will try binning by Mild/Non-Mild and Severe/Non-Severe
    # Comment out the binning scheme I'm not using
    y = merged_cholera_data.Dehydration_Status_Mild_NonMild # Mild/Non_Mild
    # y = merged_cholera_data.Dehydration_Status_Severe_NonSevere # Severe/Non_Severe

    # Features/Predictors
    print(merged_cholera_data.columns)
    patient_features = ['Age_in_Years', 'Vibrio.cholerae', 'Duration_of_Dirrhoea_in_Days',
                        'Duration_of_Dirrhoea_in_Hrs', 'ICP1', 'ICP1_Vc_ratio',
                        'Nature_of_Stool', 'AZI', 'CIP', 'DOX', 'Active_Prophages']
                        # Try without Active_Prophages to see if it improves prediction
    X = merged_cholera_data[patient_features]

    # Splitting data in training and testing subsets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # Training/fitting the model
    cholera_rf_model = RandomForestClassifier(random_state=0)
    cholera_rf_model.fit(train_X, train_y)

    # Predictions
    cholera_rf_predictions = cholera_rf_model.predict(val_X)
    cholera_rf_mae = mean_absolute_error(val_y, cholera_rf_predictions)
    print(f"Mean absolute error for predictions in cholera_rf_model: {cholera_rf_mae}")
