import pandas as pd
# import scikitplot
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

    # Calculating ICP1/2/3:Vc ratios
    merged_cholera_data["ICP1_Vc_ratio"] = (merged_cholera_data["ICP1"]/
                                            merged_cholera_data["Vibrio.cholerae"])
    merged_cholera_data["ICP2_Vc_ratio"] = (merged_cholera_data["ICP2"] /
                                            merged_cholera_data["Vibrio.cholerae"])
    merged_cholera_data["ICP3_Vc_ratio"] = (merged_cholera_data["ICP3"] /
                                            merged_cholera_data["Vibrio.cholerae"])

    # Binning Dehydration_Status into Mild or Non-Mild
    merged_cholera_data['Dehydration_Status_Mild_NonMild'] = 'Mild'
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 3]), 'Dehydration_Status_Mild_NonMild'] = 'Non_Mild'

    # Binning Dehydration_Status into Severe or Non-Severe
    merged_cholera_data['Dehydration_Status_Severe_NonSevere'] = 'Severe'
    merged_cholera_data.loc[merged_cholera_data['Dehydration_Status'].isin(
        [2, 1]), 'Dehydration_Status_Severe_NonSevere'] = 'Non_Severe'

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
