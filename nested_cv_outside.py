"""
Nested cross-validation for anxiety disorder classification
(week-1 mobile sensing + EMA + demographics).

Preprocessing (RobustScaler per platform, LOF, SVMSMOTE, feature selection)
is applied to the whole dataset before the nested CV.

How to run
----------
Computer:   python nested_cv_outside.py                 -> a window opens to choose the file
            python nested_cv_outside.py path/to/week1.xlsx
Colab:      %run nested_cv_outside.py                   -> an upload button appears
            %run nested_cv_outside.py /content/drive/MyDrive/week1.xlsx
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFE, RFECV, SelectKBest, SelectPercentile, SelectFromModel, VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (make_scorer, accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, recall_score,
                             confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# =====================================================================
# 0. Choose the data file
# =====================================================================
def choose_file():
    # 1) Path given on the command line
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    # 2) Google Colab: upload button
    try:
        from google.colab import files
        print("Choose the week-1 data file (.xlsx or .csv):")
        uploaded = files.upload()
        return list(uploaded.keys())[0]
    except ImportError:
        pass

    # 3) Computer: file picker window
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Choose the week-1 data file",
            filetypes=[("Data files", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        root.destroy()
        if file_path:
            return file_path
    except Exception:
        pass

    # 4) Type the path
    return input("Path to the week-1 data file: ").strip()


def load_data(file_path):
    if file_path.lower().endswith(".csv"):
        df_week1 = pd.read_csv(file_path)
    else:
        df_week1 = pd.read_excel(file_path)
    df = df_week1.drop('week', axis=1)
    print(df.head())
    print(df.columns)
    return df_week1


# =====================================================================
# 1. Data preparation
# =====================================================================
def prepare_data(df_week1):
    df = df_week1
    Label_column_name = 'disorder_name___10'

    label_column = df[Label_column_name]
    print(label_column)

    # Read the Excel file
    week1_data = df_week1

    # Filter by phone type
    week1_ios = week1_data[week1_data['phone'] == 'iOS']
    week1_android = week1_data[week1_data['phone'] == 'Android']

    df_ios = []
    df_android = []
    df_ios = week1_ios
    df_android = week1_android

    ID = ['participantid']

    less_mobile_sensing_features=[
       'total_haversine', 'number_of_stay_points', 'location_entropy', 'normalized_entropy',
        'total_call_duration', 'total_incoming_calls', 'total_outgoing_calls', 'total_screen_time',
        'total_unlocks', 'total_evening_screen_time', 'total_evening_unlocks', 'total_daily_screen_time',
        'total_daily_unlocks', 'total_nightly_screen_time', 'total_nightly_unlocks',
        'total_screen_estimated_sleep_time_in_secs', 'activity_entropy', 'normalized_activity_entropy',
        'percent_time_spent_stationary', 'percent_time_spent_stationary_evening', 'percent_time_spent_stationary_night',
    ]
    ema_time_features= ['ema_3b_time_vigorous_activity', 'ema_4b_time_moderate_activity', 'ema_5b_time_walk']
    dm_features =['age']
    ms_features = []
    ms_features =  less_mobile_sensing_features
    selected_columns_1 = ID+ms_features+ema_time_features+dm_features

    df = []
    df = df_ios

    print(len(selected_columns_1))
    feature_names = [col for col in selected_columns_1 if col in df.columns]
    print(feature_names)
    print(len(feature_names))
    feature_columns_ms_ios = df_ios[feature_names]
    feature_columns_ms_android = df_android[feature_names]
    Label_column_name = 'disorder_name___10'

    label_column_ios = df_ios [Label_column_name]
    label_column_android = df_android [Label_column_name]

    new_df_lessfeatures_ms_ios = pd.concat([feature_columns_ms_ios, label_column_ios], axis=1)
    new_df_lessfeatures_ms_android = pd.concat([feature_columns_ms_android, label_column_android], axis=1)

    print(new_df_lessfeatures_ms_ios.isna().sum())
    print(new_df_lessfeatures_ms_android.isna().sum())

    print(new_df_lessfeatures_ms_ios.isna().sum())
    week1_df_cleaned_ios = new_df_lessfeatures_ms_ios.dropna()
    print(week1_df_cleaned_ios.isna().sum())
    print(week1_df_cleaned_ios.shape)

    print(new_df_lessfeatures_ms_android.isna().sum())
    week1_df_cleaned_android = new_df_lessfeatures_ms_android.dropna()
    print(week1_df_cleaned_android.isna().sum())
    print(week1_df_cleaned_android.shape)

    df_ms_ios = week1_df_cleaned_ios
    df_ms_android = week1_df_cleaned_android

    df1_filtered_ios = df_ms_ios
    df1_filtered_android = df_ms_android

    participant_ids_ios = df1_filtered_ios['participantid'].tolist()
    participant_ids_android = df1_filtered_android['participantid'].tolist()

    # List of columns to one-hot encode
    ema = [
        'ema_2_hangout_with_friends',
        'ema_3a_days_vigorous_activity',
        'ema_4a_days_moderate_activity',
        'ema_5a_days_walk',
        'ema_8_sleep_disruption'
    ]

    dm_features =[ 'sex', 'education', 'sexual_orientation', 'ethnicity' ]

    selected_columns_2 = ['participantid'] + ema

    selected_columns_3 = ['participantid'] + dm_features

    # Filter df_ios to include only participants from participant_ids_ios
    df_filtered_ios_2 = df_ios[df_ios['participantid'].isin(participant_ids_ios)][selected_columns_2]

    df_filtered_android_2 = df_android[df_android['participantid'].isin(participant_ids_android)][selected_columns_2]

    # Filter df_ios to include only participants from participant_ids_ios
    df_filtered_ios_3 = df_ios[df_ios['participantid'].isin(participant_ids_ios)][selected_columns_3]

    df_filtered_android_3 = df_android[df_android['participantid'].isin(participant_ids_android)][selected_columns_3]

    df_filtered_ios_1 =df1_filtered_ios
    df_filtered_android_1 =df1_filtered_android

    # ---------- One-hot encoding (iOS) ----------
    df = df_filtered_ios_2

    # List of columns to one-hot encode
    columns_to_encode = [
        'ema_2_hangout_with_friends',
        'ema_3a_days_vigorous_activity',
        'ema_4a_days_moderate_activity',
        'ema_5a_days_walk',
        'ema_8_sleep_disruption'
    ]

    # Define the full set of expected feature names (manually defined by you)
    expected_features = [
        'ema_2_hangout_with_friends_0.0', 'ema_2_hangout_with_friends_1.0', 'ema_2_hangout_with_friends_2.0', 'ema_2_hangout_with_friends_3.0',
        'ema_2_hangout_with_friends_4.0', 'ema_2_hangout_with_friends_5.0', 'ema_3a_days_vigorous_activity_0.0',
        'ema_3a_days_vigorous_activity_1.0', 'ema_3a_days_vigorous_activity_2.0', 'ema_3a_days_vigorous_activity_3.0',
        'ema_3a_days_vigorous_activity_4.0', 'ema_3a_days_vigorous_activity_5.0', 'ema_3a_days_vigorous_activity_6.0',
        'ema_4a_days_moderate_activity_0.0', 'ema_4a_days_moderate_activity_1.0', 'ema_4a_days_moderate_activity_2.0',
        'ema_4a_days_moderate_activity_3.0', 'ema_4a_days_moderate_activity_4.0', 'ema_4a_days_moderate_activity_5.0',
        'ema_5a_days_walk_0.0', 'ema_5a_days_walk_1.0', 'ema_5a_days_walk_2.0', 'ema_5a_days_walk_3.0',
        'ema_5a_days_walk_4.0', 'ema_5a_days_walk_5.0', 'ema_5a_days_walk_6.0', 'ema_5a_days_walk_7.0',
        'ema_8_sleep_disruption_0.0', 'ema_8_sleep_disruption_1.0', 'ema_8_sleep_disruption_2.0',
        'ema_8_sleep_disruption_3.0', 'ema_8_sleep_disruption_4.0', 'ema_8_sleep_disruption_5.0'
    ]

    # One-hot encoding, but keep 'participantid'
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode, dtype=int)

    # Ensure 'participantid' is included in the encoded DataFrame
    df_encoded['participantid'] = df['participantid']

    # Reindex the columns to ensure all expected features are included
    df_encoded = df_encoded.reindex(columns=['participantid'] + expected_features, fill_value=0)

    # Display the transformed DataFrame
    print(df_encoded.head())

    df_filtered_ios_2 = df_encoded

    # ---------- One-hot encoding (Android) ----------
    df = df_filtered_android_2

    # One-hot encoding, but keep 'participantid'
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode, dtype=int)

    # Ensure 'participantid' is included in the encoded DataFrame
    df_encoded['participantid'] = df['participantid']

    # Reindex the columns to ensure all expected features are included
    df_encoded = df_encoded.reindex(columns=['participantid'] + expected_features, fill_value=0)

    # Display the transformed DataFrame
    print(df_encoded.head())

    df_filtered_android_2 = df_encoded

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    df_filtered_android_1 =df1_filtered_android
    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    df_selected_others_ios = df_combined_ios
    df_selected_others_android = df_combined_android

    print(df_selected_others_ios.isna().sum())
    df_selected_others_cleaned_ios= df_selected_others_ios.dropna()
    print(df_selected_others_cleaned_ios.isna().sum())
    print(df_selected_others_cleaned_ios.shape)

    print(df_selected_others_android.isna().sum())
    df_selected_others_cleaned_android= df_selected_others_android.dropna()
    print(df_selected_others_cleaned_android.isna().sum())
    print(df_selected_others_cleaned_android.shape)

    # ---------- RobustScaler (iOS) ----------
    # Step 1: Select only the numeric columns from df_selected_others_cleaned_ios (excluding 'participantid')
    columns_to_normalize = [col for col in selected_columns_1 if col != 'participantid']
    df_numeric = df_selected_others_cleaned_ios[columns_to_normalize].apply(pd.to_numeric, errors='coerce')

    # Step 2: Remove any rows with NaN values (if any) in the selected columns
    df_numeric = df_numeric.dropna()

    # Step 3: Normalize the selected columns using RobustScaler
    scaler = RobustScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Step 4: Add the 'participantid' back to the normalized DataFrame
    df_normalized_with_participantid = pd.concat([df_selected_others_cleaned_ios[['participantid']].reset_index(drop=True),
                                                  df_normalized.reset_index(drop=True)], axis=1)

    # Display the result
    print(df_normalized_with_participantid.head())

    df_filtered_ios_1 = df_normalized_with_participantid

    # ---------- RobustScaler (Android) ----------
    # Step 1: Select only the numeric columns from df_selected_others_cleaned_android (excluding 'participantid')
    columns_to_normalize = [col for col in selected_columns_1 if col != 'participantid']
    df_numeric = df_selected_others_cleaned_android[columns_to_normalize].apply(pd.to_numeric, errors='coerce')

    # Step 2: Remove any rows with NaN values (if any) in the selected columns
    df_numeric = df_numeric.dropna()

    # Step 3: Normalize the selected columns using RobustScaler
    scaler = RobustScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Step 4: Add the 'participantid' back to the normalized DataFrame
    df_normalized_with_participantid = pd.concat([df_selected_others_cleaned_android[['participantid']].reset_index(drop=True),
                                                  df_normalized.reset_index(drop=True)], axis=1)

    # Display the result
    print(df_normalized_with_participantid.head())

    # Optional: Save result to a new variable
    df_filtered_android_1 = df_normalized_with_participantid

    df_filtered_android_1 = df_normalized_with_participantid

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    # ---------- Demographics encoding (iOS) ----------
    # Assuming df_filtered_ios_3 is already loaded
    df = df_filtered_ios_3.copy()

    # Columns to encode
    columns_to_encode = ['sex', 'education', 'sexual_orientation', 'ethnicity']

    # Define mappings
    mappings = {
        'sex': {'Female': 1, 'Male': 0},
        'education': {
            'Finished university': 4,
            'Further education': 3,
            'Finished high school': 2,
            'Student at university': 1,
            'Currently in high school': 0
        },
        'sexual_orientation': {'Straight': 1, 'Non-Straight': 0},
        'ethnicity': {'White': 1, 'Non-White': 0}
    }

    # Preserve 'participantid'
    participant_id = df[['participantid']]

    # Debug: Print unique values before encoding
    for column in columns_to_encode:
        print(f"Unique values in '{column}' before mapping:", df[column].unique())

    # Apply encoding safely
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()  # Remove spaces
            df[column] = df[column].apply(lambda x: mapping.get(x, -1))  # Use -1 for unknown values

    # Debug: Print unique values after mapping
    for column in columns_to_encode:
        print(f"Unique values in '{column}' after mapping:", df[column].unique())

    # Merge 'participantid' with encoded columns
    df_selected_others_cleaned_ios = pd.concat([participant_id, df[columns_to_encode]], axis=1)

    df_filtered_ios_3 = df_selected_others_cleaned_ios

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    # ---------- Demographics encoding (Android) ----------
    # Assuming df_filtered_ios_3 is already loaded
    df = df_filtered_android_3.copy()

    # Preserve 'participantid'
    participant_id = df[['participantid']]

    # Debug: Print unique values before encoding
    for column in columns_to_encode:
        print(f"Unique values in '{column}' before mapping:", df[column].unique())

    # Apply encoding safely
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()  # Remove spaces
            df[column] = df[column].apply(lambda x: mapping.get(x, -1))  # Use -1 for unknown values

    # Debug: Print unique values after mapping
    for column in columns_to_encode:
        print(f"Unique values in '{column}' after mapping:", df[column].unique())

    # Merge 'participantid' with encoded columns
    df_selected_others_cleaned_android= pd.concat([participant_id, df[columns_to_encode]], axis=1)

    df_filtered_android_3 = df_selected_others_cleaned_android
    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    # ---------- Labels (iOS) ----------
    result_df = []
    # Step 1: Filter df1_filtered_ios to keep only rows where participantid is in df_combined_ios
    result_df_ios = new_df_lessfeatures_ms_ios[new_df_lessfeatures_ms_ios['participantid'].isin(df_combined_ios['participantid'])]

    # Step 2: Select only 'participantid' and 'disorder_name___10' columns
    result_df_ios = result_df_ios[['participantid', 'disorder_name___10']]

    # Step 3: Check the result
    print(result_df_ios.head())

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid'),
            result_df_ios.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    # ---------- Labels (Android) ----------
    result_df = []
    # Step 1: Filter df1_filtered_ios to keep only rows where participantid is in df_combined_ios
    result_df_android = df1_filtered_android[df1_filtered_android['participantid'].isin(df_combined_android['participantid'])]

    # Step 2: Select only 'participantid' and 'disorder_name___10' columns
    result_df_android= result_df_android[['participantid', 'disorder_name___10']]

    # Step 3: Check the result
    print(result_df_android.head())

    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid'),
            result_df_android.set_index('participantid')
        ],
        axis=1
    ).reset_index()  # This will bring 'participantid' back as a column

    df_combined_all = pd.concat([df_combined_ios, df_combined_android], ignore_index=True)

    df_combined_all['disorder_name___10_encoded'] = df_combined_all['disorder_name___10'].apply(lambda x: 1 if x == 'Anxiety disorder' else 0)

    # Print the transformed DataFrame
    print(df_combined_all[['disorder_name___10', 'disorder_name___10_encoded']])

    y=df_combined_all['disorder_name___10_encoded']
    X=df_combined_all.drop(['participantid','disorder_name___10',  'disorder_name___10_encoded'], axis=1)
    class_counts = pd.Series(y).value_counts()
    print(class_counts)

    print(X.shape)

    return X, y


# =====================================================================
# 2. LOF outlier removal
# =====================================================================
def remove_outliers(X, y):
    # List to store sets of outlier indices detected by LOF with different n_neighbors
    outliers_sets = []

    # List of n_neighbors values to try
    neighbors_list = [5, 10, 15, 20, 25, 30]

    for n_neighbors in neighbors_list:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        preds = lof.fit_predict(X)  # 1 = inlier, -1 = outlier

        outliers_idx = np.where(preds == -1)[0]
        outliers_sets.append(set(outliers_idx))

        print(f"n_neighbors={n_neighbors}: Detected {len(outliers_idx)} outliers")

    # Calculate overlap between consecutive sets of outliers
    for i in range(len(outliers_sets) - 1):
        overlap = len(outliers_sets[i].intersection(outliers_sets[i + 1]))
        print(f"Overlap between n_neighbors {neighbors_list[i]} and {neighbors_list[i+1]}: {overlap}")

    # Find outliers detected consistently across all runs (intersection)
    consistent_outliers = set.intersection(*outliers_sets)
    print(f"Consistently identified outliers across all runs: {len(consistent_outliers)}")

    # Create a mask to keep only inliers (non-outliers)
    mask = np.ones(len(X), dtype=bool)
    mask[list(consistent_outliers)] = False

    # Filter the original data to exclude consistent outliers
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"Original dataset size: {len(X)}")
    print(f"Dataset size after removing consistent outliers: {len(X_clean)}")
    print("Class distribution after outlier removal:")
    print(pd.Series(y_clean).value_counts())

    X_final_cleaned = X_clean
    y_final_cleaned = y_clean

    return X_final_cleaned, y_final_cleaned


# =====================================================================
# 3. SVMSMOTE oversampling
# =====================================================================
def oversample(X_final_cleaned, y_final_cleaned):
    feature_names=X_final_cleaned.columns

    print(X_final_cleaned.isna().sum())

    #feature_names = [f'feature_{i}' for i in range(combined_x.shape[1])]  # Create generic feature names
    df_combined = pd.DataFrame(X_final_cleaned, columns=feature_names)
    df_combined['disorder_name___10_encoded'] = y_final_cleaned
    print(df_combined.head())
    print(df_combined.shape)

    y=[]
    X=[]
    X = X_final_cleaned
    y = y_final_cleaned
    oversample = SVMSMOTE( sampling_strategy='auto',  random_state=None, k_neighbors=2, m_neighbors=4, svm_estimator=None, out_step=0.5)
    X_resampled = []
    y_resampled = []
    X_resampled, y_resampled= oversample.fit_resample(X, y)
    class_counts = pd.Series(y_resampled).value_counts()
    print(class_counts)
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    real_data = []
    real_data = resampled_data
    real_data.dropna()
    Samples_N=real_data.shape[0]

    X = X_resampled
    y = y_resampled

    return X, y


# =====================================================================
# 4. Feature selection
# =====================================================================
def select_features(X, y):
    # Custom scorer using F1 macro
    scorer = make_scorer(f1_score, average='macro')

    # Assuming X is your features DataFrame and y is your target variable
    Features_Name = X.columns

    # Store results
    feature_selection_methods = [
        "RFECV", "SelectFromModel", "MutualInfo", "RFE",
        "VarianceThreshold", "SelectKBest", "SelectPercentile", "MutualInfo_classif"
    ]
    results = {}

    # Define RandomForest parameter grid for GridSearchCV
    rf_param_grid = {
        'n_estimators': [10, 20, 30, 40, 50],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt', 'log2'],
    }

    # Loop through feature selection methods
    for method in feature_selection_methods:
        best_score = -np.inf
        best_cv_value = None
        best_n_features = None
        best_features = None
        best_feature_names = None

        for cv_value in range(2, 11):  # Loop from 2 to 10 for cross-validation folds
            cv = StratifiedKFold(n_splits=cv_value)

            # Define feature selection method
            if method == "RFECV":
                selector = RFECV(estimator=RandomForestClassifier(random_state=42), step=1, cv=cv, scoring=scorer, n_jobs=-1)
            elif method == "SelectFromModel":
                selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42))
            elif method in ["MutualInfo", "SelectKBest", "MutualInfo_classif"]:
                selector = SelectKBest(mutual_info_classif, k='all')
            elif method == "RFE":
                selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
            elif method == "SelectPercentile":
                selector = SelectPercentile(mutual_info_classif, percentile=50)
            elif method == "VarianceThreshold":
                selector = VarianceThreshold()

            # Fit the selector
            selector.fit(X, y)
            X_selected = selector.transform(X) if hasattr(selector, 'transform') else X

            # GridSearchCV for RandomForestClassifier hyperparameters
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
            grid_search.fit(X_selected, y)

            # Get the best score from GridSearchCV
            mean_score = grid_search.best_score_

            # Check if this is the best score
            if mean_score > best_score:
                best_score = mean_score
                best_cv_value = cv_value
                best_n_features = X_selected.shape[1]
                best_features = selector.get_support() if hasattr(selector, 'get_support') else None
                best_feature_names = Features_Name[best_features] if best_features is not None else None

        # Store results for this method
        results[method] = {
            'Best CV Value': best_cv_value,
            'Best Number of Features': best_n_features,
            'Best Score': best_score,
            'Best Features': best_feature_names,
            'Best RF Params': grid_search.best_params_  # Store the best RandomForest parameters
        }

    # Convert results into a DataFrame
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values(by="Best Score", ascending=False)

    # Plot results
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(df_results))

    bars = plt.bar(df_results.index, df_results["Best Score"], color=colors)
    plt.xlabel('Feature Selection Method')
    plt.ylabel('Best F1 Macro Score')
    plt.title('Comparison of Feature Selection Methods')
    plt.xticks(rotation=45, ha="right")

    # Annotate bars with best CV values
    for bar, cv in zip(bars, df_results["Best CV Value"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'CV={cv}', ha='center', va='bottom', fontsize=10)

    plt.show()

    # Find the best method
    best_method = df_results.index[0]
    best_cv_value = df_results.loc[best_method, "Best CV Value"]
    best_features = df_results.loc[best_method, "Best Features"]
    best_rf_params = df_results.loc[best_method, "Best RF Params"]

    # Select the best features from X
    X_new = X[best_features] if best_features is not None else X

    print(f"\nBest Method: {best_method}")
    print(f"Best CV Value: {best_cv_value}")
    print(f"Selected Features:\n{best_features}")
    print(f"Best RandomForest Parameters:\n{best_rf_params}")
    print(f"\nX_new (selected features based on best method):\n{X_new.head()}")  # Showing the top 5 rows of X_new

    return X_new, best_cv_value


# =====================================================================
# 5. Classifiers and hyperparameter grids
# =====================================================================
def get_catboost():
    # List of classifiers to evaluate
    classifiers = [
        CatBoostClassifier(verbose=0),
    ]

    # List of hyperparameter grids for each classifier
    param_grids = [
        # CatBoostClassifier
        {
            'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [1, 2, 3, 4, 5],
        },
    ]
    return classifiers, param_grids


def get_xgboost():
    # List of classifiers to evaluate
    classifiers = [
        XGBClassifier(),
    ]

    # List of hyperparameter grids for each classifier
    param_grids = [
        # XGBClassifier
        {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [1, 2, 3, 4, 5],
        },
    ]
    return classifiers, param_grids


def get_random_forest():
    # List of classifiers to evaluate
    classifiers = [
        RandomForestClassifier(),
    ]

    # List of hyperparameter grids for each classifier
    param_grids = [
        # RandomForestClassifier
        {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2'],
        },
    ]
    return classifiers, param_grids


def get_gradient_boosting():
    # List of classifiers to evaluate
    classifiers = [
        GradientBoostingClassifier(),
    ]

    # List of hyperparameter grids for each classifier
    param_grids = [
        # GradientBoostingClassifier
        {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [1, 2, 3, 4, 5],
        },
    ]
    return classifiers, param_grids


def get_extra_trees():
    # List of classifiers to evaluate
    classifiers = [
        ExtraTreesClassifier(),
    ]

    # List of hyperparameter grids for each classifier
    param_grids = [
        # ExtraTreesClassifier
        {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2'],
        },
    ]
    return classifiers, param_grids


# =====================================================================
# 6. Nested cross-validation
# =====================================================================
def run_nested_cv(X, y, classifiers, param_grids, best_cv_value):
    # Outer Cross Validation
    outer_cv = StratifiedKFold(n_splits=best_cv_value, shuffle=True, random_state=42)

    # Initialize lists and dictionaries
    outer_results = []
    best_models_per_fold = {}
    inner_best_models = {}

    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n========= Fold {fold_idx} =========")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=best_cv_value, shuffle=True, random_state=42)

        best_classifier = None
        best_params = None
        best_score = 0.0
        best_model_name = None

        for clf, params in zip(classifiers, param_grids):
            model_name = clf.__class__.__name__
            print(f"Evaluating {model_name}...")

            scoring_metrics = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}
            grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=inner_cv,
                                       scoring=scoring_metrics, refit='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred_inner = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_inner)
            f1 = f1_score(y_test, y_pred_inner, average='weighted')

            inner_best_models[(fold_idx, model_name)] = {
                'model': best_model,
                'params': grid_search.best_params_,
                'f1_score': f1,
                'accuracy': acc
            }

            if acc > best_score:
                best_score = acc
                best_classifier = best_model
                best_model_name = model_name
                best_params = grid_search.best_params_

        if best_classifier is None:
            best_classifier = classifiers[0]
            best_model_name = best_classifier.__class__.__name__
            print("No best model found, using default.")

        print(f"Best Hyperparameters from Inner CV for Fold {fold_idx}: "
              f"{best_model_name} with {best_params}")

        # Outer test evaluation
        y_pred_outer = best_classifier.predict(X_test)
        y_prob_outer = best_classifier.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred_outer)
        f1 = f1_score(y_test, y_pred_outer, average='weighted')
        auc_score = roc_auc_score(y_test, y_prob_outer)
        bal_acc = balanced_accuracy_score(y_test, y_pred_outer)
        sens = recall_score(y_test, y_pred_outer, pos_label=1)

        cm = confusion_matrix(y_test, y_pred_outer)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}, "
              f"Balanced Acc: {bal_acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")

        outer_results.append({
            'fold': fold_idx,
            'model_name': best_model_name,
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc_score,
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec
        })

        best_models_per_fold[fold_idx] = {
            'model': best_classifier,
            'model_name': best_model_name,
            'params': best_params,
            'metrics': outer_results[-1]
        }

        # ROC Curve for the fold
        fpr, tpr, _ = roc_curve(y_test, y_prob_outer)
        plt.plot(fpr, tpr, label=f'Fold {fold_idx} - {best_model_name} (AUC={auc_score:.2f})')

    # Final ROC Plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Best Model in Each Fold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Summary
    print("\n======= Final Summary =======")
    for res in outer_results:
        print(f"Fold {res['fold']} - {res['model_name']}: "
              f"Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}, AUC={res['auc']:.4f}, "
              f"BalAcc={res['balanced_accuracy']:.4f}, Sens={res['sensitivity']:.4f}, Spec={res['specificity']:.4f}")

    # Mean ± Std Summary
    def summarize_metric(metric):
        values = [res[metric] for res in outer_results]
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    print(f"\nOverall Accuracy: {summarize_metric('accuracy')}")
    print(f"Overall F1 Score: {summarize_metric('f1_score')}")
    print(f"Overall AUC: {summarize_metric('auc')}")
    print(f"Overall Balanced Accuracy: {summarize_metric('balanced_accuracy')}")
    print(f"Overall Sensitivity: {summarize_metric('sensitivity')}")
    print(f"Overall Specificity: {summarize_metric('specificity')}")

    return outer_results, best_models_per_fold, inner_best_models


# =====================================================================
# 7. Run everything
# =====================================================================
def main():
    file_path = choose_file()
    print(f"Using data file: {file_path}")
    df_week1 = load_data(file_path)

    # Data preparation
    X, y = prepare_data(df_week1)

    # LOF
    X_final_cleaned, y_final_cleaned = remove_outliers(X, y)

    # SVMSMOTE
    X, y = oversample(X_final_cleaned, y_final_cleaned)

    # Feature selection
    X_new, best_cv_value = select_features(X, y)
    X_selected = X_new
    y_selected = y
    X = X_selected
    y = y_selected

    # Nested CV for each classifier
    for name, get_classifiers in [
        ("CatBoostClassifier", get_catboost),
        ("XGBClassifier", get_xgboost),
        ("RandomForestClassifier", get_random_forest),
        ("GradientBoostingClassifier", get_gradient_boosting),
        ("ExtraTreesClassifier", get_extra_trees),
    ]:
        print(f"\n\n#################### {name} ####################")
        classifiers, param_grids = get_classifiers()
        run_nested_cv(X, y, classifiers, param_grids, best_cv_value)


if __name__ == "__main__":
    main()
