import pandas as pd
import os

def load_and_preprocess_data(filepath):
    """
    Loads the life insurance data from a text file and preprocesses it.

    This function is responsible for reading the raw data from the specified
    filepath, assigning meaningful column names, and mapping numerical
    categorical values to more descriptive string labels. This step is crucial
    for making the data human-readable and compatible with the Bayesian model.

    Args:
        filepath (str): The path to the lifeInsurance.txt file.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with meaningful column names
                      and mapped categorical values.
    """
    # Define column names based on the PDF document
    column_names = [
        'Gender', 'Age', 'MaritalStatus', 'Dependents',
        'PhysicalStatus', 'ChronicDiseases', 'MonthlySalary', 'Decision'
    ]

    # Load data from the text file. The data is space-separated.
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=column_names)

    # Map numerical values to meaningful categories
    df['Gender'] = df['Gender'].map({0: 'Female', 1: 'Male'})
    df['MaritalStatus'] = df['MaritalStatus'].map({0: 'Single', 1: 'Married'})
    df['PhysicalStatus'] = df['PhysicalStatus'].map({
        0: 'Sedentary', 1: 'Moderately Active', 2: 'Active'
    })
    df['ChronicDiseases'] = df['ChronicDiseases'].map({
        0: 'No Conditions', 1: 'Moderate', 2: 'Severe'
    })
    df['Dependents'] = df['Dependents'].map({
        0: '0', 1: '1', 2: '2', 3: '>=3'
    })
    df['Decision'] = df['Decision'].map({0: 'Reject', 1: 'Accept'})

    return df

def calculate_risk_score(df):
    """
    Calculates the risk score for each individual based on the rules defined in the PDF.

    This function implements the specific risk scoring logic provided in the project
    documentation. It assigns points based on age, health conditions, financial situation,
    and family responsibilities, summing them up to a total risk score. This score is
    then used to determine a recommended decision for life insurance.

    Args:
        df (pd.DataFrame): The DataFrame containing the life insurance data.

    Returns:
        pd.DataFrame: The DataFrame with an added 'RiskScore' column.
    """
    df['RiskScore'] = 0

    # Age points
    df.loc[df['Age'] < 30, 'RiskScore'] += 5
    df.loc[(df['Age'] >= 30) & (df['Age'] < 40), 'RiskScore'] += 10
    df.loc[(df['Age'] >= 40) & (df['Age'] < 50), 'RiskScore'] += 15
    df.loc[df['Age'] >= 50, 'RiskScore'] += 20

    # Health Condition (Chronic Diseases) points
    df.loc[df['ChronicDiseases'] == 'No Conditions', 'RiskScore'] += 5
    df.loc[df['ChronicDiseases'] == 'Moderate', 'RiskScore'] += 10
    df.loc[df['ChronicDiseases'] == 'Severe', 'RiskScore'] += 15

    # Financial Situation (Monthly Salary) points
    df.loc[df['MonthlySalary'] > 3500, 'RiskScore'] += 5
    df.loc[(df['MonthlySalary'] >= 1700) & (df['MonthlySalary'] <= 3500), 'RiskScore'] += 10
    df.loc[df['MonthlySalary'] < 1700, 'RiskScore'] += 15

    # Family Responsibilities (Dependents) points
    df.loc[df['Dependents'] == '0', 'RiskScore'] += 5
    df.loc[df['Dependents'] == '1', 'RiskScore'] += 10
    df.loc[df['Dependents'] == '2', 'RiskScore'] += 15
    df.loc[df['Dependents'] == '>=3', 'RiskScore'] += 15 # Assuming >=3 also gets 15 points as per PDF

    return df

def create_age_and_salary_bins(df):
    """
    Creates age and salary bins for visualization and analysis.

    This function categorizes continuous 'Age' and 'MonthlySalary' features
    into discrete bins. This binning facilitates better visualization and
    analysis of demographic and financial trends within the dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing the life insurance data.

    Returns:
        pd.DataFrame: The DataFrame with added 'AgeGroup' and 'SalaryGroup' columns.
    """
    # Age groups
    bins_age = [0, 30, 40, 50, 120] # Assuming max age is 120 for practical purposes
    labels_age = ['<30', '30-39', '40-49', '50+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)

    # Salary groups
    bins_salary = [0, 1700, 3500, df['MonthlySalary'].max() + 1] # +1 to include max value
    labels_salary = ['<1700', '1700-3500', '>3500']
    df['SalaryGroup'] = pd.cut(df['MonthlySalary'], bins=bins_salary, labels=labels_salary, right=False)

    return df

if __name__ == '__main__':
    # Example usage:
    # IMPORTANT: Adjust this path to match your local file structure
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the path to the data file relative to the script's directory
    file_path = os.path.join(script_dir, 'data', 'lifeInsurance.txt')
    df_processed = load_and_preprocess_data(file_path)
    df_with_risk = calculate_risk_score(df_processed)
    df_final = create_age_and_salary_bins(df_with_risk)

    print(df_final.head())
    print(df_final.info())
