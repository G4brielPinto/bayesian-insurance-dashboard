# Bayesian Insurance Dashboard

An academic Streamlit dashboard that explores a rule-based risk score and a Naïve Bayes classifier
for a life-insurance decision dataset. It provides interactive filters, descriptive visualisations,
model evaluation and a single-record prediction form.

## Important scope

This is an academic demonstration of probabilistic classification. It is not clinical, actuarial,
financial or insurance advice, and it must not be used to make real-world insurance decisions.

The source dataset and the internal academic report are intentionally not included. This repository
contains no personal records.

## Features

- data preprocessing and categorical-value mapping;
- an interpretable risk-score calculation;
- a Naïve Bayes model implemented from first principles;
- holdout evaluation in the Streamlit interface;
- interactive exploratory charts and filters;
- prediction for an entered, hypothetical record.

## Project structure

~~~text
.
├── bayesian_model.py
├── data_preprocessing.py
├── streamlit_app.py
└── requirements.txt
~~~

## Setup and launch

Use Python 3.10 or newer in an isolated virtual environment.

~~~bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
~~~

The browser interface opens automatically. To run the application, provide a local data file at
data/lifeInsurance.txt. That directory is ignored by Git and remains on your machine.

## Local data contract

The application expects a whitespace-separated text file with no header and these eight columns
in this exact order:

1. Gender: 0 = Female, 1 = Male
2. Age: integer
3. MaritalStatus: 0 = Single, 1 = Married
4. Dependents: 0, 1, 2 or 3 (where 3 represents 3 or more)
5. PhysicalStatus: 0 = Sedentary, 1 = Moderately Active, 2 = Active
6. ChronicDiseases: 0 = No Conditions, 1 = Moderate, 2 = Severe
7. MonthlySalary: numeric value
8. Decision: 0 = Reject, 1 = Accept

Do not commit personal, sensitive or proprietary data to this repository.

## Reproducibility notes

- The supplied interface uses a 70/30 train-test split with a fixed random seed.
- The dashboard trains its model locally from the input file.
- Metrics and historical report results are intentionally not published as new benchmark claims.

## Privacy and sharing

The repository is initially private while it is reviewed for a future portfolio release. It excludes
data, reports, local paths, credentials and personal contact details.
