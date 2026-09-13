import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split


class BayesianLifeInsuranceModel:
    """
    A Naive Bayes classifier for life insurance decision prediction.

    This implementation uses Gaussian distributions for continuous variables
    and categorical distributions for discrete variables. The model assumes
    conditional independence between features given the class label.

    Attributes:
        class_priors (Dict[str, float]): Prior probabilities for each class
        feature_stats (Dict): Statistics for each feature by class
        categorical_features (List[str]): List of categorical feature names
        continuous_features (List[str]): List of continuous feature names
        classes (List[str]): List of unique class labels
        is_fitted (bool): Whether the model has been trained

    Mathematical Foundation:
        The model implements Bayes' theorem:
        P(class|features) = P(features|class) * P(class) / P(features)

        For continuous features, assumes Gaussian distribution:
        P(x|class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

        For categorical features, uses frequency-based probability:
        P(x|class) = count(x, class) / count(class)
    """

    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the Bayesian Life Insurance Model.

        Args:
            smoothing (float): Laplace smoothing parameter to handle zero probabilities.
                             Default is 1.0 (Laplace smoothing).

        Note:
            Smoothing is particularly important for categorical variables to avoid
            zero probabilities when encountering unseen feature values during prediction.
        """
        self.smoothing = smoothing
        self.class_priors = {}
        self.feature_stats = {}
        self.categorical_features = []
        self.continuous_features = []
        self.classes = []
        self.is_fitted = False

        # Define feature types based on the life insurance dataset
        self._define_feature_types()

    def _define_feature_types(self) -> None:
        """
        Define which features are categorical and which are continuous.

        This classification is based on the life insurance dataset structure
        as defined in the project requirements. This explicit definition
        ensures that the model correctly applies Gaussian distribution for
        continuous data and frequency-based probabilities for categorical data.
        """
        self.categorical_features = [
            'Gender', 'MaritalStatus', 'Dependents',
            'PhysicalStatus', 'ChronicDiseases'
        ]
        self.continuous_features = ['Age', 'MonthlySalary']

    def _calculate_class_priors(self, y: pd.Series) -> Dict[str, float]:
        """
        Calculate prior probabilities for each class.

        Args:
            y (pd.Series): Target variable containing class labels

        Returns:
            Dict[str, float]: Dictionary mapping class labels to prior probabilities

        Mathematical Formula:
            P(class) = count(class) / total_samples
        """
        class_counts = y.value_counts()
        total_samples = len(y)

        priors = {}
        for class_label in class_counts.index:
            priors[class_label] = class_counts[class_label] / total_samples

        return priors

    def _calculate_gaussian_params(self, feature_values: pd.Series) -> Tuple[float, float]:
        """
        Calculate mean and standard deviation for Gaussian distribution.

        Args:
            feature_values (pd.Series): Values of a continuous feature for a specific class

        Returns:
            Tuple[float, float]: Mean and standard deviation

        Note:
            Adds small epsilon (1e-6) to standard deviation to prevent division by zero
            in cases where all values are identical. This is crucial for numerical stability
            when calculating Gaussian probabilities.
        """
        mean = feature_values.mean()
        std = feature_values.std()

        # Prevent division by zero by adding small epsilon
        if std == 0 or pd.isna(std):
            std = 1e-6

        return mean, std

    def _calculate_categorical_probs(self, feature_values: pd.Series,
                                   class_size: int) -> Dict[str, float]:
        """
        Calculate probabilities for categorical feature values using Laplace smoothing.

        Args:
            feature_values (pd.Series): Values of a categorical feature for a specific class
            class_size (int): Total number of samples in the class

        Returns:
            Dict[str, float]: Dictionary mapping feature values to probabilities

        Mathematical Formula:
            P(feature_value|class) = (count(feature_value, class) + α) / (count(class) + α * |V|)
            where α is the smoothing parameter and |V| is the number of unique values

        Rationale:
            Laplace smoothing (α) is applied to prevent zero probabilities for unseen
            feature values during prediction, which would otherwise lead to a zero
            overall probability for a class. The `unique_values` is ensured to be at
            least 1 to prevent division by zero if a feature has no unique values.
        """
        value_counts = feature_values.value_counts()
        # Ensure unique_values is at least 1 to prevent division by zero
        unique_values = max(1, len(feature_values.unique()))

        probabilities = {}
        for value in feature_values.unique():
            count = value_counts.get(value, 0)
            # Apply Laplace smoothing
            prob = (count + self.smoothing) / (class_size + self.smoothing * unique_values)
            probabilities[value] = prob

        return probabilities

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BayesianLifeInsuranceModel':
        """
        Train the Bayesian model on the provided dataset.

        Args:
            X (pd.DataFrame): Feature matrix with columns matching expected feature names
            y (pd.Series): Target variable with class labels ('Accept' or 'Reject')

        Returns:
            BayesianLifeInsuranceModel: Self reference for method chaining

        Raises:
            ValueError: If required features are missing from the dataset

        Training Process:
            1. Calculate class prior probabilities: Determines the baseline probability of each class.
            2. For each feature and class combination:
               - If continuous: Calculate mean and standard deviation using `_calculate_gaussian_params`.
                 These parameters define the Gaussian distribution for the feature within that class.
               - If categorical: Calculate probability distribution with smoothing using `_calculate_categorical_probs`.
                 This involves counting occurrences and applying Laplace smoothing to handle unseen categories.
            3. Store all statistics for later prediction: These pre-calculated statistics are essential
               for efficient likelihood calculation during the prediction phase.
        """
        # Validate input features
        required_features = self.categorical_features + self.continuous_features
        missing_features = set(required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Store unique classes
        self.classes = list(y.unique())

        # Calculate class priors
        self.class_priors = self._calculate_class_priors(y)

        # Initialize feature statistics storage
        self.feature_stats = {
            'categorical': defaultdict(dict),
            'continuous': defaultdict(dict)
        }

        # Calculate statistics for each class
        for class_label in self.classes:
            class_mask = (y == class_label)
            class_data = X[class_mask]
            class_size = len(class_data)

            # Process categorical features
            for feature in self.categorical_features:
                if feature in X.columns:
                    feature_values = class_data[feature]
                    self.feature_stats['categorical'][feature][class_label] = \
                        self._calculate_categorical_probs(feature_values, class_size)

            # Process continuous features
            for feature in self.continuous_features:
                if feature in X.columns:
                    feature_values = class_data[feature]
                    mean, std = self._calculate_gaussian_params(feature_values)
                    self.feature_stats['continuous'][feature][class_label] = {
                        'mean': mean,
                        'std': std
                    }

        self.is_fitted = True
        return self

    def _gaussian_probability(self, x: float, mean: float, std: float) -> float:
        """
        Calculate probability density for a value under Gaussian distribution.

        Args:
            x (float): Value to calculate probability for
            mean (float): Mean of the distribution
            std (float): Standard deviation of the distribution

        Returns:
            float: Probability density

        Mathematical Formula:
            P(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        if std == 0:
            return 1.0 if x == mean else 1e-10

        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        coefficient = 1 / (math.sqrt(2 * math.pi) * std)

        return coefficient * math.exp(exponent)

    def _predict_single(self, sample: pd.Series) -> Tuple[str, Dict[str, float]]:
        """
        Predict class for a single sample and return class probabilities.

        Args:
            sample (pd.Series): Single sample with feature values

        Returns:
            Tuple[str, Dict[str, float]]: Predicted class and probability scores for all classes

        Prediction Process:
            1. For each class, start with prior probability: This is the initial belief in a class.
            2. Multiply by likelihood of each feature given the class: This updates the belief based on observed features.
            3. Apply log transformation to prevent numerical underflow: Multiplying many small probabilities can lead to
               underflow. Log probabilities are summed instead, which is numerically stable.
            4. Convert back to probabilities and normalize: Exponentiating the log probabilities and normalizing them
               yields the final class probabilities.

        Handling Unseen Values:
            If a categorical feature value is encountered during prediction that was not present in the training data
            for a given class, Laplace smoothing is applied to assign a small, non-zero probability instead of zero.
            This prevents the entire class probability from becoming zero, which would be an undesirable outcome.
        """
        log_probabilities = {}

        for class_label in self.classes:
            # Start with log of prior probability
            log_prob = math.log(self.class_priors[class_label])

            # Add log probabilities for categorical features
            for feature in self.categorical_features:
                if feature in sample.index and feature in self.feature_stats['categorical']:
                    feature_value = sample[feature]
                    feature_probs = self.feature_stats['categorical'][feature][class_label]

                    if feature_value in feature_probs:
                        prob = feature_probs[feature_value]
                    else:
                        # Handle unseen values with smoothing
                        # Ensure unique_values is at least 1 to prevent division by zero
                        unique_values = max(1, len(feature_probs))
                        prob = self.smoothing / (sum(feature_probs.values()) +
                                               self.smoothing * unique_values)

                    log_prob += math.log(max(prob, 1e-10))  # Prevent log(0)

            # Add log probabilities for continuous features
            for feature in self.continuous_features:
                if feature in sample.index and feature in self.feature_stats['continuous']:
                    feature_value = sample[feature]
                    stats = self.feature_stats['continuous'][feature][class_label]

                    prob = self._gaussian_probability(
                        feature_value, stats['mean'], stats['std']
                    )
                    log_prob += math.log(max(prob, 1e-10))  # Prevent log(0)

            log_probabilities[class_label] = log_prob

        # Convert log probabilities back to probabilities
        max_log_prob = max(log_probabilities.values())
        probabilities = {}

        for class_label, log_prob in log_probabilities.items():
            probabilities[class_label] = math.exp(log_prob - max_log_prob)

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}

        # Return class with highest probability
        predicted_class = max(probabilities, key=probabilities.get)

        return predicted_class, probabilities

    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict classes for multiple samples.

        Args:
            X (pd.DataFrame): Feature matrix for prediction

        Returns:
            List[str]: List of predicted class labels

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        predictions = []
        for _, sample in X.iterrows():
            predicted_class, _ = self._predict_single(sample)
            predictions.append(predicted_class)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Predict class probabilities for multiple samples.

        Args:
            X (pd.DataFrame): Feature matrix for prediction

        Returns:
            List[Dict[str, float]]: List of probability dictionaries for each sample

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        probabilities = []
        for _, sample in X.iterrows():
            _, sample_probs = self._predict_single(sample)
            probabilities.append(sample_probs)

        return probabilities

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X (pd.DataFrame): Test feature matrix
            y (pd.Series): True class labels

        Returns:
            Dict[str, float]: Dictionary containing various performance metrics

        Metrics Calculated:
            - Accuracy: (True Positives + True Negatives) / Total Samples
            - Precision (Accept): True Positives (Accept) / (True Positives (Accept) + False Positives (Accept))
            - Recall (Accept): True Positives (Accept) / (True Positives (Accept) + False Negatives (Accept))
            - F1-Score (Accept): 2 * (Precision * Recall) / (Precision + Recall)
            - Precision (Reject): True Positives (Reject) / (True Positives (Reject) + False Positives (Reject))
            - Recall (Reject): True Positives (Reject) / (True Positives (Reject) + False Negatives (Reject))
            - F1-Score (Reject): 2 * (Precision * Recall) / (Precision + Recall)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        y_pred = self.predict(X)

        # Convert to numpy arrays for easier indexing
        y_true_np = y.to_numpy()
        y_pred_np = np.array(y_pred)

        metrics = {}

        # Accuracy
        metrics['accuracy'] = np.mean(y_true_np == y_pred_np)

        # For 'Accept' class
        tp_accept = np.sum((y_true_np == 'Accept') & (y_pred_np == 'Accept'))
        fp_accept = np.sum((y_true_np == 'Reject') & (y_pred_np == 'Accept'))
        fn_accept = np.sum((y_true_np == 'Accept') & (y_pred_np == 'Reject'))

        precision_accept = tp_accept / (tp_accept + fp_accept) if (tp_accept + fp_accept) > 0 else 0
        recall_accept = tp_accept / (tp_accept + fn_accept) if (tp_accept + fn_accept) > 0 else 0
        f1_accept = (2 * precision_accept * recall_accept) / (precision_accept + recall_accept) \
                    if (precision_accept + recall_accept) > 0 else 0

        metrics['precision_accept'] = precision_accept
        metrics['recall_accept'] = recall_accept
        metrics['f1_score_accept'] = f1_accept

        # For 'Reject' class
        tp_reject = np.sum((y_true_np == 'Reject') & (y_pred_np == 'Reject'))
        fp_reject = np.sum((y_true_np == 'Accept') & (y_pred_np == 'Reject'))
        fn_reject = np.sum((y_true_np == 'Reject') & (y_pred_np == 'Accept'))

        precision_reject = tp_reject / (tp_reject + fp_reject) if (tp_reject + fp_reject) > 0 else 0
        recall_reject = tp_reject / (tp_reject + fn_reject) if (tp_reject + fn_reject) > 0 else 0
        f1_reject = (2 * precision_reject * recall_reject) / (precision_reject + recall_reject) \
                    if (precision_reject + recall_reject) > 0 else 0

        metrics['precision_reject'] = precision_reject
        metrics['recall_reject'] = recall_reject
        metrics['f1_score_reject'] = f1_reject

        return metrics


if __name__ == '__main__':
    # Example Usage:
    from data_preprocessing import load_and_preprocess_data
    import os

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the path to the data file relative to the script's directory
    file_path = os.path.join(script_dir, 'data', 'lifeInsurance.txt')

    # Load and preprocess data
    df = load_and_preprocess_data(file_path)

    # Prepare data for the model
    X = df.drop('Decision', axis=1)
    y = df['Decision']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = BayesianLifeInsuranceModel()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\nPredictions for first 5 samples:")
    for i in range(5):
        print(f"Sample {i+1}: Predicted={predictions[i]}, True={y_test.iloc[i]}")

    print("\nProbabilities for first 5 samples:")
    for i in range(5):
        print(f"Sample {i+1}: {probabilities[i]}")

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
