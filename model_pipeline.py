import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse  # For handling command-line arguments

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column):
    # Drop unnecessary columns if any
    df = df.drop(columns=['State', 'Area code', 'Total day charge', 'Total eve charge', 
                          'Total night charge', 'Total intl charge'], errors='ignore')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Encode categorical target if necessary
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    
    # Encode categorical features using OneHotEncoder
    categorical_features = ['International plan', 'Voice mail plan']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    return X, y, scaler  # Return the scaler for later use

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(report)
    
    return model, accuracy, report

def save_model(model, scaler, model_path='trained_model.joblib', scaler_path='scaler.joblib'):
    # Save the model and scaler to files
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def main(args):
    file_path = "/home/amor/ml_project/churn-bigml-80.csv"  # Update with actual file path
    target_column = "Churn"  # Update with actual target column
    
    if args.task == 'preprocess' or args.task == 'all':
        print("Loading and preprocessing data...")
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df, target_column)
        print("Data preprocessing completed.")
        
        # Save the preprocessed data (optional)
        joblib.dump((X, y, scaler), 'preprocessed_data.joblib')
        print("Preprocessed data saved to preprocessed_data.joblib")
    
    if args.task == 'train' or args.task == 'all':
        if args.task == 'train':
            # Load preprocessed data if not already loaded
            try:
                X, y, scaler = joblib.load('preprocessed_data.joblib')
                print("Preprocessed data loaded.")
            except FileNotFoundError:
                print("Preprocessed data not found. Please run preprocessing first.")
                return
        
        print("Training the model...")
        model, accuracy, report = train_model(X, y)
        
        # Save the trained model and scaler
        save_model(model, scaler, model_path='trained_model.joblib', scaler_path='scaler.joblib')
        print("Model training completed.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Model Pipeline")
    parser.add_argument('task', choices=['preprocess', 'train', 'all'], 
                        help="Specify the task: 'preprocess', 'train', or 'all'")
    args = parser.parse_args()
    
    # Execute the pipeline based on the task
    main(args)