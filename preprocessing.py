import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file):
    """
    Loads a financial dataset, performs basic preprocessing, 
    and returns scaled train/test splits.
    """
    df = pd.read_csv(file)
    
    # Simple preprocessing: drop rows with missing values
    df = df.dropna()
    
    # Identify the target column. 
    # If 'Target' doesn't exist, we assume the last column is the label.
    if 'Target' in df.columns:
        y = df['Target']
        X = df.drop('Target', axis=1)
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
    # Keep only numeric columns for simplicity in this example
    X = X.select_dtypes(include=['number'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (important for models like SVM/Quantum SVC)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test