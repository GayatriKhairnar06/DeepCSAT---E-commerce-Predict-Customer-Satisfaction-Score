import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse
import joblib

def preprocess(input_csv='eCommerce_Customer_support_data.csv'):
    df = pd.read_csv(input_csv)

    text_col = 'Customer Remarks'
    target_col = 'CSAT Score'

    # Features: remove irrelevant IDs and target
    drop_cols = ['Unique id', 'Order_id', 'order_date_time',
                 'Issue_reported at', 'issue_responded',
                 'Survey_response_Date']
    features = [c for c in df.columns if c not in drop_cols + [target_col]]

    X = df[features].copy()
    y = df[target_col].astype(float)

    # Separate column types
    numeric_cols = ['Item_price', 'connected_handling_time']
    categorical_cols = [c for c in features if c not in numeric_cols + [text_col]]

    # Fill missing text
    X[text_col] = X[text_col].fillna('')

    # TF-IDF for remarks
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(X[text_col])

    # Scale numeric
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numeric_cols].fillna(0))

    # One-hot encode categoricals
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(X[categorical_cols].fillna('NA'))

    # Combine all features
    X_combined = sparse.hstack([X_text, X_num, X_cat]).tocsr()

    os.makedirs('artifacts', exist_ok=True)
    from scipy.sparse import save_npz
    save_npz('artifacts/X_combined.npz', X_combined)
    np.save('artifacts/y.npy', y)

    joblib.dump(tfidf, 'artifacts/tfidf.joblib')
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    joblib.dump(ohe, 'artifacts/ohe.joblib')
    joblib.dump({
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'text_col': text_col
    }, 'artifacts/schema.joblib')

    print("✅ Preprocessing complete — artifacts saved in /artifacts")

if __name__ == "__main__":
    preprocess('eCommerce_Customer_support_data.csv')
