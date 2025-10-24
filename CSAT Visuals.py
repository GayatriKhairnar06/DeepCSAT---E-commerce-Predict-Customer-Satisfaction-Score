import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load & clean dataset
# -----------------------------
df = pd.read_csv("eCommerce_Customer_support_data.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()
print("Cleaned Columns:", df.columns.tolist())

# -----------------------------
# Fix date parsing
# -----------------------------
date_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# -----------------------------
# Create response_time_min
# -----------------------------
if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
    df['response_time_min'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
else:
    print("⚠️ Missing timestamp columns, cannot create response_time_min")

# -----------------------------
# Create CSAT_High (target)
# -----------------------------
if 'CSAT Score' in df.columns:
    df['CSAT_High'] = df['CSAT Score'].apply(lambda x: 1 if x >= 4 else 0)
else:
    print("⚠️ Missing CSAT Score column")

# -----------------------------
# Encode categorical variables
# -----------------------------
label_cols = ['category', 'Product_category', 'Agent_name', 'Manager', 'Agent Shift']
for col in label_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# -----------------------------
# Prepare data for feature importance
# -----------------------------
feature_cols = ['Item_price', 'response_time_min', 'connected_handling_time'] + label_cols
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df['CSAT_High']

# -----------------------------
# Visual 1: CSAT Distribution
# -----------------------------
plt.figure(figsize=(6,6))
df['CSAT_High'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen','lightcoral'])
plt.title('1.Customer Satisfaction Distribution')
plt.ylabel('')
plt.show()

# -----------------------------
# Visual 2: Feature Importance
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8,6))
importances.tail(10).plot(kind='barh', color='skyblue')
plt.title('2.Top Factors Influencing Customer Satisfaction')
plt.xlabel('Importance Score')
plt.show()

# -----------------------------
# Visual 3: Average CSAT by Product Category
# -----------------------------
if 'Product_category' in df.columns:
    avg_scores = df.groupby('Product_category')['CSAT Score'].mean().sort_values()
    plt.figure(figsize=(8,5))
    avg_scores.plot(kind='bar', color='teal')
    plt.title('3.Average CSAT Score by Product Category')
    plt.ylabel('Average CSAT Score')
    plt.show()

# -----------------------------
# Visual 4: Monthly Trend
# -----------------------------
if 'order_date_time' in df.columns:
    df['Month'] = df['order_date_time'].dt.to_period('M')
    trend = df.groupby('Month')['CSAT Score'].mean()
    plt.figure(figsize=(10,5))
    trend.plot(marker='o')
    plt.title('4.Monthly Customer Satisfaction Trend')
    plt.ylabel('Average CSAT Score')
    plt.show()

# -----------------------------
# Visual 5: Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df[['CSAT Score','Item_price','response_time_min','connected_handling_time']].corr(), annot=True, cmap='coolwarm')
plt.title('5.Correlation Heatmap of Key Features')
plt.show()
