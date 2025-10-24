"# DeepCSAT---E-commerce-Predict-Customer-Satisfaction-Score" 
## Project Summary -
This project focuses on analyzing and predicting customer satisfaction using a real-world e-commerce support dataset. The goal is to help the company improve its support quality by identifying key operational factors influencing CSAT (Customer Satisfaction Score) and to build a machine learning model that can predict whether a customer will be satisfied based on their support experience.

The dataset contains structured fields such as issue types, response times, agent details, shifts, tenure buckets, product information, and open-text feedback from customers. The analysis starts with data cleaning by removing duplicates, handling null values, and engineering a response_time_min column to calculate the time taken to respond to customer issues.

Exploratory Data Analysis (EDA) is performed using the UBM (Univariate, Bivariate, Multivariate) method:

Univariate analysis explores the distribution of CSAT scores, agent shifts and sub-categories.

Bivariate analysis highlights relationships like CSAT vs. agent shift, tenure bucket, and response time.

Multivariate analysis is performed using a heatmap showing how the combination of shift and tenure impacts average CSAT.

A new binary target column CSAT_High is created to classify customer satisfaction into high (1) and low (0). After preparing the data with one-hot encoding for categorical columns, two machine learning models are trained:

Logistic Regression – to understand linear trends and feature importance.

Random Forest Classifier – to capture non-linear relationships and improve accuracy.

Both models are evaluated using accuracy score and classification reports. Random Forest achieved better results in predicting satisfaction. A confusion matrix and a feature importance plot are also visualized to better understand the model’s behavior.

Additionally, NLP techniques are applied on the Customer Remarks field:

A Word Cloud is generated to highlight frequent complaint terms.

Sentiment analysis using TextBlob helps understand the emotional tone of customer feedback.

Key insights show that longer response times reduce satisfaction, certain product categories and sub-categories are more likely to trigger dissatisfaction, and agent performance varies by shift and tenure. The NLP analysis further validates patterns seen in structured data.

In conclusion, this project successfully combines data cleaning, visual analysis, machine learning, and text mining to improve customer service strategy. It not only helps predict customer satisfaction but also uncovers actionable insights to enhance support operations.
## Visual 1: CSAT Distribution (Pie Chart)
This pie chart shows the proportion of high CSAT scores (≥4) vs low CSAT scores (<4).

CSAT_High = 1 → High satisfaction

CSAT_High = 0 → Low satisfaction

Interpretation:

If one slice is much larger, most customers are either satisfied or unsatisfied.

Helps quickly see overall customer satisfaction in the dataset.

## Visual 2: Feature Importance (Horizontal Bar Chart)
What it shows:
how important each feature is in predicting high CSAT.
### Features could include:
Item_price → Price of purchased item
response_time_min → How fast the agent responded
connected_handling_time → Time spent resolving the issue
Categorical features like category, Agent_name, etc.
### Interpretation:
Longer bars → More influence on CSAT prediction
Helps identify key factors that drive customer satisfaction:
e.g., If response_time_min has a high score, reducing response time could improve CSAT.

Axes:

X-axis: Importance score (0 → 1)
Y-axis: Feature names
## Visual 3: Average CSAT by Product Category (Bar Chart)
What it shows:

Each bar = Average CSAT score for a product category

Helps understand which product lines have higher or lower customer satisfaction.

Interpretation:

Taller bars → higher average satisfaction
Shorter bars → categories that may need quality improvement or better support
Axes:
X-axis: Product categories
Y-axis: Average CSAT score (usually 1–5 scale)
## Visual 4: Monthly Trend (Line Chart)
What it shows:

Trend of average CSAT scores over months

Helps track whether customer satisfaction is improving, declining, or stable over time.

Interpretation:

Upward slope → Improvement in satisfaction

Downward slope → Possible service or product issues during certain months

Peaks and dips → Seasonal patterns or campaign effects

Axes:

X-axis: Month of order

Y-axis: Average CSAT score
## Visual 5: Correlation Heatmap
What it shows:

Measures linear relationship between features (correlation ranges -1 to 1)

1 → perfect positive correlation

0 → no correlation

-1 → perfect negative correlation

Interpretation:

High positive correlation → As one feature increases, CSAT tends to increase

High negative correlation → As one feature increases, CSAT tends to decrease

Helps identify features that might influence CSAT and are good candidates for modeling.

Axes:

Both X and Y axes → Features
Each cell → correlation coefficient
## 🧠 Model Used

I used a Feedforward Deep Neural Network (DNN) — also called a Multilayer Perceptron (MLP) — built in TensorFlow/Keras.

## ⚙️ Why this Model Was Chosen
### 1️⃣ Mixed feature types

 data has:

Text → Customer Remarks

Numeric → Item_price, connected_handling_time

Categorical → channel_name, Agent Shift, etc.

After preprocessing:

TF-IDF encodes text as large numeric vectors.

One-Hot/Scaled numeric features become numeric arrays.
→ The result is a high-dimensional sparse vector.

For this kind of input, a fully connected neural network (DNN) handles sparse numerical input very efficiently — much faster and simpler than RNN/LSTM.

### 2️⃣ Target variable type

Your target is CSAT Score (numeric, continuous).
So this is a regression problem, not classification.
Hence the last layer:
Dense(1, activation='linear')
→ outputs a single continuous value (predicted CSAT score).
### 3️⃣ Model depth and regularization
Two hidden layers (256 and 128) = good capacity to learn nonlinear interactions.
ReLU activation → stable gradients and fast convergence.
Dropout(0.3) → prevents overfitting since dataset likely isn’t huge.
### 4️⃣ Simplicity + Generalization
This DNN is:
✅ Lightweight (few million parameters)
✅ Works well on TF-IDF or tabular features
✅ Easier to deploy in Streamlit or TensorFlow Serving
You don’t need heavy NLP models like BERT unless you want to capture complex language semantics.
