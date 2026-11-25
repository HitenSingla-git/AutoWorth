# Viva Preparation & Presentation Speaking Script
**Project:** Machine Learning-Based Used Car Price Prediction System

---

## 1. Project Overview in Simple Language
This project is an intelligent web-based valuation system that predicts the fair resale price (in Indian Rupees) of a used car based on its specifications—such as brand, model, manufacturing year, kilometres driven, fuel type, transmission, engine capacity, mileage, maximum power, seating capacity, and ownership history. Instead of relying on guesswork, it learns pricing patterns from **8,128 real historical listings** from CarDekho (India), compares three regression algorithms, analyzes depreciation trends, explains its predictions using SHAP, and provides an interactive **Streamlit** dashboard.

---

## 2. Complete Machine Learning Workflow
1. **Data Acquisition (`src/data_loader.py`):** Loaded `Car details v3.csv` (8,128 rows, 13 columns) from CarDekho India listings.
2. **Exploratory Data Analysis (`notebooks/01_Exploratory_Data_Analysis.ipynb`):** Inspected data types, missing values (1,100 missing cells across `mileage`, `engine`, `max_power`, `torque`, `seats`), duplicate rows (1,202), and selling price distribution (right-skewed, ₹29,999 to ₹1,00,00,000).
3. **Data Cleaning & Feature Engineering (`src/preprocessing.py`):**
   - Removed 1,202 exact duplicate rows.
   - Filtered 3 extreme odometer anomalies (`km_driven > 500,000` km, up to 2.36M km), leaving **6,923 unique records**.
   - Extracted numeric values from text fields (`mileage` in kmpl/km/kg, `engine` in CC, `max_power` in bhp) and marked invalid zero values / `' bhp'` as missing (`NaN`).
   - Dropped `torque` due to inconsistent multi-unit text formats.
   - Extracted `brand` (32 brands) and `car_model` (210 two-word model groups) from `name`, and computed `vehicle_age = 2020 - year`.
4. **Leak-Free Splitting & Pipeline:**
   - Performed an **80:20 train-test split** (`random_state=42`): **5,538 training rows** and **1,385 test rows**.
   - Built a Scikit-learn `ColumnTransformer`: median imputation + `StandardScaler` for numeric features; mode imputation + `OneHotEncoder(handle_unknown="ignore", min_frequency=10)` for categorical features. Fitted **only** on training data.
5. **Model Training & Comparison (`src/train.py` & `src/evaluation.py`):**
   - **Linear Regression:** R² = `0.7422`, MAE = `₹1,30,989`, RMSE = `₹2,80,911`
   - **Decision Tree Regressor (`max_depth=12, min_samples_leaf=5`):** R² = `0.8448`, MAE = `₹96,237`, RMSE = `₹2,17,932`
   - **Random Forest Regressor (`n_estimators=150, max_depth=18, min_samples_leaf=2`):** R² = `0.9217`, MAE = `₹77,039`, RMSE = `₹1,54,766`
6. **Model Selection & Serialization:** Saved the best-performing pipeline (`Random Forest Regressor`) as `models/best_model.pkl` and metrics/metadata as `models/model_metadata.json`.
7. **Explainability & Dashboard (`src/explainability.py`, `app/app.py`):** Computed Random Forest Gini feature importance, global & local SHAP TreeExplainer explanations, and built a 4-page Streamlit web application.

---

## 3. Key Concepts Explained Clearly

### Regression vs. Classification
- **Regression** predicts a **continuous numerical value** (e.g., selling price = ₹4,85,000).
- **Classification** predicts a **discrete category/label** (e.g., whether a car is "Affordable", "Mid-range", or "Luxury").
- Used-car price prediction is a **supervised regression** problem because `selling_price` is a continuous quantity.

### The Three Algorithms Used
1. **Linear Regression (Baseline):** Fits a linear equation $\hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n$ by minimizing the sum of squared errors (Ordinary Least Squares). It is fast and interpretable, but cannot model complex non-linear interactions (like luxury brand × engine power × age), resulting in a lower R² (`0.7422`).
2. **Decision Tree Regressor:** Recursively splits the feature space into rectangular regions using thresholds (e.g., `max_power <= 110 bhp`, `vehicle_age <= 5`) to minimize Mean Squared Error in leaf nodes. It captures non-linearities (`R² = 0.8448`), though a single tree can have high variance.
3. **Random Forest Regressor:** Builds an ensemble of 150 Decision Trees, each trained on a bootstrap sample of rows and random subsets of features, and averages their predictions. Averaging reduces variance and prevents overfitting, achieving the best performance (`R² = 0.9217`).

### Train-Test Split & Data Leakage
- **Train-Test Split:** Divides data into 80% training (to learn parameters) and 20% testing (to evaluate generalization on unseen cars).
- **Data Leakage:** Occurs if test-set statistics (like medians or one-hot categories) influence preprocessing before splitting. We prevent this by splitting first and fitting `ColumnTransformer` strictly inside `Pipeline.fit(X_train, y_train)`.

### Evaluation Metrics
- **R² Score (Coefficient of Determination):** Fraction of price variance explained by the model (`0.9217` means ~92.2% of price variation is explained). Higher is better.
- **MAE (Mean Absolute Error):** Average absolute rupee error (`₹77,039` for Random Forest). Easy for buyers/sellers to interpret.
- **RMSE (Root Mean Squared Error):** Square root of average squared errors (`₹1,54,766`). Penalizes large errors on expensive cars more heavily than MAE.

### Overfitting & Regularization
- **Overfitting** happens when a model memorizes noise in the training set and performs poorly on unseen test cars.
- We control overfitting by setting `max_depth=12, min_samples_leaf=5` on Decision Tree and `max_depth=18, min_samples_leaf=2` plus ensemble averaging (`n_estimators=150`) on Random Forest.

### Feature Importance & SHAP
- **Feature Importance:** Random Forest Gini importance measures how much each feature reduces node impurity across all trees. Top grouped predictors are `max_power` (~53.4%), `vehicle_age` (~21.4%), `engine` (~11.9%), and `km_driven` (~4.2%).
- **SHAP (SHapley Additive exPlanations):** Based on cooperative game theory (Shapley values), SHAP assigns every feature a fair rupee contribution pushing a prediction above or below the dataset's average prediction. We use `shap.TreeExplainer` on 200 test samples globally and on individual predictions in the UI.

### Vehicle Depreciation Analysis
- **Vehicle Age:** Average resale price drops steeply in the first 8–10 years (from ~₹9.50 Lakhs at 1 year old to ~₹2.74 Lakhs at 10 years old).
- **Kilometres Driven:** Cars under 25,000 km average significantly higher resale values than cars above 1,00,000 km.
- **Fuel & Transmission:** In this dataset, Diesel cars average ~₹6.40 Lakhs vs. Petrol ~₹3.76 Lakhs; Automatic cars average ~₹13.10 Lakhs vs. Manual ~₹4.44 Lakhs (reflecting higher representation of premium/SUV segments in Diesel and Automatic listings).

### Project Limitations
1. Historical dataset snapshot ends in **2020** (vehicle age is calculated relative to 2020), so post-2020 inflation and market shifts are not captured.
2. Listings lack **physical condition** details (accidents, paint quality, service history, tire wear, city/RTO).
3. Luxury cars (>₹25 Lakhs) are sparser in the dataset, leading to higher absolute rupee errors (RMSE > MAE).

---

## 4. Top 20 Viva Questions & Concise Answers

1. **Q: Why is used car price prediction a regression problem rather than classification?**  
   **A:** Because the target variable (`selling_price`) is a continuous numerical amount in INR rather than a finite set of discrete categories.

2. **Q: Where did your dataset come from and how many records does it have?**  
   **A:** It is the real-world CarDekho Indian Used Car Dataset (`Car details v3.csv`, sourced from Kaggle/GitHub). It contains 8,128 raw records and 13 columns; after removing 1,202 exact duplicates and 3 extreme odometer outliers (>500,000 km), 6,923 unique records remain.

3. **Q: Why did you remove 1,202 duplicate rows?**  
   **A:** Exact duplicate listings can appear in both training and test sets, causing data leakage and artificially inflating test accuracy. Removing exact duplicates ensures honest evaluation.

4. **Q: How did you handle text columns like `mileage`, `engine`, and `max_power`?**  
   **A:** We used regular expressions to extract numeric values (removing `kmpl`, `km/kg`, `CC`, `bhp`), converted invalid `' bhp'` strings and zero values to `NaN`, and imputed missing values using the training median.

5. **Q: Why did you drop the `torque` column?**  
   **A:** The `torque` column mixes multiple units (`Nm` and `kgm`) and varying RPM formats, making automated parsing error-prone while its information is already strongly captured by `engine` (CC) and `max_power` (bhp).

6. **Q: How do you compute `vehicle_age`?**  
   **A:** Using `REFERENCE_YEAR - year`, where `REFERENCE_YEAR = 2020` (the maximum manufacturing year in the dataset), clamped at `>= 0`.

7. **Q: What happens if a user enters a car brand or model never seen during training?**  
   **A:** Our `OneHotEncoder` uses `handle_unknown="ignore"`, so unseen categories produce all-zero indicator columns instead of crashing, and `min_frequency=10` groups rare models safely.

8. **Q: Why do you use `StandardScaler` if Random Forest doesn't require feature scaling?**  
   **A:** We share the same `ColumnTransformer` across all three models so they are evaluated fairly on identical preprocessed features. Scaling helps Linear Regression and does not hurt tree models.

9. **Q: How did you prevent data leakage?**  
   **A:** We split the data 80:20 (`random_state=42`) first, and wrapped imputation, scaling, and encoding inside a Scikit-learn `Pipeline` fitted strictly on `X_train`.

10. **Q: Which model performed best and what were its exact metrics?**  
    **A:** Random Forest Regressor performed best with **R² = 0.9217**, **MAE = ₹77,039**, and **RMSE = ₹1,54,766** on the 1,385-car test set.

11. **Q: Why did Random Forest outperform Linear Regression (`R² = 0.7422`)?**  
    **A:** Car depreciation and pricing are non-linear (e.g., age interacts differently with luxury vs. budget brands and engine power). Linear Regression assumes a single straight-line relationship, whereas Random Forest captures non-linear interactions.

12. **Q: Why is RMSE (`₹1,54,766`) larger than MAE (`₹77,039`)?**  
    **A:** RMSE squares errors before averaging, so a small number of high-priced luxury cars with larger absolute rupee errors increase RMSE more than MAE.

13. **Q: How did you prevent the Decision Tree from overfitting?**  
    **A:** By constraining tree complexity with `max_depth=12` and `min_samples_leaf=5`, and by using Random Forest bagging across 150 trees (`max_depth=18, min_samples_leaf=2`).

14. **Q: What is the single most important feature in your trained model?**  
    **A:** `max_power` (bhp) is the most influential predictor (~53.4% Gini importance and ~₹1.90 Lakhs mean |SHAP| impact), followed by `vehicle_age` (~21.4% Gini importance).

15. **Q: What is the difference between Gini Feature Importance and SHAP?**  
    **A:** Gini importance gives only a single global ranking based on split impurity reduction, whereas SHAP (using Shapley values) shows both the magnitude and direction (+ or - INR) of each feature for both global summaries and individual car predictions.

16. **Q: Why did you limit global SHAP computation to 200 test samples?**  
    **A:** Computing exact TreeExplainer values across all 1,385 test samples and 100+ one-hot columns is slower; a 200-sample random subset gives stable global explanations in just a couple of seconds.

17. **Q: Does your depreciation chart prove that Diesel causes higher resale prices?**  
    **A:** No—correlation in historical data does not prove causation. Diesel cars in this dataset have higher average prices partly because larger SUVs and luxury sedans are disproportionately Diesel-powered.

18. **Q: How do you ensure the model never outputs negative prices?**  
    **A:** Random Forest averages positive training prices, so its outputs are naturally non-negative; additionally, `predict_car_price` clamps predictions with `max(0.0, raw_pred)` and validates finiteness.

19. **Q: How is the trained model served in the Streamlit web app?**  
    **A:** `src/train.py` serializes the entire preprocessing + Random Forest `Pipeline` to `models/best_model.pkl` via `joblib`, and `app/app.py` loads it once using `@st.cache_resource` for fast live predictions.

20. **Q: What would you improve in future versions?**  
    **A:** Scrape fresh 2024–2026 Indian market listings, add city/RTO location and service-record features, test gradient boosting models (XGBoost/LightGBM), and predict a confidence interval alongside the point estimate.

---

## 5. 5–7 Minute Presentation Speaking Script

- **Slide 1 — Title (~20 sec):** "Good morning/afternoon everyone. Today I am presenting our Machine Learning project, *Machine Learning-Based Used Car Price Prediction System*, which predicts fair resale prices for Indian used cars and explains the key drivers behind every valuation."
- **Slide 2 — Introduction (~35 sec):** "India's pre-owned car market is growing rapidly, yet pricing remains opaque. Buyers risk overpaying while sellers risk underpricing due to subjective dealer quotes. Machine learning lets us estimate fair market value objectively from historical vehicle data."
- **Slide 3 — Problem Statement & Objectives (~40 sec):** "Our goal was to build an end-to-end, understandable system that: (1) predicts resale value from car specifications, (2) compares three regression algorithms—Linear Regression, Decision Tree, and Random Forest, (3) analyzes vehicle depreciation, (4) adds Explainable AI using SHAP, and (5) delivers everything through a clean Streamlit dashboard."
- **Slide 4 — Dataset & Features (~40 sec):** "We used the real-world CarDekho India dataset containing 8,128 vehicle listings and 13 columns—including brand/model, year, kilometres driven, fuel type, transmission, seller type, ownership, mileage, engine CC, max power, and seats. Target prices range from ₹30,000 up to ₹1 Crore, with a median of ₹4.5 Lakhs in raw data."
- **Slide 5 — Methodology (~45 sec):** "To ensure clean, leak-free modeling, we removed 1,202 exact duplicate listings and 3 extreme odometer anomalies above 500,000 km, leaving 6,923 unique cars. We parsed numeric units from mileage, engine, and power, derived vehicle age relative to 2020, and split 80:20 into 5,538 training and 1,385 test records. All median imputation, standard scaling, and OneHotEncoding run inside a Scikit-learn Pipeline fitted only on training data."
- **Slide 6 — Machine Learning Algorithms (~45 sec):** "We trained three models on the exact same split: Linear Regression as our linear baseline, a depth-limited Decision Tree to capture non-linear pricing thresholds, and a 150-tree Random Forest ensemble to reduce variance and prevent overfitting."
- **Slide 7 — Results & Model Comparison (~50 sec):** "Here are the actual test-set results: Linear Regression achieved R² of 0.7422 with MAE of ₹1.31 Lakhs; Decision Tree improved R² to 0.8448 with MAE of ₹96,237; and Random Forest Regressor clearly won with **R² = 0.9217**, **MAE = ₹77,039**, and **RMSE = ₹1,54,766**—explaining over 92% of price variance."
- **Slide 8 — Explainability & Depreciation (~50 sec):** "Using Random Forest importance and SHAP TreeExplainer, we found that maximum power (bhp) and vehicle age are the two dominant price drivers, followed by engine capacity and kilometres driven. Our depreciation charts clearly show steep value drops during the first 8–10 years and lower prices for higher odometer readings."
- **Slide 9 — Application Demonstration (~45 sec):** "Finally, we built the *USED CAR PRICE PREDICTOR* web dashboard in Streamlit with four sections: Home Overview, Live Price Prediction with local SHAP factor breakdown, Model Evaluation, and Depreciation & SHAP Analytics."
- **Slide 10 — Conclusion & Future Scope (~30 sec):** "In conclusion, Random Forest combined with clean Scikit-learn pipelines and SHAP provides accurate, transparent valuations. In the future, we can incorporate recent post-2020 market listings, city-level pricing, and vehicle condition scores. Thank you, and I welcome your questions."
