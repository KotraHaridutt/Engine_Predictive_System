# ✈️ Turbofan Engine Predictive Maintenance Dashboard

[(https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-APP-URL-GOES-HERE.streamlit.app/)

This project is an end-to-end data science application that predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. It features a live, interactive dashboard for fleet monitoring, maintenance recommendations, and model-driven insights.

---

## 🚀 Key Features

* **Fleet-Wide Monitoring:** The sidebar features a "Red Alert" list that automatically identifies engines requiring immediate maintenance, allowing managers to prioritize action.
* **Real-Time RUL Prediction:** Select any engine to see its live predicted Remaining Useful Life (RUL) in cycles, calculated by our V2 tuned Random Forest model.
* **Actionable Business Logic:** The dashboard provides clear, three-tiered maintenance recommendations (🟢 Green, 🟡 Yellow, 🔴 Red) based on a cost-benefit analysis that balances failure risk (`~$2M`) against maintenance cost (`~$250k`).
* **Model Explainability (XAI):** A "Model's Top 5 Predictive Features" chart explains *what* the model found most important for making predictions across the fleet.
* **Sensor Degradation Analysis:** Visualizes the degradation trends of the top 4 most critical sensors for any selected engine.

---

## 🛠️ Tech Stack

* **Data Analysis:** Python, Pandas, NumPy, SQL (SQLite)
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, RandomizedSearchCV)
* **Dashboard & Deployment:** Streamlit, Streamlit Community Cloud
* **Data Versioning:** Git & Git LFS (for handling the >100MB model file)
* **Data:** NASA C-MAPSS (FD001 dataset)

---

## 📈 Data & Modeling Pipeline

This project was built in a clear, phased approach:

1.  **Phase 1: Data Preparation:**
    * Loaded the raw `FD001` text files into Pandas.
    * Cleaned the data by dropping constant-value sensors.
    * **Engineered the `RUL` (Remaining Useful Life) target variable** by calculating the time-to-failure for each engine in the training set.
    * Stored the final, cleaned dataset in a persistent **SQLite database** (`turbofan.db`).

2.  **Phase 2: Exploratory Data Analysis (EDA):**
    * Used SQL queries to analyze engine lifespan distributions.
    * Built correlation heatmaps to identify the **most predictive sensors** (e.g., `sensor_11`, `sensor_4`, `sensor_7`, `sensor_12`).

3.  **Phase 3: V1 Model (Baseline):**
    * Engineered time-series features (e.g., `sensor_4_avg`, `sensor_11_std`) using a 5-cycle rolling window.
    * Trained a baseline `RandomForestRegressor`, achieving **

4.  **Phase 4: V2 Model (Tuning):**
    * Used `RandomizedSearchCV` to tune the model's hyperparameters.
    * This improved the final model performance, creating our V2 production model.

---

## 📊 V2 Model Performance

* **R-squared (R²):** **0.62**
* **Root Mean Squared Error (RMSE):** **47 cycles**

This improved RMSE allows for a more precise maintenance threshold. The app's business logic (`get_recommendation`) was updated to use this new, more accurate `rmse=47` value for its "Red Alert" buffer.

---

## 🔧 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KotraHaridutt/Engine_Predictive_System.git](https://github.com/KotraHaridutt/Engine_Predictive_System.git)
    cd Engine_Predictive_System
    ```

2.  **Install Git LFS** (required to download the model file):
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 💡 Future Improvements

* **V3 Model (LSTM):** Experiment with a Long Short-Term Memory (LSTM) neural network, which is specifically designed for time-series data, to potentially achieve a significant boost in R² score.
* **Fleet-Wide Visuals:** Add a histogram to the dashboard showing the RUL distribution of the entire 100-engine fleet, color-coded by alert status.
* **Cost Calculator:** Implement the "Cost Avoidance Calculator" to show the total dollar amount saved by servicing the red-alert engines.