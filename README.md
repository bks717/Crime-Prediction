AI Crime Risk Forecasting & Mapping System (India)

This project builds an AI-based decision support system that predicts future IPC crime trends for Indian States/UTs using official data from the National Crime Records Bureau (NCRB) under the Indian Penal Code (IPC).

The system not only predicts crime counts for 2023, but also visualizes state-wise risk on an India map for intuitive understanding and proactive governance.

Dataset

Source: NCRB – State/UT-wise IPC Crimes (2020–2022)

Format: CSV

Columns used: State/UT, 2020, 2021, 2022

Total rows containing summary totals are removed during cleaning.

What the Project Does

Cleans NCRB dataset and reshapes it into:

State | Year | Crime_Count


Encodes State names for ML processing.

Trains a Random Forest Regressor using:

Training: 2020, 2021

Testing: 2022

Evaluates model using:

R² Score

MAE

RMSE

Predicts IPC crime counts for 2023 for all States/UTs.

Generates a Crime Risk Map of India using GeoPandas.

Model Used

Random Forest Regressor (sklearn)

Learns trend patterns across years and states

Predicts next year’s crime load based on historical data

Output

Console table of predicted 2023 crime counts

Model evaluation metrics

Choropleth India heatmap showing predicted crime intensity

Side panel listing state-wise predictions

Impact (Atmanirbhar Bharat)

This system demonstrates how Indian government data can be used to build indigenous AI tools for:

Crime trend forecasting

Risk identification

Resource planning for authorities

Proactive governance without relying on foreign analytics tools

How to Run

Place NCRB_Table_1A.1.csv in the same folder.

Run:

python crime_prediction.py


View metrics, predictions, and the India crime risk map.
<img width="1911" height="967" alt="image" src="https://github.com/user-attachments/assets/efb8dc3c-c390-49ca-9453-e6ae238dcf24" />

