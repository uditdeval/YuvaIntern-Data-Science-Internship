# Week 2 – Exploratory Data Analysis and Visualization

## Dataset
UCI Bike Sharing Dataset (Capital Bikeshare, Washington, D.C., 2011–2012).
`hour.csv` is the primary file (17,379 hourly records); `day.csv` contains 731 daily records.
This is shared-mobility transportation demand data, not freight shipment data.

## Contents
- `Dataset/`: original hourly and daily CSV files and dataset documentation
- `Code/eda_analysis.py`: reproducible EDA script
- `Visualizations/`: ten charts generated from the hourly dataset
- `Output/eda_summary.txt`: calculated summary metrics
- `Report/Week2_Exploratory_Data_Analysis_Report.docx`: Word report

## Run
From this folder:
```bash
pip install -r requirements.txt
python Code/eda_analysis.py
```

## Main finding
The average hourly demand profile has clear commute-related peaks, with the highest average at 17:00. Demand differs by working-day status, season and weather.

## Limitation
Findings apply to the historical Washington, D.C. bike-sharing system and should not be generalized to freight delivery or current operations without additional data.
