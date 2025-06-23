# Titanic Data Cleaning Script

This script cleans the Titanic dataset to prepare it for machine learning models.

## Steps Performed
1. Load and inspect dataset
2. Handle missing values using median and mode
3. Encode categorical features (`Sex`, `Embarked`)
4. Normalize `Age` and `Fare` using StandardScaler
5. Visualize and remove outliers using boxplots and IQR method

## Input
- File: `Titanic-Dataset.csv`

## Output
- Cleaned DataFrame ready for ML
- Final dataset shape printed
- Boxplots shown before outlier removal
- Output files:
  - `Cleaned_Titanic_Data.csv`
  - `Cleaned_Titanic_Data.xlsx`

## Requirements
See `requirements.txt`

## How to Run
1. Place `Titanic-Dataset.csv` in the same directory.
2. Run the script:
```bash
python data_cleaning.py
