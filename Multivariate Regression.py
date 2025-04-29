# -------------------------------------------- IMPORT PACKAGES AND DATASETS --------------------------------------------
#region

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Import dataset
df = pd.read_csv('Dataset.csv')
df_wl_raw = pd.read_csv('Datasets/water-levels.csv')
df_cd = pd.read_csv('Datasets/climate-daily.csv')
df_ss = pd.read_csv('Datasets/Sunrise and Sunset.csv')

#endregion

# ----------------------------------------- ADJUST COLUMNS IN DETECTION DATASET ----------------------------------------
#region

# Create cardinal direction column
df['Cardinal Direction'] = df['Location'].replace({'Marsh': 'West', 'Pond': 'East'})

# Correct values in column Animal
df['Animal'] = df['Animal'].replace({'Alex': 'Human', 'Person': 'Human'})
df['Animal'] = df['Animal'].replace({'Racoon': 'Raccoon'})

# Correct values in column Animal
df = df[~df['Animal'].isin(['Squirrel', 'Snake', 'Heron', 'Eagle', 'Bunny', 'Birds'])]

#endregion

# ------------------------------------- CALCULATE THE NUMBER OF DETECTIONS PER WEEK ------------------------------------
#region

# Filter to four wildlife species used in analysis
wildlife_species = ['Human', 'Coyote', 'Raccoon', 'Deer']
df_species = df[df['Animal'].isin(wildlife_species)].copy()
df_species = df_species.loc[df_species.index.repeat(df_species['Count'])].reset_index(drop=True) # duplicate to count

# Convert 'Date' to datetime
df_species['Date'] = pd.to_datetime(df_species['Date'])

# Set start date and create week bins
start_date = pd.to_datetime("2023-05-10") # Wednesday start date
df_species['Week'] = ((df_species['Date'] - start_date).dt.days // 7) + 1  # Week 1 starts at May 10, 2023

# Count number of detections per species per week
detections_per_week = df_species.groupby(['Week', 'Animal']).size().reset_index(name='Detections')

# Print results
print(detections_per_week)

# Pivot the data for plotting
pivot_df = detections_per_week.pivot(index='Week', columns='Animal', values='Detections').fillna(0)

# Map Week number to actual start date
week_start_dates = pd.DataFrame({
    'Week': pivot_df.index,
    'Week Start Date': pd.to_datetime("2023-05-10") + pd.to_timedelta(pivot_df.index - 1, unit='W')
})

# Merge into pivot_df
pivot_df = pivot_df.merge(week_start_dates, left_index=True, right_on='Week')
pivot_df.set_index('Week', inplace=True)  # Optional: keep 'Week' as index

# Visualize as a graph
plt.figure(figsize=(10, 6))
pivot_df.select_dtypes(include='number').plot(kind='line', marker='o')
plt.title('Weekly Wildlife Detections Starting May 10, 2023')
plt.xlabel('Week Number')
plt.ylabel('Number of Detections')
plt.legend(title='Species')
plt.grid(True)
plt.tight_layout()
plt.show()

#endregion

# ---------------------------------------------- NDVI DIFFERENCE PER WEEK ----------------------------------------------
#region

# NDVI difference values
ndvi_differences = [
    ("2023-05-26", 0.0651),
    ("2023-06-06", 0.0596),
    ("2023-06-10", 0.0557),
    ("2023-06-11", 0.0215),
    ("2023-06-12", 0.1208),
    ("2023-06-15", 0.0388),
    ("2023-06-20", 0.0539),
    ("2023-07-06", 0.0471),
    ("2023-07-06", 0.0504),
    ("2023-07-20", 0.0223),
    ("2023-07-21", 0.0014),
    ("2023-07-27", 0.0396),
    ("2023-08-03", 0.0308),
    ("2023-08-18", -0.0037),
    ("2023-08-27", 0.0519),
    ("2023-09-20", 0.0672),
    ("2023-09-26", 0.0191),
    ("2023-09-27", 0.0100),
    ("2023-09-27", 0.0039),
]

# Convert list to DataFrame
ndvi_df = pd.DataFrame(ndvi_differences, columns=['Date', 'NDVI Difference'])
ndvi_df['Date'] = pd.to_datetime(ndvi_df['Date'])

# Create list of weeks (using same logic as above)
weeks = df_species['Week'].unique()
week_start_dates = [pd.to_datetime("2023-05-10") + pd.Timedelta(weeks=int(w)-1, unit='W') for w in weeks]
week_df = pd.DataFrame({'Week': weeks, 'Week_Start_Date': week_start_dates})

# Find closest NDVI date to each week
def closest_ndvi_value(week_date):
    closest_idx = (ndvi_df['Date'] - week_date).abs().idxmin()
    return ndvi_df.loc[closest_idx, 'NDVI Difference']

# Apply function
week_df['NDVI Difference'] = week_df['Week_Start_Date'].apply(closest_ndvi_value)

# Optional: merge with pivot_df if you want to include NDVI in the weekly detections DataFrame
df_weekly_with_ndvi = pivot_df.merge(week_df[['Week', 'NDVI Difference']], left_index=True, right_on='Week')

# Reorder columns: 'Week' first, 'Week_Start_Date' second
cols = ['Week', 'Week Start Date'] + [col for col in df_weekly_with_ndvi.columns if col not in ['Week', 'Week Start Date']]
df_weekly_with_ndvi = df_weekly_with_ndvi[cols]

#endregion

# ----------------------------------------------- MEAN WATER LEVEL PER WEEK --------------------------------------------
#region

# Create a date column
df_wl_raw['Date'] = df_wl_raw['Date (CST)'].str.split().str[0]
df_wl_raw['Date'] = pd.to_datetime(df_wl_raw['Date'], errors='coerce')

# Filter to dates starting from 2023-05-10
df_wl = df_wl_raw[df_wl_raw['Date'] >= pd.to_datetime("2023-05-10")].copy()

# Create Week number column
start_date = pd.to_datetime("2023-05-10")
df_wl['Week'] = ((df_wl['Date'] - start_date).dt.days // 7) + 1

# Compute weekly mean for all numeric columns (adjust if needed)
df_wl_weekly = df_wl.groupby('Week').mean(numeric_only=True).reset_index()

# Rename 'Value(m)' to 'Mean Water Level'
df_wl_weekly = df_wl_weekly.rename(columns={'Value(m)': 'Mean Water Level'})

# Merge with your weekly detections + NDVI table
df_weekly_with_ndvi = df_weekly_with_ndvi.merge(df_wl_weekly, on='Week', how='left')

#endregion

# ----------------------------------------------- MEAN TEMPERATURE PER WEEK --------------------------------------------
#region

# Convert 'Date' to datetime
df_cd['Date'] = pd.to_datetime(df_cd['Date'], errors='coerce')

# Filter from the same start date
df_cd = df_cd[df_cd['Date'] >= pd.to_datetime("2023-05-10")].copy()

# Assign Week number
df_cd['Week'] = ((df_cd['Date'] - pd.to_datetime("2023-05-10")).dt.days // 7) + 1

# Calculate weekly means and max
df_cd_weekly = df_cd.groupby('Week').agg({
    'MEAN_TEMPERATURE': 'mean',
    'MAX_TEMPERATURE': 'max'
}).reset_index()

# Rename columns for clarity
df_cd_weekly = df_cd_weekly.rename(columns={
    'MEAN_TEMPERATURE': 'Mean Temperature',
    'MAX_TEMPERATURE': 'Maximum Temperature'
})

# Merge with final dataframe
df_weekly = df_weekly_with_ndvi.merge(df_cd_weekly, on='Week', how='left')

#endregion

# ------------------------------------------------- GARBAGE PICKUP BINARY ----------------------------------------------
#region

# Define date range
start_period = pd.to_datetime("2023-06-26")
end_period = pd.to_datetime("2023-09-06")

# Create the new column
df_weekly['Garbage Pickup'] = df_weekly['Week'].apply(lambda x: 1 if 7 <= x <= 18 else 0)

print(df_weekly)
print(df_weekly.columns)

#endregion

# ------------------------------------------------- CORRELATION ANALYSIS -----------------------------------------------
#region

# Variable categorization
dependent_cols = ['Coyote', 'Deer', 'Raccoon']
independent_cols = ['Human', 'NDVI Difference', 'Mean Water Level', 'Mean Temperature',
                    'Maximum Temperature', 'Garbage Pickup']

# Correlations
correlation_matrix = df_weekly[independent_cols].corr(method='pearson')

# List unique pairs with correlation > 0.5 (excluding self-correlations)
seen = set()
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1 != col2:
            pair = tuple(sorted((col1, col2)))
            if pair not in seen and correlation_matrix.loc[col1, col2] > 0.5:
                seen.add(pair)
                print(f"{col1} and {col2}: {correlation_matrix.loc[col1, col2]:.3f}")

#endregion

# ------------------------------------------ MULTIVARIATE REGRESSION ANALYSIS ------------------------------------------
#region

# Variable categorization
dependent_cols = ['Coyote', 'Deer', 'Raccoon']

independent_sets = {
    'Model 1': ['Human', 'NDVI Difference', 'Mean Water Level', 'Maximum Temperature', 'Garbage Pickup'],
    'Model 2': ['Human', 'NDVI Difference', 'Mean Water Level', 'Mean Temperature', 'Garbage Pickup'],
}

# Linear regression
for dep in dependent_cols:
    for model_name, indep_cols in independent_sets.items():
        # Subset and clean data
        X = df_weekly[indep_cols]
        y = df_weekly[dep]
        combined = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

        X_clean = sm.add_constant(combined[indep_cols])
        y_clean = combined[dep]

        model = sm.OLS(y_clean, X_clean).fit()
        print(f"\n{model_name} | Dependent: {dep}")
        print(model.summary())

#endregion
