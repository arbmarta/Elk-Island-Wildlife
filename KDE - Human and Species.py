# -------------------------------------------- IMPORT PACKAGES AND DATASETS --------------------------------------------
#region

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises, chi2
from sklearn.utils import resample

# Import dataset
df = pd.read_csv('Dataset.csv')

#endregion

# ----------------------------------------- ADJUST COLUMNS IN DETECTION DATASET ----------------------------------------
#region

# Create cardinal direction column
df['Cardinal Direction'] = df['Location'].replace({'Marsh': 'West', 'Pond': 'East'})

# Correct values in column Animal
df['Animal'] = df['Animal'].replace({'Alex': 'Human', 'Person': 'Human', 'Racoon': 'Raccoon'})

# Correct values in column Animal
df = df[~df['Animal'].isin(['Squirrel', 'Snake', 'Heron', 'Eagle', 'Bunny', 'Birds'])]

#endregion

# ---------------------------------------------------- TABLE 1 DATA ----------------------------------------------------
#region

# Make sure Date is datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Filter for Human observations only
human_df = df[df['Animal'] == 'Human']

# Filter starting from May 10, 2023
human_df = human_df[human_df['Date'] >= '2023-05-10']

# Set Date as index
human_df = human_df.set_index('Date')

# Sum counts over each 7-day window per Location
weekly_sums = human_df.groupby('Location').resample('7D')['Count'].sum().reset_index()

# Now summarize: mean and range (max - min) per Location
summary = weekly_sums.groupby('Location')['Count'].agg(
    mean='mean',
    range=lambda x: x.max() - x.min()
)

# Print the summary table
print(summary)

# Sum across all weeks to get total number of Human detections per Location
total_human_detections = weekly_sums.groupby('Location')['Count'].sum()
print(total_human_detections)

#endregion

# ---------------------------------------------------- TABLE 2 DATA ----------------------------------------------------
#region

# Summarize the number of group observations
animal_counts = df['Animal'].value_counts().sort_index()
print(animal_counts)

# Summarize the number of total detections
animal_sums = df.groupby('Animal')['Count'].sum().sort_index()
print(animal_sums)

# Summarize the number of group observations by each location
animal_counts_by_direction = df.groupby(['Cardinal Direction', 'Animal']).size().sort_index()
print(animal_counts_by_direction)

# Summarize the number of total detections by each location
animal_sums_by_direction = df.groupby(['Cardinal Direction', 'Animal'])['Count'].sum().sort_index()
print(animal_sums_by_direction)

#endregion

# ------------------------------------------ FIGURE 2 - KDE AND OVERLAP SETUP ------------------------------------------
#region

# Prepare dataframe rows of animals for KDEs
wildlife_species = ['Coyote', 'Raccoon', 'Deer'] # list of species to compare against Human
df_species = df[df['Animal'].isin(['Human'] + wildlife_species)].copy() # prepare Human + all wildlife rows
df_species['Time'] = pd.to_datetime(df_species['Time'], format='%H:%M:%S', errors='coerce') # ensure Time is datetime
df_species['Time_minutes'] = df_species['Time'].dt.hour * 60 + df_species['Time'].dt.minute # minutes since midnight
df_species['Time_radians'] = df_species['Time_minutes'] * (2 * np.pi / 1440)
df_species = df_species.loc[df_species.index.repeat(df_species['Count'])].reset_index(drop=True)

# Setup bootstrapping
bootstrap_samples = 10  # (Change to 10000 for final version)
time_grid = np.linspace(0, 2 * np.pi, 240)  # 0–2π radians for 24h

# Set up the von Mises KDE
def vonmises_kde(samples, kappa=15, grid_points=240):
    grid = np.linspace(0, 2*np.pi, grid_points)
    density = np.zeros_like(grid)
    for sample in samples:
        density += vonmises.pdf(grid, kappa, loc=sample)
    density /= len(samples)
    return grid, density

# run human bootstrapping
human_df = df_species[df_species['Animal'] == 'Human']
all_human_kdes = []

for _ in range(bootstrap_samples):
    sample = human_df['Time_radians'].sample(frac=1, replace=True)
    time_grid, kde_values = vonmises_kde(sample)
    all_human_kdes.append(kde_values)

all_human_kdes = np.array(all_human_kdes)
mean_human_kde = np.mean(all_human_kdes, axis=0)

## Define simple line styles
human_line_style = {'color': 'black', 'linestyle': '-', 'linewidth': 2}
wildlife_line_style = {'color': 'blue', 'linestyle': '--', 'linewidth': 2}

# von Mises KDE function 2
def vonmises_kde_eval(samples, eval_points, kappa=8):
    """
    Evaluates von Mises KDE at given points.
    """
    densities = np.zeros_like(eval_points)
    for sample in samples:
        densities += vonmises.pdf(eval_points, kappa, loc=sample)
    densities /= len(samples)
    return densities

# Delta 4 overlap (for n > 50)
def delta4_overlap(human_samples, wildlife_samples, kappa=8):

    # Evaluate KDEs at sample points
    f_x = vonmises_kde_eval(human_samples, human_samples, kappa)     # f̂(xᵢ)
    g_x = vonmises_kde_eval(wildlife_samples, human_samples, kappa)  # ĝ(xᵢ)

    f_y = vonmises_kde_eval(human_samples, wildlife_samples, kappa)  # f̂(yⱼ)
    g_y = vonmises_kde_eval(wildlife_samples, wildlife_samples, kappa)  # ĝ(yⱼ)

    # Compute min ratios
    part1 = np.mean(np.minimum(1, g_x / f_x))
    part2 = np.mean(np.minimum(1, f_y / g_y))

    delta4 = 0.5 * (part1 + part2)
    return delta4

# Delta 1 overlap (for n < 50)
def delta1_overlap(human_samples, wildlife_samples, kappa=8, grid_points=240):
    grid = np.linspace(0, 2 * np.pi, grid_points)

    f = vonmises_kde_eval(human_samples, grid, kappa)
    g = vonmises_kde_eval(wildlife_samples, grid, kappa)

    delta1 = (2 * np.pi / grid_points) * np.sum(np.minimum(f, g))
    return delta1

# Wald test function
def wald_test_vm(human_samples, wildlife_samples, n_bootstrap=1000):
    human_means = []
    wildlife_means = []

    for _ in range(n_bootstrap):
        hs = resample(human_samples, replace=True)
        ws = resample(wildlife_samples, replace=True)

        _, loc1, _ = vonmises.fit(hs, fscale=1)
        _, loc2, _ = vonmises.fit(ws, fscale=1)

        human_means.append(loc1)
        wildlife_means.append(loc2)

    loc1_mean = np.mean(human_means)
    loc2_mean = np.mean(wildlife_means)
    se1 = np.std(human_means)
    se2 = np.std(wildlife_means)

    W = (loc1_mean - loc2_mean) ** 2 / (se1 ** 2 + se2 ** 2)
    p_val = 1 - chi2.cdf(W, df=1)

    return loc1_mean, loc2_mean, W, p_val

#endregion

# ------------------------------------------- FIGURE 2 - COEFFICIENT OF OVERLAP ----------------------------------------
#region

# Setup
alpha = 0.05

# Pre-filter Human data
human_times = df_species[df_species['Animal'] == 'Human']['Time_radians'].dropna().values

# Loop through each wildlife species
for wildlife in wildlife_species:
    wildlife_times = df_species[df_species['Animal'] == wildlife]['Time_radians'].dropna().values

    # Bootstrap overlap coefficients
    overlaps = []

    for _ in range(bootstrap_samples):
        human_sample = resample(human_times, replace=True)
        wildlife_sample = resample(wildlife_times, replace=True)

        human_density = vonmises_kde_eval(human_sample, time_grid, kappa=8)
        wildlife_density = vonmises_kde_eval(wildlife_sample, time_grid, kappa=8)

        overlap = (2 * np.pi / len(time_grid)) * np.sum(np.minimum(human_density, wildlife_density))
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    overlap_mean = np.mean(overlaps)
    overlap_ci_lower = np.percentile(overlaps, 100 * alpha / 2)
    overlap_ci_upper = np.percentile(overlaps, 100 * (1 - alpha / 2))

    # Compute Δ₁ (grid-based)
    delta1 = delta1_overlap(human_times, wildlife_times, kappa=8)

    # Compute Δ₄ (sample-based)
    delta4 = delta4_overlap(human_times, wildlife_times, kappa=8)

    # Wald test
    loc1, loc2, W, p_value = wald_test_vm(human_times, wildlife_times, n_bootstrap=1000)

    # Print results
    print(f"\n{wildlife}:")
    print(f"Δ₁ (grid method) = {delta1:.3f}")
    print(f"Δ₄ (sample method) = {delta4:.3f}")
    print(f"Overlap Coefficient (mean): {overlap_mean:.3f}")
    print(f"95% CI: [{overlap_ci_lower:.3f}, {overlap_ci_upper:.3f}]")
    print(f"Wald test (von Mises) χ² (W) = {W:.3f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant difference in activity patterns.")
    else:
        print("→ No significant difference in activity patterns.")

#endregion

# ----------------------------------------------- FIGURE 2 - PLOT THE KDE ----------------------------------------------
#region

## Create a graph for each wildlife species compared to Human
for wildlife in wildlife_species:

    subset = df_species[df_species['Animal'] == wildlife]
    all_wildlife_kdes = []

    for _ in range(bootstrap_samples):
        sample = subset['Time_radians'].sample(frac=1, replace=True)
        time_grid, kde_values = vonmises_kde(sample)
        all_wildlife_kdes.append(kde_values)

    all_wildlife_kdes = np.array(all_wildlife_kdes)
    mean_wildlife_kde = np.mean(all_wildlife_kdes, axis=0)

    # Plot Human vs current wildlife species
    plt.figure(figsize=(10, 6))

    # Plot backgrounds - solid
    plt.axvspan(321 * 2 * np.pi / 1440, 398.5 * 2 * np.pi / 1440, color='red', alpha=0.3, zorder=1)  # sunrise
    plt.axvspan(1213.5 * 2 * np.pi / 1440, 1294.25 * 2 * np.pi / 1440, color='red', alpha=0.3, zorder=1)  # sunset
    plt.axvspan(213 * 2 * np.pi / 1440, 322.25 * 2 * np.pi / 1440, color='purple', alpha=0.3,
                zorder=1)  # nautical sunrise
    plt.axvspan(1289.25 * 2 * np.pi / 1440, 1403 * 2 * np.pi / 1440, color='purple', alpha=0.3,
                zorder=1)  # nautical sunset
    plt.axvspan(793 * 2 * np.pi / 1440, 812.5 * 2 * np.pi / 1440, color='lightblue', alpha=0.3, zorder=1)  # mid-day
    plt.fill_between(time_grid, -0.5, 0, color='white', alpha=1, zorder=3)

    # Plot white under the curves
    plt.fill_between(time_grid,0, mean_human_kde, color='white', alpha=1, zorder=2) # plot white under the human curve
    plt.fill_between(time_grid,0, mean_wildlife_kde, color='white', alpha=1, zorder=2) # plot white under the wildlife curve

    # Plot the curves
    plt.plot(time_grid, mean_human_kde, label='Human', **human_line_style)
    plt.plot(time_grid, mean_wildlife_kde, label=wildlife, **wildlife_line_style)

    # Shade the overlap
    overlap = np.minimum(mean_human_kde, mean_wildlife_kde)
    plt.fill_between(
        time_grid,
        0,
        overlap,
        where=(overlap > 0),
        color='grey',
        alpha=0.5,
        zorder = 3
    )

    # Customize x-axis
    plt.xlim(0, 2 * np.pi)
    plt.xticks(
        ticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        labels=['00:00', '06:00', '12:00', '18:00', '24:00'],
        rotation=0,
        fontsize=12
    )

    # Customize y-axis
    plt.ylim(-0.00005, .65)
    plt.yticks(fontsize=12)

    # Set axis labels
    plt.xlabel('Time of Day', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')

    # Title
    plt.title(f'Human & {wildlife}', fontsize=12, fontweight='bold')

    # Remove legend
    plt.legend().remove()

    plt.tight_layout()
    plt.show()

#endregion
