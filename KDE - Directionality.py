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

# ------------------------------------------ FIGURE 2 - KDE AND OVERLAP SETUP ------------------------------------------
#region

# Prepare dataframe rows of animals for KDEs
wildlife_species = ['Human', 'Coyote', 'Raccoon', 'Deer']
df_species = df[df['Animal'].isin(wildlife_species)].copy()
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

## Define simple line styles
coming_line_style = {'color': 'black', 'linestyle': '-', 'linewidth': 2}
going_line_style = {'color': 'blue', 'linestyle': '--', 'linewidth': 2}

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

# Loop through each wildlife species
for wildlife in wildlife_species:

    # Separate Coming and Going times for the current wildlife species
    coming_times = df_species[(df_species['Animal'] == wildlife) & (df_species['Direction'] == 'Coming')]['Time_radians'].dropna().values
    going_times = df_species[(df_species['Animal'] == wildlife) & (df_species['Direction'] == 'Going')]['Time_radians'].dropna().values

    # Check if enough data exists
    if len(coming_times) == 0 or len(going_times) == 0:
        print(f"\n{wildlife}: Not enough data for Coming vs Going comparison.")
        continue

    # Bootstrap overlap coefficients
    overlaps = []
    for _ in range(bootstrap_samples):
        coming_sample = resample(coming_times, replace=True)
        going_sample = resample(going_times, replace=True)

        coming_density = vonmises_kde_eval(coming_sample, time_grid, kappa=8)
        going_density = vonmises_kde_eval(going_sample, time_grid, kappa=8)

        overlap = (2 * np.pi / len(time_grid)) * np.sum(np.minimum(coming_density, going_density))
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    overlap_mean = np.mean(overlaps)
    overlap_ci_lower = np.percentile(overlaps, 100 * alpha / 2)
    overlap_ci_upper = np.percentile(overlaps, 100 * (1 - alpha / 2))

    # Compute Δ₁ (grid-based)
    delta1 = delta1_overlap(coming_times, going_times, kappa=8)

    # Compute Δ₄ (sample-based)
    delta4 = delta4_overlap(coming_times, going_times, kappa=8)

    # Wald test
    loc1, loc2, W, p_value = wald_test_vm(coming_times, going_times, n_bootstrap=1000)

    # Print results
    print(f"\n{wildlife} (Coming vs Going):")
    print(f"Δ₁ (grid method) = {delta1:.3f}")
    print(f"Δ₄ (sample method) = {delta4:.3f}")
    print(f"Overlap Coefficient (mean): {overlap_mean:.3f}")
    print(f"95% CI: [{overlap_ci_lower:.3f}, {overlap_ci_upper:.3f}]")
    print(f"Wald test (von Mises) χ² (W) = {W:.3f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant difference between Coming and Going.")
    else:
        print("→ No significant difference between Coming and Going.")

#endregion

# ----------------------------------------------- FIGURE 2 - PLOT THE KDE ----------------------------------------------
#region

# Create a graph for each wildlife species, showing Coming and Going directions
for wildlife in wildlife_species:

    subset = df_species[df_species['Animal'] == wildlife]

    # Separate Coming and Going
    coming = subset[subset['Direction'] == 'Coming']
    going = subset[subset['Direction'] == 'Going']

    # Bootstrap Coming
    all_coming_kdes = []
    for _ in range(bootstrap_samples):
        sample = coming['Time_radians'].sample(frac=1, replace=True)
        _, kde_values = vonmises_kde(sample)
        all_coming_kdes.append(kde_values)
    all_coming_kdes = np.array(all_coming_kdes)
    mean_coming_kde = np.mean(all_coming_kdes, axis=0)

    # Bootstrap Going
    all_going_kdes = []
    for _ in range(bootstrap_samples):
        sample = going['Time_radians'].sample(frac=1, replace=True)
        _, kde_values = vonmises_kde(sample)
        all_going_kdes.append(kde_values)
    all_going_kdes = np.array(all_going_kdes)
    mean_going_kde = np.mean(all_going_kdes, axis=0)

    # Plot Coming vs Going for current species
    plt.figure(figsize=(10, 6))

    # Background shading
    plt.axvspan(321 * 2 * np.pi / 1440, 398.5 * 2 * np.pi / 1440, color='red', alpha=0.3, zorder=1)  # sunrise
    plt.axvspan(1213.5 * 2 * np.pi / 1440, 1294.25 * 2 * np.pi / 1440, color='red', alpha=0.3, zorder=1)  # sunset
    plt.axvspan(213 * 2 * np.pi / 1440, 322.25 * 2 * np.pi / 1440, color='purple', alpha=0.3, zorder=1)  # nautical sunrise
    plt.axvspan(1289.25 * 2 * np.pi / 1440, 1403 * 2 * np.pi / 1440, color='purple', alpha=0.3, zorder=1)  # nautical sunset
    plt.axvspan(793 * 2 * np.pi / 1440, 812.5 * 2 * np.pi / 1440, color='lightblue', alpha=0.3, zorder=1)  # mid-day
    plt.fill_between(time_grid, -0.5, 0, color='white', alpha=1, zorder=3)

    # Plot white under the curves
    plt.fill_between(time_grid, 0, mean_coming_kde, color='white', alpha=1, zorder=2)
    plt.fill_between(time_grid, 0, mean_going_kde, color='white', alpha=1, zorder=2)

    # Plot curves
    plt.plot(time_grid, mean_coming_kde, label='Coming', **coming_line_style)
    plt.plot(time_grid, mean_going_kde, label='Going', **going_line_style)

    # Shade the overlap between Coming and Going
    overlap = np.minimum(mean_coming_kde, mean_going_kde)
    plt.fill_between(
        time_grid,
        0,
        overlap,
        where=(overlap > 0),
        color='grey',
        alpha=0.5,
        zorder=3
    )
    
    # Customize x-axis
    plt.xlim(0, 2 * np.pi)
    plt.xticks(
        ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        labels=['00:00', '06:00', '12:00', '18:00', '24:00'],
        rotation=0,
        fontsize=12
    )

    # Customize y-axis
    plt.ylim(bottom=0.0)
    plt.yticks(fontsize=12)

    # Labels and title
    plt.xlabel('Time of Day', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'{wildlife} (Returning vs Leaving)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

#endregion
