import rasterio
import numpy as np
from xml.dom import minidom
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

## ---------------------------------------- SHAPEFILES OF LOCATIONS FOR ANALYSIS ---------------------------------------
#region

# Import shapefiles
parks_gpd = gpd.read_file("Datasets/Shapefiles/Manitoba_Parks.shp")
vb_gpd = gpd.read_file("Datasets/Shapefiles/MUNICIPALITY.shp")

# Identify areas
parks_gpd = parks_gpd[parks_gpd['NAME_E'] == 'Elk Island Provincial Park']
vb_gpd = vb_gpd[vb_gpd['MUNI_NAME'] == 'RM OF VICTORIA BEACH']

# Explode the shapefiles into individual geometries
parks_gpd = parks_gpd.explode(index_parts=True)
vb_gpd = vb_gpd.explode(index_parts=True)

# Trim to Elk Island and Victoria Beach mainland
parks_gpd = parks_gpd.loc[parks_gpd.area != parks_gpd.area.min()]
smallest_areas = vb_gpd.area.nsmallest(2)
vb_gpd = vb_gpd.loc[~vb_gpd.area.isin(smallest_areas)]
vb_gpd = vb_gpd.loc[vb_gpd.area != vb_gpd.area.max()]

# Show shapefiles on a map
parks_gpd.plot()
plt.show()

vb_gpd.plot()
plt.show()

#endregion

## ---------------------------------------------- IMPORT SATELLITE IMAGES ----------------------------------------------
#region

# Satellite images
metadata_files = [
    "Datasets/PSScene/20230506_162314_08_2459_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230515_163345_31_24a8_3B_AnalyticMS_metadata_clip.xml", #2
    "Datasets/PSScene/20230526_170711_76_248c_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230606_163226_61_2430_3B_AnalyticMS_metadata_clip.xml", #4
    "Datasets/PSScene/20230610_163206_53_241e_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230611_165718_65_225a_3B_AnalyticMS_metadata_clip.xml", #6
    "Datasets/PSScene/20230612_161507_28_2445_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230615_170517_56_2473_3B_AnalyticMS_metadata_clip.xml", #8
    "Datasets/PSScene/20230620_171154_54_247b_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230706_163214_07_24a8_3B_AnalyticMS_metadata_clip.xml", #10
    "Datasets/PSScene/20230706_170947_74_227a_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230720_163341_38_24c8_3B_AnalyticMS_metadata_clip.xml", #12
    "Datasets/PSScene/20230721_163301_79_24b9_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230727_163200_50_2429_3B_AnalyticMS_metadata_clip.xml", #14
    "Datasets/PSScene/20230803_171343_83_2490_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230818_171408_92_2473_3B_AnalyticMS_metadata_clip.xml", #16
    "Datasets/PSScene/20230827_171130_02_248c_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230920_171227_80_248b_3B_AnalyticMS_metadata_clip.xml", #18
    "Datasets/PSScene/20230926_163041_39_2460_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20230927_163308_06_2440_3B_AnalyticMS_metadata_clip.xml", #20
    "Datasets/PSScene/20230927_171214_21_2486_3B_AnalyticMS_metadata_clip.xml",
    "Datasets/PSScene/20231010_171755_65_2482_3B_AnalyticMS_metadata_clip.xml" #22
]

sr_files = [
    "Datasets/PSScene/20230506_162314_08_2459_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230515_163345_31_24a8_3B_AnalyticMS_SR_clip.tif", #2
    "Datasets/PSScene/20230526_170711_76_248c_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230606_163226_61_2430_3B_AnalyticMS_SR_clip.tif", #4
    "Datasets/PSScene/20230610_163206_53_241e_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230611_165718_65_225a_3B_AnalyticMS_SR_clip.tif", #6
    "Datasets/PSScene/20230612_161507_28_2445_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230615_170517_56_2473_3B_AnalyticMS_SR_clip.tif", #8
    "Datasets/PSScene/20230620_171154_54_247b_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230706_163214_07_24a8_3B_AnalyticMS_SR_clip.tif", #10
    "Datasets/PSScene/20230706_170947_74_227a_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230720_163341_38_24c8_3B_AnalyticMS_SR_clip.tif", #12
    "Datasets/PSScene/20230721_163301_79_24b9_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230727_163200_50_2429_3B_AnalyticMS_SR_clip.tif", #14
    "Datasets/PSScene/20230803_171343_83_2490_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230818_171408_92_2473_3B_AnalyticMS_SR_clip.tif", #16
    "Datasets/PSScene/20230827_171130_02_248c_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230920_171227_80_248b_3B_AnalyticMS_SR_clip.tif", #18
    "Datasets/PSScene/20230926_163041_39_2460_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20230927_163308_06_2440_3B_AnalyticMS_SR_clip.tif", #20
    "Datasets/PSScene/20230927_171214_21_2486_3B_AnalyticMS_SR_clip.tif",
    "Datasets/PSScene/20231010_M_171755_65_2482_3B_AnalyticMS_SR_clip.tif" #22
]

#endregion

## ------------------------ DETERMINE CRS OF SHAPEFILES AND SATELLITE IMAGES AND CONVERT TO SAME -----------------------
#region

# CRS of shapefiles
print(vb_gpd.crs)
print(parks_gpd.crs)

# CRS of .tif file
for sr_file in sr_files:
    with rasterio.open(sr_file) as src:
        print(f"{sr_file}: {src.crs}")

# Convert to same CRS
vb_gpd = vb_gpd.to_crs("EPSG:32614")
parks_gpd = parks_gpd.to_crs("EPSG:32614")

# Reprint CRS of shapefiles
print(vb_gpd.crs)
print(parks_gpd.crs)

# Reprint CRS of .tif file
for sr_file in sr_files:
    with rasterio.open(sr_file) as src:
        print(f"{sr_file}: {src.crs}")

#endregion

## ------------------------------------------------- CALCULATE NDVI - VB -----------------------------------------------
#region

# Collect NDVI stats
ndvi_means_vb = []
ndvi_dates_vb = []

# NDVI calculator loop
for sr_file, metadata_file in zip(sr_files, metadata_files):
    date_str = os.path.basename(sr_file)[:8]
    ndvi_date = datetime.strptime(date_str, "%Y%m%d")

    # Load bands
    with rasterio.open(sr_file) as src:
        band_blue = src.read(1)
        band_green = src.read(2)
        band_red = src.read(3)
        band_nir = src.read(4)
        kwargs = src.meta

    # Reflectance coefficients from metadata
    try:
        xmldoc = minidom.parse(metadata_file)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        coeffs = {}
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                i = int(bn)
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i] = float(value)
    except Exception as e:
        print(f"Skipping {metadata_file} due to XML error: {e}")
        continue

    # Apply reflectance coefficients
    band_red = band_red.astype(np.float32) * coeffs[3]
    band_nir = band_nir.astype(np.float32) * coeffs[4]
    band_green = band_green.astype(np.float32) * coeffs[2]
    band_blue = band_blue.astype(np.float32) * coeffs[1]

    # NDVI
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (band_nir - band_red) / (band_nir + band_red)

    # Save NDVI raster
    kwargs.update(dtype=rasterio.float32, count=1)
    output_ndvi = sr_file.replace("SR_clip.tif", "ndvi.tif")
    with rasterio.open(output_ndvi, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))

    # Clip to vb_gpd
    with rasterio.open(output_ndvi) as src_ndvi:
        vb_gpd = vb_gpd.to_crs(src_ndvi.crs)
        ndvi_masked, _ = mask(src_ndvi, vb_gpd.geometry, crop=True)

    # Calculate stats
    ndvi_values = ndvi_masked[0].flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    ndvi_values = ndvi_values[ndvi_values != 0]  # exclude exact 0 values

    if len(ndvi_values) > 0:
        mean_ndvi = ndvi_values.mean()
        ndvi_means_vb.append(mean_ndvi)
        ndvi_dates_vb.append(ndvi_date)
        print(f"Mean NDVI for {ndvi_date.strftime('%Y-%m-%d')}: {mean_ndvi:.4f}")
    else:
        print(f"No NDVI data for {sr_file}")
        continue

    # Normalize RGB for display
    rgb = np.dstack((band_red, band_green, band_blue))
    rgb_min = np.percentile(rgb, 2)
    rgb_max = np.percentile(rgb, 98)
    rgb_norm = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)

    # Display side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(rgb_norm)
    axs[0].set_title(f"RGB Composite\n{ndvi_date.strftime('%Y-%m-%d')}")
    axs[0].axis('off')

    ndvi_img = axs[1].imshow(ndvi_masked[0], cmap='summer', vmin=0, vmax=1)
    axs[1].set_title("Clipped NDVI")
    axs[1].axis('off')
    fig.colorbar(ndvi_img, ax=axs[1], fraction=0.046, pad=0.04, label='NDVI')
    plt.tight_layout()
    plt.show()

#endregion

## ------------------------------------------------- CALCULATE NDVI - VB -----------------------------------------------
#region

# Collect NDVI stats
ndvi_means_elk = []
ndvi_dates_elk = []

# NDVI calculator loop
for sr_file, metadata_file in zip(sr_files, metadata_files):
    date_str = os.path.basename(sr_file)[:8]
    ndvi_date = datetime.strptime(date_str, "%Y%m%d")

    # Load bands
    with rasterio.open(sr_file) as src:
        band_blue = src.read(1)
        band_green = src.read(2)
        band_red = src.read(3)
        band_nir = src.read(4)
        kwargs = src.meta

    # Reflectance coefficients from metadata
    try:
        xmldoc = minidom.parse(metadata_file)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        coeffs = {}
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                i = int(bn)
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i] = float(value)
    except Exception as e:
        print(f"Skipping {metadata_file} due to XML error: {e}")
        continue

    # Apply reflectance coefficients
    band_red = band_red.astype(np.float32) * coeffs[3]
    band_nir = band_nir.astype(np.float32) * coeffs[4]
    band_green = band_green.astype(np.float32) * coeffs[2]
    band_blue = band_blue.astype(np.float32) * coeffs[1]

    # NDVI
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (band_nir - band_red) / (band_nir + band_red)

    # Save NDVI raster
    kwargs.update(dtype=rasterio.float32, count=1)
    output_ndvi = sr_file.replace("SR_clip.tif", "ndvi.tif")
    with rasterio.open(output_ndvi, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))

    # Clip to parks_gpd
    with rasterio.open(output_ndvi) as src_ndvi:
        parks_gpd = parks_gpd.to_crs(src_ndvi.crs)
        ndvi_masked, _ = mask(src_ndvi, parks_gpd.geometry, crop=True)

    # Calculate stats
    ndvi_values = ndvi_masked[0].flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    ndvi_values = ndvi_values[ndvi_values != 0]  # exclude exact 0 values

    if len(ndvi_values) > 0:
        mean_ndvi = ndvi_values.mean()
        ndvi_means_elk.append(mean_ndvi)
        ndvi_dates_elk.append(ndvi_date)
        print(f"Mean NDVI for {ndvi_date.strftime('%Y-%m-%d')}: {mean_ndvi:.4f}")
    else:
        print(f"No NDVI data for {sr_file}")
        continue

    # Normalize RGB for display
    rgb = np.dstack((band_red, band_green, band_blue))
    rgb_min = np.percentile(rgb, 2)
    rgb_max = np.percentile(rgb, 98)
    rgb_norm = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)

    # Display side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(rgb_norm)
    axs[0].set_title(f"RGB Composite\n{ndvi_date.strftime('%Y-%m-%d')}")
    axs[0].axis('off')

    ndvi_img = axs[1].imshow(ndvi_masked[0], cmap='summer', vmin=0, vmax=1)
    axs[1].set_title("Clipped NDVI")
    axs[1].axis('off')
    fig.colorbar(ndvi_img, ax=axs[1], fraction=0.046, pad=0.04, label='NDVI')
    plt.tight_layout()
    plt.show()

#endregion

## ------------------------------------------------------ PLOT NDVI ----------------------------------------------------
#region

# Plot NDVI over time for Victoria Beach and Elk Island
plt.figure(figsize=(10, 5))

# Plot Victoria Beach
plt.plot(ndvi_dates_vb, ndvi_means_vb, marker='o', linestyle='-', label='Victoria Beach', color='green')

# Plot Elk Island
plt.plot(ndvi_dates_elk, ndvi_means_elk, marker='s', linestyle='--', label='Elk Island', color='blue')

plt.title("NDVI Over Time")
plt.xlabel("Date")
plt.ylabel("Mean NDVI")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#endregion

## -------------------------------------------------- DIFFERENCE IN NDVI -----------------------------------------------
#region

print("\nNDVI Differences (Elk Island - Victoria Beach):")
for date_elk, ndvi_elk, date_vb, ndvi_vb in zip(ndvi_dates_elk, ndvi_means_elk, ndvi_dates_vb, ndvi_means_vb):
    if date_elk != date_vb:
        print(f"⚠️ Mismatched dates: {date_elk} vs {date_vb}")
        continue
    diff = ndvi_elk - ndvi_vb
    print(f"{date_elk.strftime('%Y-%m-%d')}: {diff:.4f}")

#endregion

## -------------------------------------------------- ELK ISLAND NDVI --------------------------------------------------
#region

print("\nElk Island NDVI Values:")
for date, ndvi in zip(ndvi_dates_elk, ndvi_means_elk):
    print(f"{date.strftime('%Y-%m-%d')}: {ndvi:.4f}")

#endregion
