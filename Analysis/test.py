import rasterio
import numpy as np

path = r"D:\Rospin-solar-site-suitability\Toate fisierele\LandCover.tif"

with rasterio.open(path) as src:
    data = src.read(1)

    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Rezoluție:", src.res)
    print("Shape:", data.shape)
    print("NoData:", src.nodata)
    print("Dtype:", data.dtype)

    values, counts = np.unique(data, return_counts=True)

    print("\nVALORI UNICE:")
    for value, count in zip(values[:100], counts[:100]):
        print(value, "->", count, "pixeli")

    print("\nMin:", np.nanmin(data))
    print("Max:", np.nanmax(data))