# Albers R6 WKT
CRS = (
    'PROJCS["NAD83 / Conus Albers",'
    'GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101,'
    'AUTHORITY["EPSG","7019"]],'
    "TOWGS84[0,0,0,0,0,0,0],"
    'AUTHORITY["EPSG","6269"]],'
    'PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4269"]],'
    'PROJECTION["Albers_Conic_Equal_Area"],'
    'PARAMETER["standard_parallel_1",43.0],'
    'PARAMETER["standard_parallel_2",48.0],'
    'PARAMETER["latitude_of_center",34.0],'
    'PARAMETER["longitude_of_center",-120.0],'
    'PARAMETER["false_easting",600000.0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,'
    'AUTHORITY["EPSG","9001"]],'
    'AXIS["X",EAST],'
    'AXIS["Y",NORTH]]'
)

# The NAIP bands stored during sampling. Subsets of these bands may be used during
# training, but all bands will be present in the datasets.
BANDS = ("R", "G", "B", "N")
