from netCDF4 import Dataset    # Note: python is case-sensitive!
import numpy as np
#
# reading information from input file
#
file = '/N/slate/ckieu/tmp/output/2020/fnl_20200501_00_00.nc'
f = Dataset(file)
ny = f.dimensions['lat'].size
nx = f.dimensions['lon'].size
y = f.variables['lat']
x = f.variables['lon']
clon = 17.
clat = 5.
rscale = 3.
print("Domain size is",nx,ny)
print(x)
#
# create a new file
#
try: ncfile.close()  # just to be safe, make sure dataset is not already open.
except: pass
ncfile = Dataset('./test.nc',mode='w',format='NETCDF4_CLASSIC') 
print(ncfile)
#
# create dimension
#
lat_dim = ncfile.createDimension('lat', ny)      # latitude axis
lon_dim = ncfile.createDimension('lon', nx)      # longitude axis
#time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
for dim in ncfile.dimensions.items():
    print(dim)
#
# create attribute
#
ncfile.title='Segementation 2D data for Unet model'
ncfile.mask="Segementation values are 0 (outside) or 1 (inside) of TC genesis location"
ncfile.storm_center = "Storm center location in lon/lat degree "+str(clon)+' , '+str(clat)
ncfile.rscale = "Storm mask scale (in degree) for segmentation is "+str(rscale)
print(ncfile)
#
# create variables now with _ values
#
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
#time = ncfile.createVariable('time', np.float64, ('time',))
#time.units = 'hours since 1800-01-01'
#time.long_name = 'time'
mask = ncfile.createVariable('mask',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
mask.units = 'Dimensionless'
mask.standard_name = 'Mask data <0/1> for a single time only' 
print(mask)
#
# assign values for lat,lon,mask now with slice value filling. 
#
lat[:] = y[:]
lon[:] = x[:]
print(lat[:])
dx = x[1]-x[0]
dy = y[1]-y[0]
print("Model dx, dy, clon, clat, scale: ",dx,dy,clon,clat,rscale)
rlat = ny - clat
for j in range(ny):
  for i in range(nx):
    radius = np.sqrt(((ny-j-clat)*dx)**2 + ((i-clon)*dy)**2)
    if radius <= rscale:
      mask[j,i] = 1
    else:
      mask[j,i] = 0

