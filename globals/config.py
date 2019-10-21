from globals.fits import Fit
from globals.averaging import Average

# -------------------------- Configure the global variables used in the project

# fit to lane lines on warped image in pixels
fit = Fit()
# fit to lane lines on warped image in meters
fit_real = Fit()
# curvature of lane line
curvature = Average()
# offset of vehicle from middle of lane
offset = Average()
