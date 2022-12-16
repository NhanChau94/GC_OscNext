import numpy as np
from scipy.interpolate import RegularGridInterpolator
import math

# 4D Multilinear interpolation

# Regular grid:
# input: grid (x1,x2,x3,x4), values[x1][x2][x3][x4], interpolation grid
# output: 4D array interpolated values
def RegularGrid_4D(x, values, xinterp):

    # Grid original:
    f_interp = RegularGridInterpolator(x, values)

    # prepare interpolation points
    x1int, x2int, x3int, x4int = np.broadcast_arrays(xinterp[0].reshape(-1,1,1,1), 
                                        xinterp[1].reshape(1,-1,1,1),
                                        xinterp[2].reshape(1,1,-1,1),
                                        xinterp[3])

    coord = np.vstack((x1int.flatten(), # weird but works
                   x2int.flatten(),
                   x3int.flatten(),
                   x4int.flatten()
                   ))

    Val_intp = f_interp(coord.T).reshape(len(xinterp[0]), len(xinterp[1]), len(xinterp[2]), len(xinterp[3]))
    return Val_intp


# 2D Multilinear interpolation

# Regular grid:
# input: grid (x1,x2), values[x1][x2], interpolation grid
# output: 2D array interpolated values
def RegularGrid_2D(x, values, xinterp):

    # Grid original:
    f_interp = RegularGridInterpolator(x, values)

    # prepare interpolation points
    x1int, x2int = np.broadcast_arrays(xinterp[0].reshape(-1,1), 
                                        xinterp[1].reshape(1,-1))

    coord = np.vstack((x1int.flatten(), # weird but works
                   x2int.flatten()
                   ))

    Val_intp = f_interp(coord.T).reshape(len(xinterp[0]), len(xinterp[1]))
    return Val_intp



#------------------------------------------------------------
## Get the open angle from GC, psi from RA and DEC

def psi_f(RA,decl):
    return np.arccos(np.cos(np.pi/2.-(-29.*np.pi/180))*np.cos(np.pi/2.-decl)
                      +np.sin(np.pi/2.-(-29.*np.pi/180))*np.sin(np.pi/2.-decl)*
                       np.cos(RA-266.*np.pi/180))


# Set of functions for time convertor



def date_to_jd(year, month, day, hour, minute, seconde):

    #Convert a date to Julian Day.
    
    #Parameters (int)
    #----------
    #year : YYYY
    #month : MM
    #day : DD
    #hour : HH
    #minute : MN
    #seconde : SC
        
    #Examples
    #--------
    #Convert 6 a.m., February 17, 1985 to Julian Day
    #>>> date_to_jd(1985,2,17,6,0,0)
    #2446113.75
    
    hour += (minute/60.)+(seconde/3600.)
    day_frac = hour/24.
    day += day_frac

    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    
    #Checks where we are in relation to October 15, 1582, the beginning
    #of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        #After start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
        
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
        
    D = math.trunc(30.6001 * (monthp + 1))
    
    jd = B + C + D + day + 1720994.5
    
    return jd

def jd_to_date(jd):

    #Convert Julian Day to date.
    
    #Parameters
    #----------
    #jd : float
    #    Julian Day
        
    #Returns
    #-------
    #year : int
    #    Year as integer. Years preceding 1 A.D. should be 0 or negative.
    #    The year before 1 A.D. is 0, 10 B.C. is year -9.
        
    #month : int
    #    Month as integer, Jan = 1, Feb. = 2, etc.
    
    #day : float
    #    Day, may contain fractional part.
        
    #Examples
    #--------
    #Convert Julian Day 2446113.75 to year, month, and day.
    #>>> jd_to_date(2446113.75)
    #(1985, 2, 17.25)
    
    jd = jd + 0.5
    
    F, I = math.modf(jd)
    I = int(I)
    
    A = math.trunc((I - 1867216.25)/36524.25)
    
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
        
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    
    day = C - E + F - math.trunc(30.6001 * G)
    
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
        
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
        
    return year, month, day

def mjd_to_jd(mjd):
    
    #Convert Modified Julian Day to Julian Day.
        
    #Parameters
    #----------
    #mjd : float
    #    Modified Julian Day
        
    #Returns
    #-------
    #jd : float
    #    Julian Day
    
    return mjd + 2400000.5

    
def jd_to_mjd(jd):

    #Convert Julian Day to Modified Julian Day
    
    #Parameters
    #----------
    #jd : float
    #    Julian Day
        
    #Returns
    #-------
    #mjd : float
    #    Modified Julian Day
    
    return jd - 2400000.5
    
