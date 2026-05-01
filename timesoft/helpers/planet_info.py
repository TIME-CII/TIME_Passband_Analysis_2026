from importlib_resources import files

from astropy.time import Time

import os
import numpy as np

from astropy.coordinates import get_body
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('jpl')
import astropy.units as u

from scipy.interpolate import interp1d


# Mars
# File Format: Y, M, D, H, M?, MJD, T_BB at 30, 80, 115, 150, 200, 230, 260, 300, 330, 360, 425, 650, 800, 950, and 1000 GHz
mars_file = files('timesoft.helpers').joinpath('casa_mars_table.npy')
mars_casa = np.load(mars_file)
mars_casa_freq = np.array([30,80,115,150,200,230,260,300,330,360,425,650,800,950,1000])
mars_casa_dates = np.array([Time({'year':int(mars_casa[0,i]),'month':int(mars_casa[1,i]),'day':int(mars_casa[2,i]),'hour':int(mars_casa[3,i])}, format='ymdhms') for i in range(len(mars_casa[0]))])
def mars_tbb_func(freq,time=None):
    """Get mars bb temperature at a given frequency and time
    
    If no time is given, the average Mars spectrum will be used

    Parameters
    ----------
    freq: float
        Frequency in GHz
    time: astropy.time.Time (optional)
        The date and time of the observation

    Returns
    -------
    tbb: float
        The blackbody temperature of Mars at the specified frequency (and time)
    """

    if time is None:
        return interp1d(mars_casa_freq,np.mean(mars_casa,axis=1)[6:])(freq)
    
    else:
        # Assumes times are ordered in ascending order and gets the last negative timedelta
        deltatime = np.array([(t.unix-time.unix) for t in mars_casa_dates])
        
        # Special case if we have the exact time
        if np.any(deltatime==0):
            i1 = np.nonzero(deltatime==0)[0][0]
            return interp1d(mars_casa_freq,mars_casa[6:,i1])(freq)
        
        # Otherwise linearly interpolate in time
        i1 = np.argmax(deltatime[deltatime<=0])
        tbb1 = interp1d(mars_casa_freq,mars_casa[6:,i1])(freq)
        tbb2 = interp1d(mars_casa_freq,mars_casa[6:,i1+1])(freq)
        tbb = interp1d([deltatime[i1],deltatime[i1+1]],[tbb1,tbb2])(0)

        return tbb

# Other planets
# File Format: frequency, T_BB
jupiter_file = files('timesoft.helpers').joinpath('casa_jupiter_table.npy')
jupiter_casa = np.load(jupiter_file)
_jupiter_tbb_func = interp1d(jupiter_casa[0],jupiter_casa[1])
def jupiter_tbb_func(freq,time):
    return _jupiter_tbb_func(freq)
    
venus_file = files('timesoft.helpers').joinpath('casa_venus_table.npy')
venus_casa = np.load(venus_file)
_venus_tbb_func = interp1d(venus_casa[0],venus_casa[1])
def venus_tbb_func(freq,time):
    return _venus_tbb_func(freq)

uranus_file = files('timesoft.helpers').joinpath('casa_uranus_table.npy')
uranus_casa = np.load(uranus_file)
_uranus_tbb_func = interp1d(uranus_casa[0],uranus_casa[1])
def uranus_tbb_func(freq,time):
    return _uranus_tbb_func(freq)

saturn_file = files('timesoft.helpers').joinpath('gildas_saturn_table.txt')
saturn_gildas = np.genfromtxt(saturn_file)
_saturn_tbb_func = interp1d(saturn_gildas[0],saturn_gildas[1])
def saturn_tbb_func(freq,time):
    return _saturn_tbb_func(freq)


# How to assess each planet:
planets = {
    'mars':{
        'ax_major':3396.19*2, # Table 4 of https://astropedia.astrogeology.usgs.gov/download/Docs/WGCCRE/WGCCRE2015reprint.pdf
        'ax_minor':3376.20*2, # km
        'tbbfunc':mars_tbb_func,
        'tbbcorrection_aro':1.025,
    },
    'jupiter':{
        'ax_major':71492*2,
        'ax_minor':66854*2,
        'jpl_de440_id':5,
        'tbbfunc':jupiter_tbb_func,
        'tbbcorrection_aro':1.0562,
    },
    'venus':{
        'ax_major':6051.8*2,
        'ax_minor':6051.8*2,
        'jpl_de440_id':2,
        'tbbfunc':venus_tbb_func,
        'tbbcorrection_aro':1,
    },
    'uranus':{
        'ax_major':25559*2,
        'ax_minor':24973*2,
        'jpl_de440_id':7,
        'tbbfunc':uranus_tbb_func,
        'tbbcorrection_aro':1,
    },   
    'saturn':{
        'ax_major':58232*2, # From https://ssd.jpl.nasa.gov/planets/phys_par.html
        'ax_minor':58232*2,
        'jpl_de440_id':6,
        'tbbfunc':saturn_tbb_func,
        'tbbcorrection_aro':1,
    },   
}

def get_planet(planet_name,time,freq,aro_correction=False,Tbg=2.7):
    """Retrieve the size and flux of a planet
    
    Parameters
    ----------
    planet_name: str
        Planet name from ['Mars','Jupiter','Venus','Uranus']
    time: astropy.time.Time instance
        Time of the observation in UT
    freq: float
        frequency of the observation in GHz
    aro_correction: bool
        if True rescale the flux of Jupiter in Mars
    Tbg: float
        temperature of the CMB

    Returns
    -------
    flux: float
        Flux in Jy
    size_major: float
        Planet major axis (diameter) in arcseconds
    size_minor: float
        Planet minor axis (diameter) in arcseconds
    """

    if planet_name.lower() not in planets.keys():
        raise ValueError("Only Mars, Jupiter, Saturn, Venus, and Uranus are implemented")
    planet = planets[planet_name.lower()]

    # Look up distance to planet
    dist = get_body(planet_name, time=time).distance.to(u.km).value

    # Compute aparent size - convert major/minor axis to arcseconds at calculated distance
    size_major = planet['ax_major'] / dist * 180/np.pi * 3600
    size_minor = planet['ax_minor'] / dist * 180/np.pi * 3600

    # Compute BB temp of planet
    temp = planet['tbbfunc'](freq,time)
    
    if aro_correction:
        temp = temp * planet['tbbcorrection_aro']

    # Convert BB temperature to a flux in Jy
    const_k = 1.380649e-23
    const_h = 6.62607015e-34
    const_c = 2.998e8
    b1 = 2*const_h*(freq*1e9)**3/const_c**2 * (np.exp(const_h*freq*1e9/const_k/temp)-1)**-1
    b2 = 2*const_h*(freq*1e9)**3/const_c**2 * (np.exp(const_h*freq*1e9/const_k/Tbg)-1)**-1
    db = b1-b2
    area = np.pi*(size_major*size_minor/4) * (np.pi/180/3600)**2
    flux = db * area * 1e26

    return flux, size_major, size_minor

