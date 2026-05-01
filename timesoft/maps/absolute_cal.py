import numpy as np 
import matplotlib.pyplot as plt 
from astropy.constants import h, k_B, c
from astropy.coordinates import get_body, EarthLocation
from astropy.time import Time 
import astropy.units as u 
from scipy.interpolate import interp1d
from timesoft.calibration import DetectorConstants
from timesoft import Timestream
from timesoft.detector_cuts import *
from timesoft.calibration import Offsets
from timesoft.helpers.nominal_frequencies import *
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.stats import t
from pathlib import Path


def find_base_dir(start: Path) -> Path:
    for p in [start, *start.parents]: ### iterate through potential paths until you find where the casa files are then return that path
        if (p / "casa_jupiter_table.npy").exists():
            return p
    raise FileNotFoundError("Could not find base_folder")

def lin_fit(x,alpha,beta):
    return x*alpha + beta

# data_names = ['3C454.txt']
# data_names = ['3C279.txt']

'''
data cut from 3C279 txt file 
J1256-0547|J1256-057|J125610-054722|3C279|B1253-055	194.0465274,	2.78E-08,	-5.789312556,	2.78E-08,	4.17E+11,	ALMA-Band 8,	6.65,	0.23,	0,	-17316.9,	ALMA,	2022-04-18T00:00:00Z,	Not Applicable
J1256-0547|J1256-057|J125610-054722|3C279|B1253-055	194.0465274,	2.78E-08,	-5.789312556,	2.78E-08,	8.61E+11,	ALMA-Band 10,	4.9,	0.7,	0,	-17316.9,	ALMA,	2018-11-27T11:50:00Z,	Not Applicable

'''


def estimate_quasar_flux(qname, debug=False):
    data_f = qname + '.txt'

    data = np.loadtxt(data_f, delimiter=',', skiprows=5, usecols=[4,6,7])
    freq = data[:,0] / 1e9; flux = data[:,1]; flux_err = data[:,2]

    sort_inds = np.argsort(freq)

    freq = freq[sort_inds]; flux = flux[sort_inds]; flux_err = flux_err[sort_inds]


    nfreq = np.linspace(int(np.min(freq)),int(np.max(freq)))
    lognfreq = np.log(nfreq)
    logfreq = np.log(freq)
    fig, axs = plt.subplots(1, figsize=(8,8))

    print(freq, flux)
    p,dp = curve_fit(lin_fit, np.log(freq), np.log(flux), sigma=flux_err/flux, absolute_sigma=True,p0=[-1/3,3])

    if debug:


        axs.errorbar(freq, flux, yerr=flux_err, fmt='o', mfc='white', color='black', capsize=2)
        axs.plot(nfreq, np.exp(lin_fit(lognfreq,*p)), color='black')

        n = freq.size
        dof = n - 2  # Degrees of freedom 
        t_value = t.ppf(0.95/ 2, dof)
        y_fit = lin_fit(logfreq, *p)
        # y_err = t_value * np.sqrt(np.diag(dp))
        se = np.sqrt(np.sum((np.log(flux) - y_fit) ** 2) / (n - 2))
        ci = t_value * se * np.sqrt(1 / n + (logfreq - np.mean(logfreq)) ** 2 / np.sum((logfreq - np.mean(logfreq)) ** 2))
        plt.fill_between(freq, np.exp(y_fit - ci), np.exp(y_fit + ci), alpha=0.6, color='blue')


        n = freq.size
        dof = n - 2  # Degrees of freedom 
        t_value = t.ppf(0.68/ 2, dof)
        y_fit = lin_fit(logfreq, *p)
        se = np.sqrt(np.sum((np.log(flux) - y_fit) ** 2) / (n - 2))
        ci = t_value * se * np.sqrt(1 / n + (logfreq - np.mean(logfreq)) ** 2 / np.sum((logfreq - np.mean(logfreq)) ** 2))
        plt.fill_between(freq, np.exp(y_fit - ci), np.exp(y_fit + ci), alpha=0.2)
        # axs.plot(nfreq, np.exp(lin_fit(np.log(nfreq),-1/3,3))
        # axs[1].errorbar(freq, flux, yerr=flux_err, fmt='o', mfc='white', color='black', capsize=2)
        axs.set_yscale('log')
        axs.set_xscale('log')

        axs.axvline(183,color='black',linestyle='dashed')
        axs.axvline(323, color='black', linestyle='dashed')
        axs.set_title(data_f.strip('.txt'))
        axs.set_ylabel('Flux [Jy]')#; axs[1].set_ylabel('Flux [Jy]')
        axs.set_xlabel('Frequency [GHz]')#; axs[1].set_xlabel('Frequency [GHz]')
        fig.savefig(data_f.split('.')[0])

    time_chans =  [f_ind_to_val(f) for f in range(60)]
    log_time_freq = np.log(time_chans)

    return np.exp(lin_fit(log_time_freq, *p))



class absolute_calibration():
    def __init__(self, maps_header):
        """construct the class object for the absolute calibration, this 
        function calls the ``find_angular_size`` method to estimate the 
        angular size of the planet, the ``coupling`` method to estimate the coupling
        of the planet to our telescope, and finally the ``planck_law`` method 
        to estimate the fiducial blackbody spectrum of our calibration source. 
        Note that this only works for planets 

        Parameters
        ----------
        channel_freuquencies : array
            The frequencies in GHz to compute the fiducial spectrum along 
        maps_header : dict
            A dictionary of useful values from the maps object, this 
            should carry in it the planet name, and the date/time of this observation.

        Returns
        -------
        fiducial_model : fiducial model 
        An array of flux values in Jy/B 
        """
        self.channel_frequencies = np.array([f_ind_to_val(i) for i in range(60)])
        self.maps_header = maps_header 

        BASE_DIR = find_base_dir(Path(__file__).resolve())

        self.temp_dict = {'jupiter' : BASE_DIR / 'casa_jupiter_table.npy',  
                          'mars' : BASE_DIR / 'casa_mars_table.npy',
                          'uranus' : BASE_DIR / 'casa_uranus_table.npy',
                          'venus' :BASE_DIR / 'casa_venus_table.npy'}
        
        self.planet_sizes = {'jupiter' : 69911e3,
                             'mars' : 3390e3,
                             'uranus': 25559e3,
                             'venus': 6025e3} #units of m, source JPL / NASA
        
    def planck_law(self,nu,T):
        """Calculate a blackbody spectrum with Planck's law

        Parameters
        ----------
        nu : array 
        frequencies in Hz
        T : array or float 
        the temperature of the object, in the case of planets this should be the T_b model from CASA

        Returns
        -------
        blackbody : blackbody
        A blackbody in units of W/m^2/Hz/Sr
        """

        # print oc.value, h.value, k_B.value 
        print(h.value, c.value, k_B.value)
        blackbody = 2 * h.value * nu**3 / c.value**2 / np.expm1(h.value * nu / (k_B.value * T))
        return blackbody

    def coupling(self,nu, ang_diameter):
        """ 
        Calculate the coupling between the intrinsic flux of the source and the telescope's beam 

        Parameters
        ----------
        nu : array
        frequencies in Hz 

        Returns
        -------
        coupling : coupling 
        The coupling of the intrinsic flux to the telescope beamsize 
        """

        fig, axs = plt.subplots(1,2, figsize=(8,4))

        beam_fwhm = 1.2 * 2.998e8 / nu /1e9 / 12 * 180/np.pi*3600

        # beam_fwhm = 1.22*(3e8 / (nu))/12*1.1 * 206265 #### convert to arcseconds with 206265
        # planet_diam = np.sqrt(4 / (np.pi) *self.planet_sr)
        planet_diam = np.sqrt(self.planet_sr) * 2
        x = ang_diameter / beam_fwhm
        coupling = (1-np.exp(-np.log(2)*x**2))/(np.log(2)*x**2)

        axs[0].plot(nu, beam_fwhm)
        axs[0].set_xlabel('Frequencies')
        axs[0].set_ylabel('Beam FWHM')
        axs[1].plot(nu, coupling)
        axs[1].set_xlabel('Frequencies')
        axs[1].set_ylabel('Coupling')
        fig.tight_layout()
        fig.savefig('testing_planet_gains_estimation_stuff', bbox_inches='tight')
        # axs[2].plot()

        return coupling

    def find_angular_size(self):
        """ 
        Computes the angular size of a planet given the time of observation (assumes kitt peak location)

        Parameters
        ----------
        None

        Returns
        -------
        planet_sr : planet_sr
        the solid angle of the planet, it is also saved as a class variable. 
        
        """
        # date = ['2021-02-09T05:00:00']
        date = self.planet_header['datetime']
        t = Time(date, format='unix')
        loc = EarthLocation.of_site('Kitt Peak')
        planet_eph = get_body(self.planet_header['object'], t, loc)
        earth_eph = get_body('earth',t, loc)

        planet_rad = self.planet_sizes[self.planet_header['object']]
        dist = earth_eph.separation_3d(planet_eph).to(u.m).value
        angular_diameter = (2*planet_rad) / dist * 206265 #### 206265 is conversion from radian to arcsec

        self.planet_sr = np.pi * (angular_diameter/3600 * np.pi/180)**2 /4 # size in str
        # self.planet_sr = np.pi * angular_radius**2
        print(self.planet_sr)
        return self.planet_sr, angular_diameter

    def beam_to_sr(self):
        conv = np.zeros((len(self.channel_frequencies)))
        for i in range(len(self.channel_frequencies)):
            #### replace fwhm theoretical with actual fitted values
            fwhm = 1.2 * 2.998e8 / self.channel_frequencies[i] / 1e9 / 12 * 180 / np.pi * 3600
            conv[i] = fwhm**2 * (1/3600)**2 * (np.pi / 180)**2 * (np.pi / (4 * np.log(2)))
        return conv

    def make_fiducial_planet_spectrum(self, test_plot=False, save_path=None):

        solid_angle, ang_diameter = self.find_angular_size() #first find the solid angle subtended 
        datafile = self.temp_dict[self.planet_header['object']] #next open the relevant temperature data file 
        data = np.load(datafile, allow_pickle=True) 
        if self.planet_header['object'].upper() != 'mars'.upper():
             #compute the coupling to the telescope 
            bb = self.planck_law(data[0]*1e9, data[1]) * 1e26 #converts to Jy
            interpolator = interp1d(data[0],bb)
            # couple = 1 #### for Jupiter where the planet size is bigger than the beam we do an area normalized fit.
        else:
            dtime = datetime.datetime.fromtimestamp(self.maps_header['timestamp'])
            YYYY = float(dtime.strftime("%Y"))
            MM   = float(dtime.strftime("%m"))
            DD   = float(dtime.strftime("%d"))


            this_year_data = np.where(data[0,:] == YYYY)[0]
            this_month_data = np.where(data[1,:] == MM)[0]
            this_data = np.intersect1d(this_year_data, this_month_data)
            this_day_data = np.where(data[2,:] == DD)[0]
            this_data = np.intersect1d(this_day_data, this_data)
            temp = np.nanmedian(data[6::,this_data], axis=1)
            std = np.nanstd(data[6::,this_data], axis=1)
            mars_freqs =  np.array([30, 80, 115, 150, 200, 230, 260, 300, 330, 360, 425, 650, 800, 950, 1000])

            fig, axs = plt.subplots(1, figsize=(6,6))
            axs.plot(mars_freqs, temp)
            axs.set_xlabel('Frequency')
            axs.set_ylabel('Temperature [k]')
            fig.savefig(save_path + 'fid_temp', bbox_inches='tight')
            bb = self.planck_law(mars_freqs*1e9, temp)*1e26

            interpolator = interp1d(mars_freqs,bb)

        couple = self.coupling(self.channel_frequencies, ang_diameter)

        interped_bb = interpolator(self.channel_frequencies) * self.planet_sr * couple

        if test_plot:
            fig, axs = plt.subplots(1,3,figsize=(8,4))
            fig.suptitle('Jupiter')
            axs[0].plot(data[0], data[1])
            # axs[0].plot(mars_freqs, temp)
            axs[0].set_xlim(150,350)
            axs[0].set_xlabel('Frequency [GHz]')
            axs[0].set_ylabel('Temperature [K]')
            axs[1].set_xlim(150,350)
            # axs[1].set_ylim(0,1000)

            axs[0].set_ylim(164,167)
            # axs[0].set_ylim(207,216)

            axs[1].plot(self.channel_frequencies, couple)
            axs[1].set_xlabel('Frequency [GHz]')
            axs[1].set_ylabel('Coupling Term')
            axs[1].set_xlim(183,323)

            axs[2].set_xlim(183,323)
            axs[2].plot(self.channel_frequencies, interped_bb, label='W/O Coupling')
            axs[2].plot(self.channel_frequencies, interped_bb * couple, label='W/ Coupling')
            axs[2].set_xlabel('Frequency [GHz]')
            axs[2].legend(framealpha=0)
            axs[2].set_ylabel('Flux [Jy]')
            fig.tight_layout()
            fig.savefig(save_path + 'fiducial_spectrum', bbox_inches='tight')
            plt.close(fig)
        return interped_bb
