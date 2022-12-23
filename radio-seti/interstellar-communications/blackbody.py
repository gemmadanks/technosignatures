"""Quantifying radiation from a blackbody."""

import numpy as np
import math

from utils import surface_area_sphere
from constants import STEFAN_BOLTZMANN, PLANCK, BOLTZMANN, LIGHT_SPEED

def stefan_boltzmann_law(temperature):
    """Stefan-Boltzmann's law.
    
    Args:
        temperature: temperature in Kelvin
        
    Returns:
        Total energy radiated per unit surface area of a black body across all wavelengths per unit time.
    """
    return STEFAN_BOLTZMANN * pow(temperature, 4)
    
def spectral_radiance_planck(frequency, temperature):
    """Planck's law of black-body radiation.
    
    Computes spectral radiance (energy emitted per second per m2 per steradian per Hz)
    using Planck's law.
    
    Args:
        frequency: Frequency of radiation in Hz
        temperature: Temperature of blackbody in Kelvin
    Returns:
        Spectral radiance at frequency given (Wm-2sr-1Hz-1)
    """
    return (((2 * PLANCK * pow(frequency, 3))
             / pow(LIGHT_SPEED, 2)) 
            * (1 / (np.exp((PLANCK * frequency) / (BOLTZMANN * temperature))
                  - 1)))

def spectral_radiance_rayleigh_jeans(frequency, temperature):
    """Rayleigh-Jean's law of black-body radiation.
    
    Computes spectral radiance (energy emitted per second per m2 per steradian per Hz)
    using Rayleigh-Jean's law.
    
    Args:
        frequency: Frequency of radiation in Hz
        temperature: Temperature of blackbody in Kelvin
    Returns:
        Spectral radiance at frequency given (Wm-2sr-1Hz-1)
    """
    return (2 * pow(frequency, 2) * BOLTZMANN * temperature) / pow(LIGHT_SPEED, 2)

def monochromatic_luminosity_planck(frequency, temperature, radius):
    """Compute frequency dependent luminosity of a star based on Planck's law of
    blackbody radiation.
    
    Planck's law integrated over area and all solid angles.
    
    Args:
        frequency: Frequency of radiation in Hz
        temperature: Temperature of blackbody in Kelvin
        radius: Radius of spherical blackbody
    Returns:
        Luminosity at the surface of a star at frequency given (Wm-2Hz-1)
    """
    return 4 * pow(math.pi, 2) * pow(radius, 2) * spectral_radiance_planck(frequency, temperature)

def monochromatic_luminosity_rayleigh_jeans(frequency, temperature, radius):
    """Compute frequency dependent luminosity of a star based on Rayleigh-Jeans' law.
    
    Planck's law integrated over area and all solid angles.
    Works for low frequencies only.
    
    Args:
        frequency: Frequency of radiation in Hz
        temperature: Temperature of blackbody in Kelvin
        radius: Radius of spherical blackbody
    Returns:
        Luminosity at the surface of a star at frequency given
    """
    return 4 * pow(math.pi, 2) * pow(radius, 2) * spectral_radiance_rayleigh_jeans(frequency, temperature)

def flux_from_luminosity(luminosity, distance):
    """Compute total flux from luminosity of a star.
    
    Args
        luminosity: luminosity at surface of a star (Wm-2)
        distance: distance from star (metres)
    """
    return luminosity / surface_area_sphere(distance)

def luminosity_from_mass(solar_masses):
    """Compute luminosity of a star from its mass.
    
    Args:
        solar_masses: mass of star in solar masses
    Returns:
        Luminosity (Wm-2)
    """
    return pow(solar_masses, 3.5)

def luminosity_from_temperature(temperature, radius):
    """Compute total luminosity of a star using Stefan-Boltzmann's law.
    """
    return stefan_boltzmann_law() * surface_area_sphere(radius) * pow(temperature, 4)

def power_from_star(flux_from_star, diameter_mirror):
    """Compute power (W) received by a mirror.
    
    Args:
        flux_from_star: Radiation flux crossing detector per unit area (Wm-2)
        diameter_mirror: diameter of the mirror (m)
    Returns:
        Power received by mirror (watts)
    """
    return flux_from_star * pow(diameter_mirror, 2)