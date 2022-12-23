"""Formulae from Cocconi & Morisson 1959.

   G. Cocconi and P. Morrison, “Searching for Interstellar Communications,” Nature, vol. 184, pp. 844–846, 1959.
"""

import math

from utils import light_years_to_metres
from constants import SUN_RADIUS, LIGHT_SPEED

def luminosity_sunlike(frequency):
    """Compute luminosity at given frequency from sun-like star (Cocconi & Morrison 1959).
    
    Args:
        frequency: frequency of radiation (Hz)
    """
    return pow(10, -15) * pow(frequency, 2) * 4 * math.pi

def flux_sunlike(frequency, distance):
    """Compute flux at given frequency from sun-like star (Cocconi & Morrison 1959).
    
    Args:
        frequency: frequency of radiation (Hz)
        distance: distance from star (metres)
    """
    return luminosity_sunlike(frequency) / pow(distance, 2)

def flux_from_galaxy(frequency, diameter_mirror=100):
    """Compute background flux at given frequency from the galaxy (Cocconi & Morrison 1959).
    
    Args:
        frequency: frequency of radiation (Hz)
        diameter_mirror: diameter of the detector (metres)
    """
    return (pow(10, -12.5) / frequency) * pow((LIGHT_SPEED / (diameter_mirror * frequency)), 2)

def power_from_galaxy(frequency, diameter_detector):
    """Compute background power at given frequency from the galaxy (Cocconi & Morrison 1959).
    
    Args:
        frequency: frequency of radiation (Hz)
        diameter_detector: diameter of the detector (m)
    """
    return flux_from_galaxy(frequency, diameter_mirror=diameter_detector) * pow(diameter_detector, 2)

def frequency_minimum(light_years_to_star, diameter_detector, constant=1e4):
    """Compute the minimum frequency for a SETI signal to be distinguished above background radiation.
    
    Accounts for radiation from star and radiation from the galaxy (Cocconi & Morrison 1959).
    
    Args:
        light_years_to_star: distance to star in light years
        diameter_detector: diameter of the detector (m)
    """
    distance_to_star =  light_years_to_metres(light_years_to_star)
    return constant * pow((distance_to_star / diameter_detector), 0.4)
