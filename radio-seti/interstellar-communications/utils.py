"""Helper functions."""

import math

from constants import LIGHT_YEAR, LIGHT_SPEED

def light_years_to_metres(num_light_years):
    """Convert light years to metres.
    
    Args:
        num_light_years: number of light years
    Returns:
        Distance in metres
    """
    return num_light_years * LIGHT_YEAR

def wavelength_to_frequency(wavelength):
    """Convert wavelength to frequency.
    
    Args:
        wavelength: wavelength in metres
    Returns:
        Frequency in Hertz
    """
    return LIGHT_SPEED / wavelength

def surface_area_sphere(radius):
    """Comput surface area of a sphere.
    
    Args:
        radius: radius of sphere
    Returns:
        Surface area of sphere
    """
    return 4 * math.pi * pow(radius, 2)
