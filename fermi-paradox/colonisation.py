import math
import numpy as np
import pandas as pd
import plotly.express as px

def scale_milky_way(num_stars_to_model = None):
    radius_milky_way = 105700/2
    thin_disk_height = 1470/2
    thick_disk_height = 8500/2
    area_milky_way_disk = math.pi * radius_milky_way**2
    num_stars_in_milky_way = 100e9
    num_g_stars = num_stars_in_milky_way * 0.07
    num_habitable_stars = int(num_g_stars * 0.25)
    
    print(f"There are around {num_habitable_stars} habitable star systems in the Milky Way")

    if not num_stars_to_model: 
        print("Keeping full sized Milky Way")
        num_stars_to_model = num_habitable_stars

    scaling_factor = int(num_habitable_stars/num_stars_to_model)
    print(f"Scaling galaxy to host {num_stars_to_model} stars...")
    print(f"Reducing volume of Milky Way by a factor of {scaling_factor}")
    
    galaxy_thickness = thick_disk_height / scaling_factor**(1./3.)
    galaxy_radius = radius_milky_way / scaling_factor**(1./3.)
    volume_milky_way = area_milky_way_disk * thick_disk_height
    
    galaxy_volume = math.pi * galaxy_radius**2 * galaxy_thickness
    assert math.ceil(volume_milky_way / galaxy_volume) == scaling_factor

    return galaxy_radius, galaxy_thickness, num_stars_to_model

def generate_galaxy(num_stars, radius, thickness, seed=1):
    print(f"Generating galaxy with radius {radius} and thickness {thickness} containing {num_stars} stars...")
    x_coords, y_coords = normal_distribution_on_circle(num_stars, radius, seed)
    z_coords = np.random.default_rng(seed+2).normal(loc=0, scale=thickness * 0.34, size=num_stars)
    galaxy = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords})
    galaxy['state'] = 'unknown'
    galaxy['colony_age'] = 0
    galaxy['time_to_colonisation'] = np.nan
    galaxy['year_of_colonisation'] = np.nan
    return galaxy

def normal_distribution_on_circle(num_stars, radius, seed):
    random_angles = 2 * math.pi * np.random.default_rng(seed).random(size=num_stars)
    radii = np.sqrt(np.random.default_rng(seed+1).normal(loc=0, scale=radius * 0.34, size=num_stars)**2)
    x_coords = radii * [math.cos(angle) for angle in random_angles]
    y_coords = radii * [math.sin(angle) for angle in random_angles]
    return x_coords, y_coords

def evolve_eti(seed, star_map):
    eti_star = star_map.sample(random_state=seed).index[0]
    eti_x = star_map.loc[eti_star, 'x']
    eti_y = star_map.loc[eti_star, 'y']
    eti_z = star_map.loc[eti_star, 'z']
    distance_from_galactic_centre = math.sqrt(eti_x**2 + eti_y**2 + eti_z**2)
    star_map.loc[eti_star, 'state'] = 'eti'
    print(f"ETI evolves on a star {distance_from_galactic_centre} from galactic centre ({eti_x}, {eti_y}, {eti_z})")
    return star_map, distance_from_galactic_centre

def plot_galaxy(star_map):
    axes_max = np.max(star_map[['x', 'y', 'z']])
    fig = px.scatter_3d(star_map, x='x', y='y', z='z',
                        color='state',
                        opacity=0.6
                        )
    fig.update_traces(marker_size=2)
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=10, range=[-axes_max, axes_max],),
            yaxis = dict(nticks=10, range=[-axes_max, axes_max],),
            zaxis = dict(nticks=10, range=[-axes_max, axes_max],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    return fig

def colonise_galaxy(star_map, search_radius, travel_speed, max_num_targets=None, mission_success_rate=1.0):
    missions = plan_new_missions(star_map, search_radius, travel_speed, max_num_targets)
    star_map = update_starmap(missions, star_map)
    print(f"{len(missions)} new missions launched from newly emerged ETI!")
    year = 1
    colonising = True 
    while colonising:
        star_map = progress_missions(star_map, year)
        new_missions = plan_new_missions(star_map, search_radius, travel_speed, max_num_targets)
        if new_missions:
            print(f"{len(new_missions)} new missions launched in year {year}!")
            star_map = update_starmap(new_missions, star_map)
        if len(star_map[star_map['state'] == 'target']) == 0:
            num_stars_colonised = len(star_map[star_map['state'] == 'colonised'])
            num_stars_unexplored = len(star_map[star_map['state'] == 'unknown'])
            print(f"""Year {year}: No more missions can be launched!
                       {num_stars_colonised} stars have been colonised.
                       {num_stars_unexplored} stars are unreachable.""")
            colonising = False
        year = year + 1

    return star_map

def find_neighbours(star_dict, radius, star_map):
    max_x = star_map['x'] - star_dict['x']
    max_y = star_map['y'] - star_dict['y']
    max_z = star_map['z'] - star_dict['z']
    nearby_stars = star_map[(max_x < radius) &
                            (max_y < radius) &
                            (max_z < radius)
                           ].copy()
    nearby_stars['distance'] = np.sqrt(max_x**2 + max_y**2 + max_z**2)
    return nearby_stars[nearby_stars['distance'] < radius].sort_values(by='distance')

def find_targets(neighbours, max_num_targets=None):
    targets = neighbours[neighbours['state'] == 'unknown'].copy()
    if max_num_targets:
        targets = uninhabited_neighbours.head(max_num_targets)
    return targets

def time_to_colonisation(targets, travel_speed):
    targets['time_to_colonisation'] = np.ceil(targets.loc[:, 'distance'] / travel_speed).astype(int)
    targets = targets.reset_index(names = 'target_star')[['target_star', 'time_to_colonisation']]
    targets.set_index('target_star', inplace=True)
    return targets.to_dict(orient='dict')

def plan_new_missions(star_map, search_radius, travel_speed, max_num_targets):
    inhabited_states = ['eti', 'colonised']
    new_missions = {}
    newly_colonised_stars = star_map[(star_map['state'].isin(inhabited_states)) &
                                     (star_map['colony_age']==0)].to_dict(orient='index')
    num_newly_colonised_stars = len(newly_colonised_stars)
    if num_newly_colonised_stars > 0:
        print(f"Finding neighbours for {num_newly_colonised_stars} newly colonised star{'s' if num_newly_colonised_stars > 1 else ''}")
        neighbours = {star_id: find_neighbours(data, search_radius, star_map) for star_id, data in newly_colonised_stars.items()}
        targets = {star_id: find_targets(stars, max_num_targets) for star_id, stars in neighbours.items()}
        new_missions = [time_to_colonisation(stars, travel_speed)['time_to_colonisation'] for star_id, stars in targets.items()]
        new_missions = {star: time for mission in new_missions for star, time in mission.items()}
    return new_missions

def update_starmap(new_missions, star_map):
    for star_id, time_to_colonisation in new_missions.items():
        star_map.loc[star_id, 'time_to_colonisation'] = time_to_colonisation
        star_map.loc[star_id, 'state'] = 'target'
    return star_map

def progress_missions(star_map, year, mission_success_rate=1.0):
    star_map.loc[:, 'time_to_colonisation'] = star_map.loc[:, 'time_to_colonisation'] - 1
    star_map.loc[star_map['state'].isin(['colonised', 'eti']), 'colony_age'] = star_map.loc[star_map['state'].isin(['colonised', 'eti']), 'colony_age'] + 1
    num_stars_reached = len(star_map.loc[star_map['time_to_colonisation'] == 0])
    if num_stars_reached > 0:
        print(f"{num_stars_reached} stars reached by ETI in year {year}")
        star_map = colonise_stars(star_map, num_stars_reached, mission_success_rate, year)
    return star_map

def colonise_stars(star_map, num_stars_reached, mission_success_rate, year):
    missions_succeeded = np.random.default_rng(seed=None).binomial(1, mission_success_rate, num_stars_reached).astype(bool)
    stars_reached = star_map.loc[star_map['time_to_colonisation'] == 0].index
    stars_colonised = stars_reached[missions_succeeded]
    stars_failed = stars_reached[~missions_succeeded]
    print(f"{len(stars_failed)} missions failed")

    star_map.loc[stars_colonised, 'state'] = 'colonised'
    star_map.loc[stars_colonised, 'year_of_colonisation'] = year
    
    star_map.loc[stars_failed, 'state'] = 'unknown'
    star_map.loc[stars_failed, 'time_to_colonisation'] = np.nan

    return star_map




