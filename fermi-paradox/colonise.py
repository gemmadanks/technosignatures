import math
import numpy as np
import pandas as pd
import plotly.express as px

def scale_milky_way(num_habitable_stars, num_stars_to_model=None):
    """Scale galaxy using size of the Milky Way.
    
    The Milky Way is assumed to have a diameter of 105700 light years,
    a thin disk height of 1470 light years and a thick disk height of
    8500 light years.

    Args:
        num_habitable_stars (int): number of stars at full scale (in
                                   the Milky Way)
        num_stars_to_model (int): number of stars to model (if None
                                  will use num_habitable_stars)
    Returns:
        galaxy_radius: radius of scaled galaxy
        galaxy_thickness: thickness of scaled galaxy
        num_stars_to_model: number of stars in scaled galaxy
    """
    milky_way = {'radius': 105700/2, 'thin_disk_height': 1470/2, 'thick_disk_height': 8500/2}
    area_milky_way_disk = math.pi * milky_way['radius']**2

    if not num_stars_to_model: 
        print("Keeping full-sized Milky Way")
        num_stars_to_model = num_habitable_stars

    scaling_factor = int(num_habitable_stars/num_stars_to_model)
    print(f"Scaling galaxy to host {num_stars_to_model} stars...")
    print(f"Reducing volume of Milky Way by a factor of {scaling_factor}")
    
    galaxy_thickness = milky_way['thick_disk_height'] / scaling_factor**(1./3.)
    galaxy_radius = milky_way['radius'] / scaling_factor**(1./3.)
    volume_milky_way = area_milky_way_disk * milky_way['thick_disk_height']
    
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

def colonise_galaxy(star_map, search_radius, travel_speed,
                    max_num_targets=None, mission_success_rate=1.0,
                    planning_time=0, longevity=None):
    print("ETI is ready to colonise the galaxy...")
    missions = plan_new_missions(star_map, search_radius, travel_speed, max_num_targets)
    star_map = update_starmap(missions, star_map)
    print(f"{len(missions)} new missions launched from newly emerged ETI!\n")
    year = 1
    colonising = True 
    while colonising:
        star_map = progress_colonies(star_map, year, longevity)
        star_map = progress_missions(star_map, year)    
        new_missions = plan_new_missions(star_map, search_radius, travel_speed, max_num_targets)
        if new_missions:
            print(f"{len(new_missions)} new missions launched in year {year}!")
            star_map = update_starmap(new_missions, star_map)
            num_stars_colonised = len(star_map[star_map['state'] == 'colonised'])
            progress = round(100 * (num_stars_colonised/len(star_map)), 1)
            print(f"\n{progress}% of galaxy colonised...\n")
        
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
    neighbours = nearby_stars[(nearby_stars['distance'] < radius) & (nearby_stars['distance'] > 0)].sort_values(by='distance')
    if neighbours.empty:
        print(f"No neighbours found...")
    else:
        print(f"{len(neighbours)} neighbours found...")
    return neighbours

def find_targets(neighbours, max_num_targets=None):
    inhabited = ['eti', 'colonised', 'extinct']
    targets = pd.DataFrame(data=None, columns=neighbours.columns)

    if not neighbours.empty:
        num_inhabited_neighbours = len(neighbours[neighbours['state'].isin(inhabited)])
        num_targetted_neighbours = len(neighbours[neighbours['state'] == 'target'])
        if num_inhabited_neighbours > 0:
            print(f"{num_inhabited_neighbours} neighbouring stars already colonised!")
        if num_targetted_neighbours > 0:
            print(f"{num_targetted_neighbours} neighbouring stars already targetted!")
        targets = neighbours[neighbours['state'] == 'unknown'].copy()
        print(f"Found {len(targets)} possible target stars")
        if not targets.empty and max_num_targets:
            targets = targets.head(max_num_targets)
    return targets

def time_to_colonisation(targets, travel_speed):
    targets['time_to_colonisation'] = np.ceil(targets.loc[:, 'distance'] / travel_speed).astype(int)
    targets = targets.reset_index(names = 'target_star')[['target_star', 'time_to_colonisation']]
    targets.set_index('target_star', inplace=True)
    return targets.to_dict(orient='dict')

def plan_new_missions(star_map, search_radius, travel_speed, max_num_targets, planning_time=0):
    inhabited = ['eti', 'colonised']
    new_missions = {}
    stars_ready_for_missions = star_map[(star_map['state'].isin(inhabited)) &
                                        (star_map['colony_age'] == planning_time)].to_dict(orient='index')
    num_stars_ready_for_missions = len(stars_ready_for_missions)
    if num_stars_ready_for_missions > 0:
        print(f"Finding neighbours for {num_stars_ready_for_missions} star{'s' if num_stars_ready_for_missions > 1 else ''} ready to launch new missions")
        neighbours = {star_id: find_neighbours(data, search_radius, star_map) for star_id, data in stars_ready_for_missions.items()}
        targets = {star_id: find_targets(stars, max_num_targets) for star_id, stars in neighbours.items()}
        new_missions = [time_to_colonisation(stars, travel_speed)['time_to_colonisation'] for star_id, stars in targets.items() if not stars.empty]
        new_missions = {star: time for mission in new_missions for star, time in mission.items()}
    return new_missions

def update_starmap(new_missions, star_map):
    for star_id, time_to_colonisation in new_missions.items():
        star_map.loc[star_id, 'time_to_colonisation'] = time_to_colonisation
        star_map.loc[star_id, 'state'] = 'target'
    return star_map

def progress_colonies(star_map, year, longevity=None):
    stars_with_colonies = star_map.loc[star_map['state'].isin(['colonised', 'eti'])].index
    star_map.loc[stars_with_colonies, 'colony_age'] = star_map.loc[stars_with_colonies, 'colony_age'] + 1
    if longevity:
        extinct_colonies = (star_map['state'].isin(['colonised', 'eti'])) & (star_map['colony_age'] >= longevity)
        num_extinct_colonies = len(star_map[extinct_colonies])
        if num_extinct_colonies > 0:
            print(f"{num_extinct_colonies} colonies have gone extinct")
            star_map.loc[extinct_colonies, 'state'] = 'extinct'
    return star_map

def progress_missions(star_map, year, mission_success_rate=1.0):
    star_map.loc[:, 'time_to_colonisation'] = star_map.loc[:, 'time_to_colonisation'] - 1
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

def report(star_map):
    num_stars = len(star_map)
    num_unexplored_stars = len(star_map[star_map['state'] == 'unknown'])
    if num_unexplored_stars < num_stars - 1:
        colonisation_timespan = int(star_map['year_of_colonisation'].max())
        
        num_extinct_stars = len(star_map[star_map['state'] == 'extinct'])
        num_surviving_stars = len(star_map[star_map['state'] == 'colonised'])
        print(f"""Colonisation took {colonisation_timespan} years and left {num_unexplored_stars} stars unexplored. 
                  {num_surviving_stars} colonies survive.
                  {num_extinct_stars} colonies went extinct.""")
        if num_stars < 1e5:
            plot_galaxy(star_map).show()
            px.histogram(star_map, x='year_of_colonisation').show()
        else:
            print("Plotting results for a sample of 10,000 stars...")
            plot_galaxy(star_map.sample(10000)).show()
            px.histogram(star_map.sample(10000), x='year_of_colonisation').show()
    else:
        print("Colonisation failed!")



