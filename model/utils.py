import numpy as np
import pandas as pd
import geopandas as gpd


DEFAULT_INF = 1e20  # Infinity

def get_centriod(geodataframe):
    """
    Get the centroid of geometric objects

    Parameters:
    geodataframe: Spatial dataframe (type: geodataframe, including geometry column)

    Returns:
    centroid: Centroid (type: geodataframe, including geometry column)
    """
    # Copy GeoDataFrame to avoid modifying the original data
    centroid = geodataframe.copy()
    # Check if the GeoDataFrame is already in a projected coordinate system
    if centroid.crs.is_geographic:
    #     # Convert to a commonly used projected coordinate system, such as WGS 84 / Pseudo-Mercator
        centroid = centroid.to_crs(epsg=3857)
    centroid['geometry'] = centroid['geometry'].centroid
    return centroid

def pre_data(Vs, Es, Ds, args):
    """
    Data preprocessing

    Parameters:
    Vs: Set of nodes (node vector file type: geodataframe including unit_id, geometry columns; unit_id is of text type; geometry is point)
    Es: Set of edges (OD flow type: dataframe including O_id, D_id, flow; O_id, D_id correspond to unit_id in Vs, type is str; flows are OD flows)
    Ds: Distance between nodes (distance type: dataframe, including O_id, D_id, distance columns; O_id, D_id correspond to unit_id in Vs, type is str; distance is the distance between OD)
    args: Parameters (type: argparse.ArgumentParser, including K, dataset, unitid_name, Oid_name, Did_name, flow_name, distance_name, seed)
    """

    if args.verbose:
        print('Raw data loaded...')
        print('  --Nodes:', len(Vs))
        print('  --Edges:', len(Es))
        print('  --Flows:', Es['flow'].sum())

    # Rename the original columns in Vs, Es, HEs, Ds to unit_id, O_id, D_id, flow, distance
    Vs = Vs.rename(columns={args.unitid_name: 'unit_id'})
    Es = Es.rename(columns={args.Oid_name: 'O_id', args.Did_name: 'D_id', args.flow_name: 'flow'})
    Ds = Ds.rename(columns={args.Oid_name: 'O_id', args.Did_name: 'D_id', args.distance_name: 'distance'})
    
    # Generate an empty dataframe with three columns: O_id, D_id, data, to store the distance/flow between OD
    # O_id, D_id are calculated based on the unit_id of Vs, which are all possible combinations
    df = pd.DataFrame()
    df['O_id'] = np.repeat(Vs['unit_id'].values, len(Vs['unit_id']))
    df['D_id'] = np.tile(Vs['unit_id'].values, len(Vs['unit_id']))
    df['data'] = 0.0
    # Remove records where O_id is equal to D_id
    df = df[df['O_id'] != df['D_id']].reset_index(drop=True)

    # Calculate the adjacency matrix of the Vs if Vs is polygon
    if args.verbose:
        print('  --Calculating adjacency matrix...')
    # Generate an empty dataframe with three columns: O_id, D_id, data, to store the adjacency matrix
    adj_df = df.copy(deep=True)
    # Use merge to speed up data filling
    adj_df = adj_df.merge(Vs[['unit_id', 'geometry']], left_on='O_id', right_on='unit_id', how='left')
    adj_df = adj_df.merge(Vs[['unit_id', 'geometry']], left_on='D_id', right_on='unit_id', how='left')
    adj_df['data'] = adj_df.apply(lambda x: x['geometry_x'].touches(x['geometry_y']), axis=1)
    adj_df = adj_df.drop(columns=['geometry_x', 'geometry_y', 'unit_id_x', 'unit_id_y'])

    if args.verbose:
        print('  --Calculating interaction table...')
    # Create a new DataFrame to store interaction data
    interaction_df = df.copy(deep=True)
    # Use merge to speed up data filling
    interaction_df = interaction_df.merge(Es[['O_id', 'D_id', 'flow']], on=['O_id', 'D_id'], how='left')
    interaction_df['flow'] = interaction_df['flow'].fillna(0)
    interaction_df['data'] += interaction_df['flow']
    interaction_df.drop(columns=['flow'], inplace=True)

    # Handle reverse OD pairs
    reverse_Es = Es.rename(columns={'O_id': 'D_id', 'D_id': 'O_id'})
    interaction_df = interaction_df.merge(reverse_Es[['O_id', 'D_id', 'flow']], on=['O_id', 'D_id'], how='left')
    interaction_df['flow'] = interaction_df['flow'].fillna(0)
    interaction_df['data'] += interaction_df['flow']
    interaction_df.drop(columns=['flow'], inplace=True)

    if args.verbose:
        print('  --Calculating distance table...')
    # Create a new DataFrame to store distance data
    distance_df = df.copy(deep=True)
    # Use merge to speed up data filling
    distance_df = distance_df.merge(Ds[['O_id', 'D_id', 'distance']], on=['O_id', 'D_id'], how='left')
    distance_df['distance'] = distance_df['distance'].fillna(0)
    distance_df['data'] = distance_df['distance']

    distance_df.drop(columns=['distance'], inplace=True)

    # Handle reverse OD pairs
    reverse_Ds = Ds.rename(columns={'O_id': 'D_id', 'D_id': 'O_id'})
    distance_df = distance_df.merge(reverse_Ds[['O_id', 'D_id', 'distance']], on=['O_id', 'D_id'], how='left')
    distance_df['distance'] = distance_df['distance'].fillna(0)
    distance_df['data'] = np.where(distance_df['distance'] > 0, distance_df['distance'], distance_df['data'])
    distance_df.drop(columns=['distance'], inplace=True)

    Vs_centroid = get_centriod(Vs)
    # Data filtering, only filter out nodes and OD pairs with OD flow
    if args.verbose:
        print('  --Data filtering...')

    interaction_df, distance_df, adj_df, Vs_flited = flit_data(Vs, interaction_df, distance_df, adj_df, args)

    # Get the coordinate array corresponding to each Vs_flited
    # Extract coordinates from Vs_centroid according to the order of Vs_flited
    # First convert the coordinate system of Vs_centroid to 4326
    Vs_centroid = Vs_centroid.to_crs(epsg=3857)
    Vs_coord = []
    for unit_id in Vs_flited:
        point = Vs_centroid[Vs_centroid['unit_id'] == unit_id]['geometry'].values[0]
        Vs_coord.append([point.x, point.y])
    if args.verbose:
        print('Data preprocessing completed')
        print('  --Total nodes:', len(Vs_flited))
        print('  --Edges:', len(interaction_df))
        print('  --Flows:', interaction_df['data'].sum())
        print('  --Coordinates of nodes with flows:', len(Vs_coord))
    return interaction_df, distance_df, adj_df, Vs_flited, Vs_coord


def flit_data(Vs, interaction_df, distance_df, adj_df, args):
    """
    Data filtering, only filter out nodes and OD pairs with OD flow

    Parameters:
    Vs: Set of nodes (node vector file type: geodataframe including unit_id, geometry columns; unit_id is of text type; geometry is point)
    interaction_df: Interaction table (type: dataframe, including O_id, D_id, data columns)
    distance_df: Distance table (type: dataframe, including O_id, D_id, data columns)

    Returns:
    interaction_df: Interaction table (type: dataframe, including O_id, D_id, data columns)
    distance_df: Distance table (type: dataframe, including O_id, D_id, data columns)
    Vs_flited: Filtered nodes (type: list)
    """

    if args.verbose:
        print('    --Filtering OD interactions equal to 0...')
    # Extract 'O_id', 'D_id' pairs with interactions less than the threshold from interaction_df
    zero_interaction = interaction_df[interaction_df['data'] <= args.flow_threshold][['O_id', 'D_id']]
    # Extract records in interaction_df corresponding to zero_interaction
    interaction_df = interaction_df[~interaction_df[['O_id', 'D_id']].isin(zero_interaction).all(axis=1)]

    if args.verbose:
        print('    --Filtering OD interactions equal to 0...')
    distance_df = distance_df[~distance_df[['O_id', 'D_id']].isin(zero_interaction).all(axis=1)]

    if args.verbose:
        print('    --Filtering distance equal to 0...')
    # Extract 'O_id', 'D_id' pairs with distance equal to 0 from distance_df
    zero_distance = distance_df[distance_df['data'] == 0][['O_id', 'D_id']]
    # Extract records in distance_df corresponding to zero_distance
    distance_df = distance_df[~distance_df[['O_id', 'D_id']].isin(zero_distance).all(axis=1)]
    # Extract records in interaction_df corresponding to zero_distance
    interaction_df = interaction_df[~interaction_df[['O_id', 'D_id']].isin(zero_distance).all(axis=1)]

    if args.verbose:
        print('    --Filtering nodes with OD flow...')
    # Generate an empty dataframe with two columns: unit_id, data, to store the flow of nodes
    Vs_df = pd.DataFrame()
    Vs_df['unit_id'] = Vs['unit_id']
    Vs_df['data'] = 0

    # Calculate the total flow in the O_id direction
    o_flow = interaction_df.groupby('O_id')['data'].sum().reset_index()
    o_flow.columns = ['unit_id', 'o_data']

    # Calculate the total flow in the D_id direction
    d_flow = interaction_df.groupby('D_id')['data'].sum().reset_index()
    d_flow.columns = ['unit_id', 'd_data']

    # Merge flow data back to Vs_df
    Vs_df = Vs_df.merge(o_flow, on='unit_id', how='left').fillna(0)
    Vs_df = Vs_df.merge(d_flow, on='unit_id', how='left').fillna(0)

    # Calculate total flow
    Vs_df['data'] += Vs_df['o_data'] + Vs_df['d_data']

    # Filter out nodes with flow greater than 0
    Vs_df = Vs_df[Vs_df['data'] > 0]['unit_id']

    Vs_flited = list(set(Vs_df.values))

    # Filter interaction_df and distance_df, only keep data where both O_id and D_id are in Vs_df
    if args.verbose:
        print('    --Filtering interaction flow...')
    interaction_df = interaction_df[(interaction_df['O_id'].isin(Vs_flited)) & (interaction_df['D_id'].isin(Vs_flited))]

    if args.verbose:
        print('    --Filtering interaction distance...')
    distance_df = distance_df[(distance_df['O_id'].isin(Vs_flited)) & (distance_df['D_id'].isin(Vs_flited))]

    if args.verbose:
        print('    --Filtering adjacency matrix...')
    adj_df = adj_df[(adj_df['O_id'].isin(Vs_flited)) & (adj_df['D_id'].isin(Vs_flited))]

    # Ensure the order of O_id and D_id columns in interaction_df and distance_df is consistent
    interaction_df = interaction_df.sort_values(by=['O_id', 'D_id']).reset_index(drop=True)
    distance_df = distance_df.sort_values(by=['O_id', 'D_id']).reset_index(drop=True)
    adj_df = adj_df.sort_values(by=['O_id', 'D_id']).reset_index(drop=True)

    # Sort Vs_flited
    Vs_flited = sorted(Vs_flited)

    return interaction_df, distance_df, adj_df, Vs_flited