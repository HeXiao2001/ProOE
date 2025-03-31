import pandas as pd
import geopandas as gpd
from model.model import ProOE
import argparse
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

parser = argparse.ArgumentParser(description='ProOE')
parser.add_argument('--dataset', type=str, default='heterogeneity', help='The dataset to use')   # Four demo datasets: 1. NYC_taxi; 2. continuous; 3. heterogeneity; 4. fuzziness.
parser.add_argument('--Flows_suffix', type=str, default='', help='The suffix of the Flows file would be used')  # Consider Flows of the same area at different times, such as NYC yellow taxi data _2023, _2018, _2013   
parser.add_argument('--flow_threshold', type=int, default=0, help='The threshold of flow to filter out the flows')  # 0

parser.add_argument('--unitid_name', type=str, default='LocationID', help='The name of the column that contains the unit id in spatialunit.geojson')
parser.add_argument('--Oid_name', type=str, default='O_id', help='The name of the column that contains the origin id in flow.csv|distance.csv')
parser.add_argument('--Did_name', type=str, default='D_id', help='The name of the column that contains the destination id in flow.csv|distance.csv')
parser.add_argument('--flow_name', type=str, default='flow', help='The name of the column that contains the flow in flow.csv')
parser.add_argument('--distance_name', type=str, default='distance', help='The name of the column that contains the distance in distance.csv')

parser.add_argument('--max_iter', type=int, default=200, help='The maximum number of iterations to fit the model')
parser.add_argument('--min_iter', type=int, default=20, help='The minimum number of iterations to fit the model')
parser.add_argument('--num_realizations', type=int, default=2, help='The number of realizations to run')

parser.add_argument('--K', type=int, default=4, help='The number of communities to detect') # None or int  # None: the number of communities is determined by the model itself; int: the number of communities is fixed.
parser.add_argument('--max_k', type=int, default=20, help='The maximum number of communities to detect(when K is None)')  # 20

parser.add_argument('--spatial_continuity_method', type=str, default='Adj', help='The method for spatial continuity constrains, including geographic adjacency(Adj) and spatial clusting(HDBSCAN)')  # Adj, HDBSCAN
parser.add_argument('--alpha', type=float, default=2, help='The parameter used for spatial continuity guidance')  # 2

parser.add_argument('--seed', type=int, default=None, help='The random seed to use')   # None
parser.add_argument('--verbose', type=bool, default=True, help='Whether to display detailed information during runtime.') # True or False
args = parser.parse_args()

if __name__ == '__main__':
    
    print('Reading raw data...')
    Vs = gpd.read_file(f'./data/input/{args.dataset}/SpatialUnit.geojson')
    Vs[args.unitid_name] = Vs[args.unitid_name].astype(str)
    Es = pd.read_csv(f'./data/input/{args.dataset}/Flows' + args.Flows_suffix + '.csv', dtype={args.Oid_name: str, args.Did_name: str, args.flow_name: float})
    Ds = pd.read_csv(f'./data/input/{args.dataset}/Distances.csv', dtype={args.Oid_name: str, args.Did_name: str , args.distance_name: float})
    
    # Data preprocessing, generate edge(trip) table, distance table based on the original input data, and filter out valid data (with trip)
    print('Data preprocessing...')
    utils = SourceFileLoader('utils', './model/utils.py').load_module()
    Es_p, Distances_p, Adj_p, Vs_p, Vs_coord_p = utils.pre_data(Vs, Es, Ds, args)   

    print('Running ProOE...')
    ProOE_model = ProOE(Es_p, Distances_p, Adj_p, Vs_p, Vs_coord_p, args.K, max_iter=args.max_iter, spatial_continuity_method = args.spatial_continuity_method, alpha = args.alpha, min_iter= args.min_iter, max_k=args.max_k, num_realizations=args.num_realizations, seed=args.seed, verbose=args.verbose)

    print('Saving results...')
    ProOE_model.save_result(Vs, args.unitid_name,f'./data/output/{args.dataset}/', args.Flows_suffix+'_'+str(args.K))
