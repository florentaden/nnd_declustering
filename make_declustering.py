import numpy as np
import pandas as pd
from mpi4py import MPI
from sys import exit

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()
doublesize = MPI.DOUBLE.Get_size()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
b_value = 1
fractal_dimension = 1.6
q = 0.5
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# -- southern california 1981-2011
catalog_name = 'southern_california_2014-2019'
output_name = catalog_name + '_nnd'
# -- read catalog
catalog_df = pd.read_hdf(catalog_name + '.h5', 'table')
catalog_df = catalog_df[catalog_df.magnitude >= 2]
catalog_df['x'] = catalog_df['longitude']
catalog_df['y'] = catalog_df['latitude']
catalog = np.array(catalog_df)

# -- apply declustering and give the children dataframe
children_df = catalog_df[1:]
number_of_children, number_of_columns = children_df.shape

""" shared children array (output) """
number_of_points = number_of_children * (number_of_columns + 5)
nbytes = doublesize * number_of_points
win = MPI.Win.Allocate_shared(nbytes, doublesize, comm=COMM)
buf, _ = win.Shared_query(0)
buf = np.array(buf, dtype='B', copy=False)
children = np.ndarray(buffer=buf, dtype='d', shape=(number_of_children, number_of_columns+5))
if rank == 0:
    children.fill(0)
    children[:, :number_of_columns] = np.array(children_df)

COMM.Barrier()

for j, child in list(enumerate(children))[rank::size]: # for each event
    # potential parents
    k = (catalog[:, 6] < child[6])
    parents = np.zeros((k.sum(), number_of_columns+5))
    parents[:, :number_of_columns] = catalog[k]

    # compute temporal distance with all parents
    parents[:, 13] = child[6] - parents[:, 6]
    parents[:, 13] = parents[:, 13]*10**(-q*b_value*parents[:, 10])

    # compute physical distance with all parents
    parents[:, 14] = np.sqrt((child[8]-parents[:, 8])**2 + \
                 (child[7]-parents[:, 7])**2)
    parents[:, 14] = (parents[:, 14]**fractal_dimension)*10**(
        (q-1)*b_value*parents[:, 10])

    # compute nearest_neighbor_distance metric with all parents
    parents[:, 15] = parents[:, 13]*parents[:, 14]

    nearest_neighbor = np.argmin(parents[:, 15])
    children[j, 14] = parents[nearest_neighbor, 14]
    children[j, 13] = parents[nearest_neighbor, 13]
    children[j, 15] = parents[nearest_neighbor, 15]
    children[j, 16] = parents[nearest_neighbor, 10]
    children[j, 17] = nearest_neighbor

if rank == 0:
    columns = ['year', 'month', 'day', 'hour', 'minute', 'seconde', 'time',
              'latitude', 'longitude', 'depth', 'magnitude', 'x', 'y',
              'Tij', 'Rij', 'Nij', 'parent_magnitude', 'neighbor']
    children_df = pd.DataFrame(children, columns=columns)
    children_df.to_hdf(output_name + '.h5', 'table')
