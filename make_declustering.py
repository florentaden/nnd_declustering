import numpy as np
import pandas as pd
from mpi4py import MPI
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
doublesize = MPI.DOUBLE.Get_size()

# -- all CPUs communicator
ALL_COMM = MPI.COMM_WORLD
size = ALL_COMM.Get_size()
rank, size = ALL_COMM.Get_rank(), ALL_COMM.Get_size()

# -- all CPUs inside node communicator
NODE_COMM = ALL_COMM.Split_type(MPI.COMM_TYPE_SHARED)
node_rank, node_size = NODE_COMM.rank, NODE_COMM.size

# -- get one relay node per node
value = (rank, node_rank)
values = np.array(ALL_COMM.allgather(value))
relay_ranks = values[values[:, 1] == 0, 0]
group = ALL_COMM.group.Incl(relay_ranks)
RELAY_COMM = ALL_COMM.Create_group(group)

if rank in relay_ranks:
    boss_rank, boss_size = RELAY_COMM.rank, RELAY_COMM.size

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
catalog_df = catalog_df.reindex(['year', 'month', 'day', 'hour', 'minute',
    'seconde', 'time', 'latitude', 'longitude', 'depth', 'magnitude'], axis=1)
catalog = np.array(catalog_df)

# -- apply declustering and give the children dataframe
children_df = catalog_df[1:]
number_of_children, number_of_columns = children_df.shape

""" shared children array (output) """
number_of_points = number_of_children * (number_of_columns + 5)
nbytes = doublesize * number_of_points
win = MPI.Win.Allocate_shared(nbytes if node_rank == 0 else 0, doublesize,
    comm=NODE_COMM)
buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.DOUBLE.Get_size()
buf = np.array(buf, dtype='B', copy=False)
children = np.ndarray(buffer=buf, dtype='d', shape=(number_of_children,
    number_of_columns+5))

if node_rank == 0:
    children.fill(0)
    children[:, :number_of_columns] = np.array(children_df)

ALL_COMM.Barrier()

for j, child in list(enumerate(children))[rank::size]: # for each event
    # potential parents
    k = (catalog[:, 6] < child[6])
    parents = np.zeros((k.sum(), number_of_columns+5))
    parents[:, :number_of_columns] = catalog[k]

    # compute temporal distance with all parents
    parents[:, 11] = child[6] - parents[:, 6]
    parents[:, 11] = parents[:, 11]*10**(-q*b_value*parents[:, 10])

    # compute physical distance with all parents
    parents[:, 12] = np.sqrt((child[8]-parents[:, 8])**2 + \
                 (child[7]-parents[:, 7])**2)
    parents[:, 12] = (parents[:, 12]**fractal_dimension)*10**(
        (q-1)*b_value*parents[:, 10])

    # compute nearest_neighbor_distance metric with all parents
    parents[:, 13] = parents[:, 11]*parents[:, 12]

    nearest_neighbor = np.argmin(parents[:, 13])
    children[j, 12] = parents[nearest_neighbor, 12]
    children[j, 11] = parents[nearest_neighbor, 11]
    children[j, 13] = parents[nearest_neighbor, 13]
    children[j, 14] = parents[nearest_neighbor, 10]
    children[j, 15] = nearest_neighbor

ALL_COMM.Barrier()

if rank in relay_ranks:
    if rank == 0:
        target = np.zeros_like(children)
    else:
        target = None

    RELAY_COMM.Reduce([children, MPI.DOUBLE], [target, MPI.DOUBLE], op=MPI.SUM,
        root=0)

if rank == 0:
    columns = ['year', 'month', 'day', 'hour', 'minute', 'seconde', 'time',
              'latitude', 'longitude', 'depth', 'magnitude',
              'Tij', 'Rij', 'Nij', 'parent_magnitude', 'neighbor']
    children_df = pd.DataFrame(target, columns=columns)
    children_df.to_hdf(output_name + '.h5', 'table')
