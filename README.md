# NND declustering
Nearest-Neighbor-Distance algorithm to decluster seismic catalog in Python
based on the work of [Baiesi & Paczuski 2004](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066106?casa_token=ZGclnOGPo3wAAAAA%3A4nuNCLLdPsX95_ctb2I9y-na1IjG0UcMKoTVCaLtRfr5ic3SM2FJDxW0T44k0pjBqHtyeXowO_IiDQ) and [Zaliapin et al. 2008](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.101.018501?casa_token=CsYTRxLKeFUAAAAA%3AJ4t3rSz_rdHaCOgfz_CHGfQRUTIGIo_doFkVckoPIkFIVygGwn_RNlUKlKaOOSDegL3H4e_lW5oWfQ).

The code ``make_declustering.py`` finds for each event of a given catalog its nearest neighbor. The utilisator is invited to select his own threshold to make the distinction between aftershocks and background events.

An example is provided with the [Southern California Earthquake Data Center](https://scedc.caltech.edu/eq-catalogs/) catalog from 2014 to 2019.

It requires the package [``mpi4py``](https://mpi4py.readthedocs.io/en/stable/) which can be simply install with:
```
conda install -c anaconda mpi4py
```
or
```
pip install mpi4py
```
for more information have a look [here](https://mpi4py.readthedocs.io/en/stable/install.html#).

The file needs to contain the following headers:
	- year
	- month
	- day
	- hour 
	- minute
	- second
	- time: first event is 0, can be second, day, etc..)
	- latitude
	- longitude
	- magnitude

The output will add the following columns:
	- Tij: the rescaled Time with the with the nearest neighbor
        - Rij: the rescaled Distance 
	- Nij: the nearest neighbor distance metric
	- parent_magnitude
	- neighbor: the index of the nearest neighbor

The code is able to run in serial:
```
python make_declustering.py
```
or in parallel on a laptop or a node on a cluster (>= 1 node):
```
mpirun -np <number of cpu desired> python make_declustering.py
```

![Aden-Antoniow2020](Figure.png)
