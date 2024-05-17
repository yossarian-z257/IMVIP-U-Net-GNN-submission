# U-Net with GNN Bottleneck: A Novel Architecture for RANS Simulations in Airfoil Dynamics

To install these under linux run, use e.g.: 
```
sudo pip install torch numpy
sudo apt-get install openfoam11 gmsh
```
(Details can be found on the installation pages of [PyTorch](https://pytorch.org/get-started/locally/) and 
[OpenFOAM](https://openfoam.org/download/5-0-ubuntu/).)

## Data generation

Note that you can skip the next two steps if you download the training
data packages below. Simply make sure you have `data/train` and `data/test`
in the source directory, then you can continue with the training step.

### Download airfoils

First, enter the `data` directory. 
Download the airfoil profiles by running `./download_airfoils.sh`, this
will create `airfoil_database` and `airfoil_database_test` directories.
(The latter contains a subset that shouldn't be used for training.) The
airfoild database should contain 1498 files afterwards.

### Generate data

Now run `python ./dataGen.py` to generate a first set of 100 airfoils.
This script executes _openfoam_ and runs _gmsh_ for meshing the airfoil profiles. 

Once `dataGen.py` has finished, you should find 100 .npz files in a new
directory called `train`. You can call this script repeatedly to generate 
more data, or adjust
the `samples` variables to generate more samples with a single call. 
For a first test, 100 samples are sufficient, for higher quality models, more
than 10k are recommended..

Output files are saved as compressed numpy arrays. The tensor size in each
sample file is 6x128x128 with dimensions: channels, x, y. 
The first three channels represent the input,
consisting (in this order) of two fields corresponding to the freestream velocities in x and y
direction and one field containing a mask of the airfoil geometry as 
a mask. The last three channels represent the target, containing one pressure and two velocity
fields. 

## Test evaluation

To compute relative inference errors for a test data set, you can use the `./runTest.py` script.
By default, it assumes that the test data samples (with the same file format as the training samples)
are located in `../data/test`. Hence, you either have to generate data in a new directory with the
`dataGen.py` script from above, or download the test data set via the link below.

run `runTest.py` 

This code is heavilty borrowed ideas from 

[Deep-Flow-Predictions](https://github.com/thunil/Deep-Flow-Prediction) , and  
[Diffusion-based-Flow-Predictions](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction)

