There are 3 versions of the code (both py and ipynb): original serial DFO algorithm, partially parallelized DFO algorithm and fully parallelized DFO algorithm

There are also 2 alternatives for running the code:

1. Using local machine if a Nvidia GPU is available 
 1.1 You need to have Numba 0.56.4 installed
 1.2 You need to have CUDA toolkit 11.8 installed 
 1.3 Run it in a Pythen IDE or local Jupyter Notebook

2. Using Google Collab - upload all 3 ipynb files (all versions of the algorithm):
 2.1 You need to change the runtime type to a GPU in the Runtime menu
 2.1 You might need to run each algorithm individually because collab limits the runtime with a GPU to one at a time
 2.1 What can be done is open another separate notebook with the GPU runtime and copy paste form the other files as needed

This is my dissertation. A serial natural computing algorithm was modified to improve execution times by taking advantage of the parallel capabilities of GPUs.