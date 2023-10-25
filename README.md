# DDPN

This repository provides tools and methodologies for computing the distributed homology of datasets and for the utilization of this data structure for downstream machine learning tasks. Key components include:

 - Distributed Homology Computation: Methods for extracting topological features from data.

 - Vectorization: Techniques to transform the computed homology into a format suitable for machine learning tasks.

 - Deep Set of Set Networks: Neural network architectures specifically designed to process the vectorized topological data.


The main.py file provides an example of how to use the code from this library to compute the distributed homology of a dataset of pointclouds and train a DDPN model using the resulting persistence diagrams.

## GNN inclusion

The GNN functionality can be added to the DSSN framework by simple insertion of a MessagePassingModel or other GNN-models in the 'outer_transform'-agument of the DSSN module. Additional info about edge_weights will have to be included as a parallel input, probably in the structure of an adjacency matrix.
