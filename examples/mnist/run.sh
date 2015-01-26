#!/bin/bash
mpirun -np 4 -hostfile examples/mnist/hostfile ./build/singa -server_threads=2\
	-cluster_conf=examples/mnist/cluster.conf -model_conf=examples/mnist/mlp.conf


