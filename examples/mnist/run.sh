#!/bin/bash
mpirun -np 3 -hostfile examples/mnist/hostfile ./build/singa \
	-cluster_conf=examples/mnist/cluster.conf -model_conf=examples/mnist/mlp.conf


