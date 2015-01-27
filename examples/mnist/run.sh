#!/bin/bash
mpirun -np 20 -hostfile examples/mnist/hostfile ./build/singa -server_threads=1 \
	-cluster_conf=examples/mnist/cluster.conf -model_conf=examples/mnist/mlp.conf


