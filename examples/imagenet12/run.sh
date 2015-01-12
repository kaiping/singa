mpirun  -np 21 -hostfile examples/imagenet12/hostfile ./singa.bin \
	-system_conf=examples/imagenet12/system.conf \
  -model_conf=examples/imagenet12/model.conf
