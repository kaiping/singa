##Train LeNet on MNIST dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) has 60000 gray images. Each image is
of size 28*28 and represents one digit (i.e., 0,1,...,9). 50000 images are used
as training dataset to train a classifier for recognizing the digits in the rest
10000 images.

####Train on single node
####Data Preparation
1. Download the MNIST dataset to folder `$DATA/mnist/`

        cd $DATA/mnist
        wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

2. Create training data shard and test data shard. Goto SINGA code folder.

        cd singa/
        ./build/loader -datasource mnist -imagefile $DATA/mnist/train-images-idx3-ubyte -labelfile $DATA/mnist/train-labels-idx1-ubyte -shard_folder $DATA/mnist/train/
        ./build/loader -datasource mnist -imagefile $DATA/mnist/t10k-images-idx3-ubyte -labelfile $DATA/mnist/t10k-labels-idx1-ubyte -shard_folder $DATA/mnist/test/

    two files will be generated: `$DATA/mnist/train/shard.dat` for training, and `$DATA/mnist/test/shard.dat` for test.

####Model Configuration
We construct the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
as shown in the figure by setting the model configuration file, i.e., `example/mnist/model.conf`.

