Lapis
=====

Distributed deep learning system


Coding
===
Please follow the google's c plus plus coding style, including the naming,
header files, formatting, comments, etc.
http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

Use astyle to format code.
The cpplint.py can help to detect lines that do not follow the style.

External libraries include gflag (commandline parsing), glog (logging),
google-protobuf (configuration), gtest (unit test), boost (filesystem, thread),
mpi, opencv (read image, only used in rgb_dir_source.cc).

gtest usage:
g++ -isystem ${GTEST_DIR}/include -pthread path/to/your_test.cc libgtest.a \
-o your_test

C++11 features are used.
