Lapis
=====

Distributed deep learning system


Coding
===
Please follow the google's c plus plus coding style, including the naming,
header files, formatting, comments, etc.
http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

Use astyle to format code.  The cpplint.py can help to detect lines that do not
follow the style. There is a target in Makefile, called flint. By running
`make flint', it will call astyle and cpplint.py automatically. There may be
some errors after cpplint.py, then you have to correct them manually. You can
also update the astyle.conf to configure the astyle to make it more smart to
follow the google style.

External libraries include gflag (commandline parsing), glog (logging),
google-protobuf (configuration), gtest (unit test).

gtest usage:
g++ -isystem ${GTEST_DIR}/include -pthread path/to/your_test.cc libgtest.a \
-o your_test

C++11 features are used.
