# Fast Efficient Reaction Network (FERN) Solver

## Prerequisites
You will need git and cmake to build FERN.

## Checkout and build

From a shell, execute the following commands:


git clone https://github.com/jayjaybillings/fern
mkdir fern-build
cd fern-build
cmake ../fern -DCMAKE_BUILD_TYPE=Debug -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.5
make

## Running FERN

You can run FERN from any directory and it only requires one of its INI files
to run. So, assuming you are in the build directory and using one of the test
files in the data directory, you would run

./fern-exec ../fern/data/alpha.ini

## Questions
Questions can be directed to me at jayjaybillings <at> gmail <dot> com.
