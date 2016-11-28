# Fast Efficient Reaction Network (FERN) Solver

# Student Code - Research Code Only! Not Software!

**NOTE** - This code was written primarily by students. It is of poor quality and
should not be used for anything more than basic research. An effort is underway to
rewrite it as part of (Fire)[https://github.com/jayjaybillings/fire].

## Prerequisites
You will need git and cmake to build FERN.

## Checkout and build

From a shell, execute the following commands:


```bash
git clone --recursive https://github.com/jayjaybillings/fern
mkdir fern-build
cd fern-build
cmake ../fern -DCMAKE_BUILD_TYPE=Debug -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.5
make
```

Build flags, such as -Wall, can be set by prepending the CXX_FLAGS variable to 
the cmake command as such

```bash
CXX_FLAGS='-Wall' cmake ../fern -DCMAKE_BUILD_TYPE=Debug -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.5
```

Optimization flags should be handled by setting -DCMAKE_BUILD_TYPE=Release 
instead of Debug. Likewise, an optimized build with debug information can be 
acheived by setting -DCMAKE_BUILD_TYPE=RelWithDebugInfo.

## Running FERN

You can run FERN from any directory and it only requires one of its INI files
to run. So, assuming you are in the build directory and using one of the test
files in the data directory, you would run

```bash
./fern-exec ../fern/data/alpha.ini
```

## Updating Submodules

FERN uses the Fire Framework for various utilities. This is included as a git
submodule, which can be updated by running the the following from the FERN
source directory

```bash
bash ./tools/updateSubModules.sh
```

FERN always checks out the latest release version of the submodule from the 
repository, so it should not be necessary to update it. However, the utility
should be used as required.

## Questions
Questions can be directed to me at billingsjj <at> ornl <dot> gov.
