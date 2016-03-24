# Scalable-Locally-Injective-Mappings
This supplemental archive contains both source code and binaries for the implementation
used in the ETH Technical Report:

“Scalable Locally Injective Mappings"
Michael Rabinovich, Roi Poranne, Daniele Panozzo and Olga Sorkine-Hornung

It currently only supports mesh parametrization minimizing the symmetric Dirichlet isometric energy.

The content is as follows

binaries/     -- Binaries for Windows, OSX, and Linux (64 bit)
ext/          -- External dependencies (libigl, Eigen, Thread Building Blocks)
src/          -- Source code
camelhead.obj -- Example mesh file

The implementation needs to solve a sparse linear system, and either Eigen or
PARDISO can be used for this purpose, where the latter is significantly faster.

Due to licensing restrictions, we unfortunately cannot include PARDISO in this
archive, and thus Eigen is used by default. If you wish to use the PARDISO solver
instead, please save the .dylib/.so/.dll file of the latest release (5.0.0)
in the directory 'ext/pardiso' and recompile using CMake.

To parameterize a mesh, invoke the binary as follows (e.g. on OSX)

$ ./binaries/macos-x86_64/ReweightedARAP camelhead.obj camelhead_parameterized.obj

Acknowledgements:
We thank Wenzel Jakob for porting this code to Linux and Windows and for
improving the cmake build system.
This work was supported in part by the ERC Starting Grant iModel (StG-2012-306877) and a gift from Adobe.
