https://docs.google.com/document/d/1deTogAn5uWrPHf3_1-ruQ5EJh24s5Lh3s3Xfk6MsrWM/edit?usp=sharing

Documentation on running and manipulating MATLAB version of FERN
Author: Daniel Shyles

The latest version of FERN dates from late September, 2016. It was built to run on a Windows machine, but can be easily converted to run on a Unix system, by simply changing the file directory in some file pointers. I have included in the Git repository my latest Windows version as well as a Unix-ready version. 

To run FERN in Matlab, you must ensure that the required data files (rateLibrary_3.data, CUDAnet_3.inp, Y_3.out) are available and in the proper relative directories. If the Matlab file is kept in a consistent location relative to these files (outside the “fern” folder as currently arranged in the repository), all should be well. You should be able to simply copy the Matlab file and “fern” folder into Matlab’s working directory, and it should run.

/////****** Key Parameters ******/////
pEquilOn, and AsyOn: These are boolean inputs that will enable or disable the partial equilibrium and asymptotic approximations, respectively. The calculation will run with all permutations of states.

plotJavaCompare: For the benefit of comparing the output with the Java standard, I’ve included three output files from the Java, (Y_3.out, Y_16.out, and Y_150.out), which you can switch out depending on which network you’re running. When this is switched on, a plot of the Java will popup with our results superimposed. 



/////******   Outstanding issues   ******/////

Needless to say, FERN is not working as we would like. Currently, the accuracy of the abundances is poor as baryon number is not currently being conserved while the timestep is being pushed. When baryon number is conserved, there will certainly be a sacrifice in timestepping.


It seems clear that we must investigate new timesteppers, and consider, perhaps an implicit solver that incorporates these algorithms in tandem with explicit, using iterative timestepper switching depending on which isotopes and RGs satisfy the algorithms.


...more to come!
