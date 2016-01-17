#! /bin/bash

clear

echo "Hello, I will generate and plot your data now"

echo "Hi, $USER!"
echo

KERNELSFILE='../src/kernels.cpp'
NETWORKFILE='../src/Network.cpp'
KernPLOTBOOLLINE=$(awk '/const bool GNUplot/{ print NR; exit }' $KERNELSFILE)
NetPLOTBOOLLINE=$(awk '/const bool plotOutput/{ print NR; exit }' $NETWORKFILE)
NetRADICALBOOLLINE=$(awk '/const bool plotRadicals/{ print NR; exit }' $NETWORKFILE)

echo "plotOutput variable is set on line $KernPLOTBOOLLINE of $KERNELSFILE. I will set it to TRUE"
echo "plotOutput variable is set on line $NetPLOTBOOLLINE of $NETWORKFILE. I will set it to TRUE"
echo "plotRadicals variable is set on line $NetRADICALBOOLLINE of $NETWORKFILE. I will set it to TRUE"
echo

#show Lines we want to change
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE 
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

#TURN ON PLOTTING
sed -i '' $KernPLOTBOOLLINE"s/.*/const bool GNUplot = 1;/" $KERNELSFILE
sed -i '' $NetPLOTBOOLLINE"s/.*/const bool plotOutput = 1;/" $NETWORKFILE
sed -i '' $NetRADICALBOOLLINE"s/.*/const bool plotRadicals = 1;/" $NETWORKFILE

#show lines we changed
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

#get ready to run FERN
cd ../ 
#run FERN and save output to file
make run > output/fernOutput.dat 
cd scripts
# TURN OFF PLOTTING BEFORE EXIT
sed -i '' $KernPLOTBOOLLINE"s/.*/const bool GNUplot = 0;/" $KERNELSFILE
sed -i '' $NetPLOTBOOLLINE"s/.*/const bool plotOutput = 0;/" $NETWORKFILE
sed -i '' $NetRADICALBOOLLINE"s/.*/const bool plotRadicals = 0;/" $NETWORKFILE
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

