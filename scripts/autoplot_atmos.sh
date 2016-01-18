#! /bin/bash

clear

echo "Hello, I will generate and plot your data now"

echo "Hi, $USER!"
echo


GNUfile='../atmos.gnu'
KERNELSFILE='../src/kernels.cpp'
NETWORKFILE='../src/Network.cpp'
OUTPUTFILE='../output/fernOutput.dat'

#Format GNUplot file if arguments have been specified
GNUtitleLine=$(awk '/set title/{ print NR; exit }' $GNUfile)
printf "%s\n" "$GNUtitleLine"
GNUscaleLine=$(awk '/logscale/{ print NR; exit }' $GNUfile)
GNUformatxyLine=$(awk '/format xy/{ print NR; exit }' $GNUfile)
GNUformatyLine=$(awk '/format y "/{ print NR; exit }' $GNUfile)
GNUformatxLine=$(awk '/format x "/{ print NR; exit }' $GNUfile)
GNUylabelLine=$(awk '/ylabel/{ print NR; exit }' $GNUfile)
GNUyrangeLine=$(awk '/yrange/{ print NR; exit }' $GNUfile)
GNUxrangeLine=$(awk '/xrange/{ print NR; exit }' $GNUfile)

#save some old values in case we need to replace defaults
OldTitle=$(sed $GNUtitleLine"q;d" $GNUfile)
Oldyrange=$(sed $GNUyrangeLine"q;d" $GNUfile)
Oldxrange=$(sed $GNUxrangeLine"q;d" $GNUfile)

#setup arg variables
title=0 #option to change title
scale=0 #option to change scale
#check arguments
if [[ "$@" == "title" ]]
then 
echo "Please enter a new title for this plot, and press ENTER:"
read title
echo "title $title"
fi
if [[ "$@" == "scale" ]]
then
echo "What scale would you like for this plot? Please enter 'linear' or 'log':"
read scale
echo "scale $scale"
fi

#check if user requests to change range of the axes
if [[ "$@" == "range" ]]
  then
  # get current range from file
  #var1=$(echo $STR | cut -f1 -d-)
  xrangemin=$(echo $Oldxrange | cut -f2 -d[)
  xrangemin=$(echo $xrangemin | cut -f1 -d:)
echo $xrangemin
  xrangemax=$(echo $Oldxrange | cut -f2 -d:)
  xrangemax=$(echo $xrangemax | cut -f1 -d])
echo $xrangemax
  #get x-range from user
  check="N"
  re='^-?[0-9]+([.][0-9]+)?([*][0-9]+)?([/\^/][0-9]+)?$'
  while [[ "$check" == "N" ]]
    do
    cont=0 #boolean for continuing the script
    while [[ "$cont" == 0 ]]
    do
      echo "Please enter a MINIMUM for the x-axis (eg: 5e-13, 10.3, etc):"
      echo "To keep the default, which is $xrangemin, enter D"
      read entry
      #check if entry is a number
        #first if in exponential form, put into format: *10^# that bash can read.
        numCheck=`echo ${entry} | sed -e 's/[eE]+*/\\*10\\^/'`
      #regex for all real numbers, allows scientific notation
      if ! [[ $numCheck =~ $re ]] && [[ "$numCheck" != "D" ]] && [[ "$numCheck" != "d" ]] ; then
        echo "error: Not a number, please enter a number"
        echo
      elif [[ "$numCheck" == "D" ]] || [[ "$numCheck" == "d" ]] ; then 
        cont=1
      else 
        xrangemin=${entry}
        cont=1
      fi
    done
    
echo 'x-axis minimum:' $xrangemin

    echo  
    echo "Sweet, thanks! Now, please enter a MAXIMUM for the x-axis:"

    cont=0
    while [[ "$cont" == 0 ]]
    do
      echo "To keep the default, which is $xrangemax, enter D"
      read entry
      #check if entry is a number
        #first if in exponential form, put into format: *10^# that bash can read.
        numCheck=`echo ${entry} | sed -e 's/[eE]+*/\\*10\\^/'`
      if ! [[ $numCheck =~ $re ]] && [[ "$numCheck" != "D" ]] && [[ "$numCheck" != "d" ]] ; then
        echo "error: Not a number, please enter a number"
        echo
        echo "Please enter a MAXIMUM for the x-axis (eg: 5e-13, 10.3, etc):"
      else cont=1
      fi
    done
    read xrangemax

    if [[ "$xrangemin" > 0 || "$xrangemax" > 0 ]]
    then
      echo "You entered [$xrangemin:$xrangemax] as the new x-axis range. Is that okay?"
      echo "Enter N if you want to change your entry. Otherwise press the 'any' key:"
      read check
    fi
  done
  #check if should be new default x-range
  echo "Would you like to set this as the new default x-range?"
  read check
  #get y-range from user
  check="N"

  #while [[ "$check" == "N" ]]
  #do
  #  echo "Please enter a custom range for the y-axis (format: [#:#]):"
  #  read yrange
  #  echo "You entered $yrange as the new y-axis range"
    
  #done

fi


#GNUscaleTypeLine=$(awk '/const bool GNUplot/{ print NR; exit }' $KERNELSFILE)
#GNUscaleNumFormatLine1=$(awk '/const bool GNUplot/{ print NR; exit }' $KERNELSFILE)
#sed -i '' $KernPLOTBOOLLINE"s/.*/const bool GNUplot = 1;/" $KERNELSFILE

#Find lines in FERN source files we want to manipulate
KernPLOTBOOLLINE=$(awk '/const bool GNUplot/{ print NR; exit }' $KERNELSFILE)
KernRADICALBOOLLINE=$(awk '/const bool plotRadicals/{ print NR; exit }' $KERNELSFILE)
NetPLOTBOOLLINE=$(awk '/const bool plotOutput/{ print NR; exit }' $NETWORKFILE)
NetRADICALBOOLLINE=$(awk '/const bool plotRadicals/{ print NR; exit }' $NETWORKFILE)

echo "GNUplot variable is set on line $KernPLOTBOOLLINE of $KERNELSFILE. I will set it to TRUE"
echo "plotRadicals variable is set on line $KernRADICALBOOLLINE of $KERNELSFILE. I will set it to TRUE"
echo "plotOutput variable is set on line $NetPLOTBOOLLINE of $NETWORKFILE. I will set it to TRUE"
echo "plotRadicals variable is set on line $NetRADICALBOOLLINE of $NETWORKFILE. I will set it to TRUE"
echo

#show Lines we want to change
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE 
sed $KernRADICALBOOLLINE"q;d" $KERNELSFILE 
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

#TURN ON PLOTTING
sed -i '' $KernPLOTBOOLLINE"s/.*/const bool GNUplot = 1;/" $KERNELSFILE
if [ $@ == "rad" ]; then #plot radical species
  sed -i '' $KernRADICALBOOLLINE"s/.*/const bool plotRadicals = 1;/" $KERNELSFILE
fi
sed -i '' $NetPLOTBOOLLINE"s/.*/const bool plotOutput = 1;/" $NETWORKFILE
if [ $@ == "rad" ]; then
sed -i '' $NetRADICALBOOLLINE"s/.*/const bool plotRadicals = 1;/" $NETWORKFILE
fi

#show lines we changed
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE
sed $KernRADICALBOOLLINE"q;d" $KERNELSFILE
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

#get ready to run FERN
cd ../ 
#run FERN and save output to file
make run > output/fernOutput.dat 
cd scripts
#find lines we want to manipulate in fernOutput.dat
colBeginning=$(awk '/BEGIN PLOTTING \(COLHEADER\)/{ print NR; exit }' $OUTPUTFILE)
colEnd=$(awk '/END PLOTTING \(COLHEADER\)/{ print NR; exit }' $OUTPUTFILE)
dataBeginning=$(awk '/STARTOUTPUT/{ print NR; exit }' $OUTPUTFILE)
dataEnd=$(awk '/ENDOUTPUT/{ print NR; exit }' $OUTPUTFILE)

echo "Beginning of column headers is on line $colBeginning of $OUTPUTFILE."
echo "End of column headers is on line $colEnd of $OUTPUTFILE."
echo "Beginning of data is on line $dataBeginning of $OUTPUTFILE."
echo "End of data is on line $dataEnd of $OUTPUTFILE."

#delete unnecessary lines
sed -i".bak" '1,'$colBeginning'd;'$colEnd','$dataBeginning'd;'$dataEnd',$d' $OUTPUTFILE


cd ../
gnuplot load 'atmos.gnu'
cd scripts
# TURN OFF PLOTTING BEFORE EXIT
sed -i '' $KernPLOTBOOLLINE"s/.*/const bool GNUplot = 0;/" $KERNELSFILE
sed -i '' $KernRADICALBOOLLINE"s/.*/const bool plotRadicals = 0;/" $KERNELSFILE
sed -i '' $NetPLOTBOOLLINE"s/.*/const bool plotOutput = 0;/" $NETWORKFILE
sed -i '' $NetRADICALBOOLLINE"s/.*/const bool plotRadicals = 0;/" $NETWORKFILE
sed $KernPLOTBOOLLINE"q;d" $KERNELSFILE
sed $KernRADICALBOOLLINE"q;d" $KERNELSFILE
sed $NetPLOTBOOLLINE"q;d" $NETWORKFILE 
sed $NetRADICALBOOLLINE"q;d" $NETWORKFILE 

