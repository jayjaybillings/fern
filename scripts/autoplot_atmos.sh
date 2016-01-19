#! /bin/bash

clear

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
GNUautoscaleLine=$(awk '/autoscale/{ print NR; exit }' $GNUfile)

#save some old values in case we need to replace defaults
OldTitle=$(sed $GNUtitleLine"q;d" $GNUfile)
Oldyrange=$(sed $GNUyrangeLine"q;d" $GNUfile)
Oldxrange=$(sed $GNUxrangeLine"q;d" $GNUfile)
Oldformatxy=$(sed $GNUformatxyLine"q;d" $GNUfile)
Cleanformatxy=$(echo $Oldformatxy | cut -f2 -d#)
Oldformatx=$(sed $GNUformatxLine"q;d" $GNUfile)
Cleanformatx=$(echo $Oldformatx | cut -f2 -d#)
Oldformaty=$(sed $GNUformatyLine"q;d" $GNUfile)
Cleanformaty=$(echo $Oldformaty | cut -f2 -d#)

#check arguments

#CHECK TITLE
if [[ "$@" =~ "title" ]] ; 
then 
echo "Editing Title"
echo "Please enter a new title for this plot, and press ENTER:"
cont=0
  while [[ "$cont" == 0 ]]
    do
    read title
    echo "The plot will be titled '$title'. Is that okay? Y/N"
    read check
    if [[ $check == "Y" ]] || [[ $check == "y" ]] ; then
      sed -i '' $GNUtitleLine"s/.*/set title \"$title\" font \"helvetica,18\"/" $GNUfile      
      cont=1
    elif [[ $check == "N" ]] || [[ $check == "n" ]] ; then 
      echo
      echo "Please enter a new title for this plot, and press ENTER:"
    else
      echo "Please type Y or N"
    fi
  done
fi

#CHECK SCALE
if [[ "$@" =~ "scale" ]]
then
echo "Editing Scale"
echo "What scale would you like for this plot? Please enter 'lin' or 'log':"
cont=0
  while [[ "$cont" == 0 ]]
    do
    read scale
    if [[ $scale == "lin" ]] ; then
      sed -i '' $GNUscaleLine"s/.*/unset logscale/" $GNUfile      
      sed -i '' $GNUformatxyLine"s/.*/#$Cleanformatxy/" $GNUfile      
      sed -i '' $GNUformatxLine"s/.*/$Cleanformatx/" $GNUfile      
      sed -i '' $GNUformatyLine"s/.*/$Cleanformaty/" $GNUfile      
      cont=1
    elif [[ $scale == "log" ]] ; then
      sed -i '' $GNUscaleLine"s/.*/set logscale/" $GNUfile      
      sed -i '' $GNUformatxyLine"s/.*/$Cleanformatxy/" $GNUfile      
      sed -i '' $GNUformatxLine"s/.*/#$Cleanformatx/" $GNUfile      
      sed -i '' $GNUformatyLine"s/.*/#$Cleanformaty/" $GNUfile      
      cont=1
    else
      echo "Please type lin or log"
    fi
  done
fi

#CHECK RANGE
if [[ "$@" =~ "range" ]]
  then
  echo "Editing Range"
  # get current range from file
  xrangemin=$(echo $Oldxrange | cut -f2 -d[)
  xrangemin=$(echo $xrangemin | cut -f1 -d:)

  xrangemax=$(echo $Oldxrange | cut -f2 -d:)
  xrangemax=$(echo $xrangemax | cut -f1 -d])

  yrangemin=$(echo $Oldyrange | cut -f2 -d[)
  yrangemin=$(echo $yrangemin | cut -f1 -d:)

  yrangemax=$(echo $Oldyrange | cut -f2 -d:)
  yrangemax=$(echo $yrangemax | cut -f1 -d])

echo $xrangemax
  #get x-range from user
  check="N"
  re='^-?[0-9]+([.][0-9]+)?([*][0-9]+)?([/\^/][0-9]+)?$'
  while [[ "$check" == "N" ]] || [[ "$check" == "n" ]]
    do
    clear
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
    echo "Cool! Now, please enter a MAXIMUM for the x-axis:"

    cont=0

    while [[ "$cont" == 0 ]]
      do
      echo "To keep the default, which is $xrangemax, enter D"
      read entry
      #check if entry is a number
        #first if in exponential form, put into format: *10^# that bash can read.
        numCheck=`echo ${entry} | sed -e 's/[eE]+*/\\*10\\^/'`
      #regex for all real numbers, allows scientific notation
      if ! [[ $numCheck =~ $re ]] && [[ "$numCheck" != "D" ]] && [[ "$numCheck" != "d" ]] ; then
        echo "error: Not a number, please enter a number"
        echo
        echo "Please enter a MAXIMUM for the x-axis (eg: 5e-13, 10.3, etc):"
      elif [[ "$numCheck" == "D" ]] || [[ "$numCheck" == "d" ]] ; then 
        cont=1
      else 
        xrangemax=${entry}
        cont=1
      fi
    done
    
    echo 'x-axis maximum:' $xrangemax

    echo  
    echo "Great! Now let's take care of the y-axis, please enter a MINIMUM for the y-axis:"
    cont=0

    while [[ "$cont" == 0 ]]
      do
      echo "To keep the default, which is $yrangemin, enter D"
      read entry
      #check if entry is a number
        #first if in exponential form, put into format: *10^# that bash can read.
        numCheck=`echo ${entry} | sed -e 's/[eE]+*/\\*10\\^/'`
      #regex for all real numbers, allows scientific notation
      if ! [[ $numCheck =~ $re ]] && [[ "$numCheck" != "D" ]] && [[ "$numCheck" != "d" ]] ; then
        echo "error: Not a number, please enter a number"
        echo
        echo "Please enter a MINIMUM for the y-axis (eg: 5e-13, 10.3, etc):"
      elif [[ "$numCheck" == "D" ]] || [[ "$numCheck" == "d" ]] ; then 
        cont=1
      else 
        yrangemin=${entry}
        cont=1
      fi
    done
    
    echo 'y-axis minimum:' $yrangemin

    echo
    echo "Okay, last is the MAXIMUM for the y-axis:"

    cont=0

    while [[ "$cont" == 0 ]]
      do
      echo "To keep the default, which is $yrangemax, enter D"
      read entry
      #check if entry is a number
        #first if in exponential form, put into format: *10^# that bash can read.
        numCheck=`echo ${entry} | sed -e 's/[eE]+*/\\*10\\^/'`
      #regex for all real numbers, allows scientific notation
      if ! [[ $numCheck =~ $re ]] && [[ "$numCheck" != "D" ]] && [[ "$numCheck" != "d" ]] ; then
        echo "error: Not a number, please enter a number"
        echo
        echo "Please enter a MAXIMUM for the y-axis (eg: 5e-13, 10.3, etc):"
      elif [[ "$numCheck" == "D" ]] || [[ "$numCheck" == "d" ]] ; then 
        cont=1
      else 
        yrangemax=${entry}
        cont=1
      fi
    done
    
    echo 'y-axis maximum:' $yrangemax
    echo
    echo
    echo "You entered [$xrangemin:$xrangemax] as the new x-axis range, and [$yrangemin:$yrangemax] as the new y-axis range."
    echo
    echo "Is that okay? Enter N if you want to change your entry. Otherwise press the 'any' key:"
    read check
  done

  echo "Would you like to make this the new default for plotting range? Y/N:"
  cont=0
  while [[ $cont == 0 ]] 
    do
    read check
    if [[ $check == "Y" ]] || [[ $check == "y" ]] ; then
      #update xrange and yrange default
      FinalxrangeLine="#set xrange [$xrangemin:$xrangemax]"
      FinalyrangeLine="#set yrange [$yrangemin:$yrangemax]"
      echo "Alright, saving this as the new default: xrange [$xrangemin:$xrangemax], yrange [$yrangemin:$yrangemax]"
      cont=1
    elif [[ $check == "N" ]] || [[ $check == "n" ]] ; then
      FinalxrangeLine=$Oldxrange
      FinalyrangeLine=$Oldyrange
      echo "Keeping the old values as default"
      cont=1
    else
      echo "Please enter Y or N to continue"
    fi
  done
  sed -i '' $GNUxrangeLine"s/.*/set xrange [$xrangemin:$xrangemax]/" $GNUfile
  sed -i '' $GNUyrangeLine"s/.*/set yrange [$yrangemin:$yrangemax]/" $GNUfile
  sed -i '' $GNUautoscaleLine"s/.*/#set autoscale/" $GNUfile

  echo "Alrighty then. Plotting! *plot plot plot plot*"
  echo
  echo
fi
#end range conditions


#Find lines in FERN source files we want to manipulate
KernPLOTBOOLLINE=$(awk '/const bool GNUplot/{ print NR; exit }' $KERNELSFILE)
NetPLOTBOOLLINE=$(awk '/const bool plotOutput/{ print NR; exit }' $NETWORKFILE)
KernRADICALBOOLLINE=$(awk '/const bool plotRadicals/{ print NR; exit }' $KERNELSFILE)
NetRADICALBOOLLINE=$(awk '/const bool plotRadicals/{ print NR; exit }' $NETWORKFILE)

#CHECK RADICALS
if [[ $@ =~ "rad" ]] ; 
then 
echo "Plotting Radicals"
sed -i '' $GNUylabelLine"s/.*/set ylabel \"Y(molecules/cm^3)\"/" $GNUfile
sed -i '' $KernRADICALBOOLLINE"s/.*/const bool plotRadicals = 1;/" $KERNELSFILE
sed -i '' $NetRADICALBOOLLINE"s/.*/const bool plotRadicals = 1;/" $NETWORKFILE
fi

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
sed -i '' $NetPLOTBOOLLINE"s/.*/const bool plotOutput = 1;/" $NETWORKFILE

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

#reset other parameters to default
if [[ $1 == "range" ]] || [[ $2 == "range" ]] || [[ $3 == "range" ]] || [[ $4 == "range" ]]; then 
  sed -i '' $GNUxrangeLine"s/.*/$FinalxrangeLine/" $GNUfile
  sed -i '' $GNUyrangeLine"s/.*/$FinalyrangeLine/" $GNUfile
  sed -i '' $GNUautoscaleLine"s/.*/set autoscale/" $GNUfile
fi

#make sure all parameters for Radical plotting is back to default (non-radical)
sed -i '' $GNUylabelLine"s/.*/set ylabel \"Y(ppb)\"/" $GNUfile

