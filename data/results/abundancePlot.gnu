set macro  # Enable macro definition

# Some macro definitions

label_color = "#867961"
tic_color = "#383838"
title_color = "#383838"
myblue_color = "#5ea2c6
myred_color = "#bb6255"
mygreen_color = "#668874"

# Macros defining line widths and pointsizes.  Reference with "@Macro"
LW1 = "1"
LW2 = "2"

# Set the default tic colors
set xtics textcolor rgb tic_color
set ytics textcolor rgb tic_color

# Set the default point size if points are plotted
set pointsize 0.5    # Size of plotted points

# Line styles

set style line 1 lt rgb myblue_color lw @LW1 pt 1      # Plus
set style line 2 lt rgb myred_color lw @LW1 pt 2       # Cross
set style line 3 lt rgb mygreen_color lw @LW1 pt 3     # Asterisk
set style line 4 lt rgb "black" lw @LW1 pt 4           # Square dot
set style line 5 lt rgb "purple" lw @LW1 pt 5          # Filled square
set style line 6 lt rgb "red" lw @LW1  pt 6            # Circle dot
set style line 7 lt rgb "royalblue" lw @LW1 pt 7       # Filled circle
set style line 8 lt rgb "blue" lw @LW1 pt 8            # Define linestyle 8
set style line 9 lt rgb "green" lw @LW1 pt 9           # Filled triangle
set style line 10 lt rgb "orchid" lw @LW1 pt 10        # Triangle down dot
set style line 11 lt rgb "gold" lw @LW1 pt 11          # Filled triangle down
set style line 12 lt rgb "navy" lw @LW1 pt 12          # Diamond dot
set style line 13 lt rgb "light-red" lw @LW1 pt 13     # Filled diamond
set style line 14 lt rgb "magenta" lw @LW1 pt 14       # Pentagon dot
set style line 15 lt rgb "orange-red" lw @LW1  pt 15   # Filled pentagon
set style line 16 lt rgb "olive" lw @LW1 pt 64         # Open square
set style line 17 lt rgb "violet" lw @LW1 pt 65        # Open circle
set style line 18 lt rgb "gray40" lw @LW1 pt 66        # Open triangle
set style line 19 lt rgb "yellow4" lw @LW1 pt 67       # Open triangle down
set style line 20 lt rgb "dark-orange" lw @LW1 pt 68   # Open diamond

set style line 51 lt rgb myblue_color lw @LW1 pt 1      # Plus
set style line 52 lt rgb myred_color lw @LW1 pt 2       # Cross
set style line 53 lt rgb mygreen_color lw @LW1 pt 3     # Asterisk
set style line 54 lt rgb "black" lw @LW1 pt 4           # Square dot
set style line 55 lt rgb "purple" lw @LW1 pt 5          # Filled square
set style line 56 lt rgb "red" lw @LW1  pt 6            # Circle dot
set style line 57 lt rgb "royalblue" lw @LW1 pt 7       # Filled circle
set style line 58 lt rgb "blue" lw @LW1 pt 8            # Define linestyle 8
set style line 59 lt rgb "green" lw @LW1 pt 9           # Filled triangle
set style line 60 lt rgb "orchid" lw @LW1 pt 10        # Triangle down dot
set style line 61 lt rgb "gold" lw @LW1 pt 11          # Filled triangle down
set style line 62 lt rgb "navy" lw @LW1 pt 12          # Diamond dot
set style line 63 lt rgb "light-red" lw @LW1 pt 13     # Filled diamond
set style line 64 lt rgb "magenta" lw @LW1 pt 14       # Pentagon dot
set style line 65 lt rgb "orange-red" lw @LW1  pt 15   # Filled pentagon
set style line 66 lt rgb "olive" lw @LW1 pt 64         # Open square
set style line 67 lt rgb "violet" lw @LW1 pt 65        # Open circle
set style line 68 lt rgb "gray40" lw @LW1 pt 66        # Open triangle
set style line 69 lt rgb "yellow4" lw @LW1 pt 67       # Open triangle down
set style line 70 lt rgb "dark-orange" lw @LW1 pt 68   # Open diamond

#set xtics rotate        # Rotates x tic numbers by 90 degrees
#set ytics rotate        # Rotates y tic numbers by 90 degrees
# Set tic labeling with color
set xtics textcolor rgb tic_color
set ytics textcolor rgb tic_color
set bmargin 4  # Bottom margin

# Width and height of postscript figure in inches
width = 8
height = 5
# Set screen display to same aspect ratio as postscript plot
set size ratio height/width

set title 'Double (lines) vs single (symbols) precision; 150 network' textcolor rgb title_color
set xlabel 'log time (s)' textcolor rgb tic_color
set ylabel 'log Mass fraction X' textcolor rgb tic_color
# Uncomment following to set log or log-log plots
#set logscale x
#set logscale y

# Plot limits
set xrange [-19:-9]
set yrange[-20:0]

set pointsize 0.75    # Size of the plotted points

# Legend Controls
set key outside      # Legend outside plot
set key top right    # Move legend to upper left
#unset key           # Don't show legend
# Control format of legend
set key samplen 1 spacing .9 font ",12"

#set timestamp       # Date/time

# Read data from file and plot to screen

# Lines
plot "fern_150.dat" using 1:5 ls 1 with lines title 'n' 
replot "fern_150.dat" using 1:6 ls 2 with lines title 'p'
replot "fern_150.dat" using 1:7 ls 3 with lines title 'o17'
replot "fern_150.dat" using 1:8 ls 4 with lines title 'o18'
replot "fern_150.dat" using 1:9 ls 5 with lines title 'f17'
replot "fern_150.dat" using 1:10 ls 6 with lines title 'f18'
replot "fern_150.dat" using 1:11 ls 7 with lines title 'he4'
replot "fern_150.dat" using 1:12 ls 8 with lines title 'c12'
replot "fern_150.dat" using 1:13 ls 9 with lines title 'o16'
replot "fern_150.dat" using 1:14 ls 10 with lines title 'ne20'
replot "fern_150.dat" using 1:15 ls 11 with lines title 'mg24'
replot "fern_150.dat" using 1:16 ls 12 with lines title 'si28'
replot "fern_150.dat" using 1:17 ls 13 with lines title 's32'
replot "fern_150.dat" using 1:18 ls 14 with lines title 'ar36'
replot "fern_150.dat" using 1:19 ls 15 with lines title 'ca40'
replot "fern_150.dat" using 1:20 ls 16 with lines title 'ti44'
replot "fern_150.dat" using 1:21 ls 17 with lines title 'cr48'
replot "fern_150.dat" using 1:22 ls 18 with lines title 'fe52'
replot "fern_150.dat" using 1:23 ls 19 with lines title 'ni56'
replot "fern_150.dat" using 1:24 ls 20 with lines title 'fe54'

# Points
replot "fern_150_single.dat" using 1:5 ls 1 with points title 'n'
replot "fern_150_single.dat" using 1:6 ls 2 with points title 'p'
replot "fern_150_single.dat" using 1:7 ls 3 with points title 'o17'
replot "fern_150_single.dat" using 1:8 ls 4 with points title '018'
replot "fern_150_single.dat" using 1:9 ls 5 with points title 'f17'
replot "fern_150_single.dat" using 1:10 ls 6 with points title 'f18'
replot "fern_150_single.dat" using 1:11 ls 7 with points title 'he4'
replot "fern_150_single.dat" using 1:12 ls 8 with points title 'c12'
replot "fern_150_single.dat" using 1:13 ls 9 with points title 'o16'
replot "fern_150_single.dat" using 1:14 ls 10 with points title 'ne20'
replot "fern_150_single.dat" using 1:15 ls 11 with points title 'mg24'
replot "fern_150_single.dat" using 1:16 ls 12 with points title 'si28'
replot "fern_150_single.dat" using 1:17 ls 13 with points title 's32'
replot "fern_150_single.dat" using 1:18 ls 14 with points title 'ar36'
replot "fern_150_single.dat" using 1:19 ls 15 with points title 'ca40'
replot "fern_150_single.dat" using 1:20 ls 16 with points title 'ti44'
replot "fern_150_single.dat" using 1:21 ls 17 with points title 'cr48'
replot "fern_150_single.dat" using 1:22 ls 18 with points title 'fe52'
replot "fern_150_single.dat" using 1:23 ls 19 with points title 'ni56'
replot "fern_150_single.dat" using 1:24 ls 20 with points title 'fe54'
 


#replot "async_150_full_perTimestep.dat" using 1:3 ls 3 with points title 'Integration time, one step


# Plot to postscript file

set out "abundanceVsTime.eps"    # Output file
set terminal postscript eps size width, height enhanced color solid lw 2 "Arial" 32
replot               # Plot to postscript file

# Plot to PNG file

set out "abundanceVsTime.png"
# Assume 72 pixels/inch and make bitmap twice as large for display resolution
set terminal pngcairo transparent size 2*width*72, 2*height*72 lw 2 enhanced font 'Arial,28'
replot

quit