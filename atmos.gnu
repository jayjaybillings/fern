# Some macro definitions

label_color = "#867961"
tic_color = "#383838"
title_color = "#383838"
myblue_color = "#5ea2c6
myred_color = "#bb6255"
mygreen_color = "#668874"

set style line 1 lw 3 lt rgb myblue_color
set style line 2 lw 3 lt rgb myred_color
set style line 3 lw 3 lt rgb mygreen_color
set style line 4 lw 3 lt rgb "black"
set style line 5 lw 3 lt rgb "purple"
set style line 6 lw 3 lt rgb "dark-green" 
set style line 7 lw 3 lt rgb "#8B008B"
set style line 8 lw 3 lt rgb "magenta" 
set style line 9 lw 3 lt rgb "orange" 
set style line 10 lw 3 lt rgb "blue" 
set style line 11 lw 3 lt rgb "green"
set style line 12 lw 3 lt rgb "dark-orange"
set style line 13 lw 3 lt rgb "olive"
set style line 14 lw 3 lt rgb "gray40"
set style line 15 lw 3 lt rgb "salmon"
set style line 16 lw 3 lt rgb "seagreen"
set style line 17 lw 3 lt rgb "#A0522D"
set style line 18 lw 3 lt rgb "turquoise"
set style line 19 lw 3 lt rgb "#6A5ACD"
set style line 20 lw 3 lt rgb "#C0C0C0"
set style line 21 lw 3 lt rgb "olivedrab"
set style line 22 lw 3 lt rgb "#9370DB"
set style line 23 lw 3 lt rgb "#87CEFA"
set style line 24 lw 3 lt rgb "#B22222"

set terminal pngcairo background "#ffffff" size 1700, 1010
#datafile = 'polluted_dt1e-4.dat'
#datafile = 'test.txt'
datafile = 'testzenith0.txt'
#datafile = 'polluted_dt1e-5.dat'
#datafile = 'polluted_dt1e-3.dat'
#datafile = 'rural_dt1e-4.dat'
#datafile = 'rural_dt1e-5.dat'
#datafile = 'rural_dt1e-3.dat'
#datafile = 'remotropo_dt1e-4.dat'
#datafile = 'remotropo_dt1e-5.dat'
#datafile = 'remotropo_dt1e-3.dat'
set key autotitle columnheader
set key outside
#set title "CHASER Moderately Polluted, zenith=π/2, fluxFrac=1e8" font "helvetica,18"
set title "CHASER Moderately Polluted, zenith=0, fluxFrac=1e8" font "helvetica,18"
#set title "CHASER Moderately Polluted, t_m_a_x=300s, dt=1e-4" font "helvetica,18"
#set title "CHASER Moderately Polluted, t_m_a_x=300s, dt=1e-3" font "helvetica,18"
#set title "CHASER Moderately Polluted, t_m_a_x=300s, dt=1e-5" font "helvetica,18"
#set title "CHASER Rural Continental, t_m_a_x=300s, dt=1e-4" font "helvetica,18"
#set title "CHASER Rural Continental, t_m_a_x=300s, dt=1e-3" font "helvetica,18"
#set title "CHASER Rural Continental, t_m_a_x=300s, dt=1e-5" font "helvetica,18"
#set title "CHASER Remote Troposphere, t_m_a_x=300s, dt=1e-4" font "helvetica,18"
#set title "CHASER Remote Troposphere, t_m_a_x=300s, dt=1e-3" font "helvetica,18"
#set title "CHASER Remote Troposphere, t_m_a_x=300s, dt=1e-5" font "helvetica,18"
set xlabel "log(t)"
set ylabel "Y(ppb)"
set yrange [5e-13:1e3]
set xrange [5e-2:5e2]
set logscale
set format xy "10^{%L}"
set output "~/Desktop/Research/gnuplotScripts/plot.png"
plot for [i=2:25] datafile every ::30::99 u 1:i w lines ls i-1, \
datafile every ::99::99 u 1:2:("O3") with labels font "arial,10" offset 2,-.2 notitle, \
datafile every ::99::99 u 1:3:("O2") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:4:("O1D") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:5:("NO2") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:6:("NO3") with labels font "arial,10" offset 2,.2 notitle, \
datafile every ::99::99 u 1:7:("CO") with labels font "arial,10" offset 2,.2 notitle, \
datafile every ::99::99 u 1:8:("C2H6") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:9:("C3H8") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:10:("C2H4") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:11:("C3H6") with labels font "arial,10" offset 5,.2 notitle, \
datafile every ::99::99 u 1:12:("C5H8") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:13:("C10H16") with labels font "arial,10" offset 3,-.2 notitle, \
datafile every ::99::99 u 1:14:("CH3CHO") with labels font "arial,10" offset 6,-.1 notitle, \
datafile every ::99::99 u 1:15:("MACR") with labels font "arial,10" offset 6,.1 notitle, \
datafile every ::99::99 u 1:16:("OH") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:17:("HO2") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:18:("CH3O2") with labels font "arial,10" offset 6 notitle, \
datafile every ::99::99 u 1:19:("CH3COO2") with labels font "arial,10" offset 2 notitle, \
datafile every ::99::99 u 1:20:("MACRO2") with labels font "arial,10" offset 4 notitle, \
datafile every ::99::99 u 1:21:("H2O") with labels font "arial,10" offset 2,-.3 notitle, \
datafile every ::99::99 u 1:22:("CO2") with labels font "arial,10" offset 2,.2 notitle, \
datafile every ::99::99 u 1:23:("H2") with labels font "arial,10" offset 2,.2 notitle, \
datafile every ::99::99 u 1:24:("CH3COOH") with labels font "arial,10" offset 7 notitle, \
datafile every ::99::99 u 1:25:("HCOOH") with labels font "arial,10" offset 5,.2 notitle
#Polluted (dt=1e-4) label placement ^^^

#datafile every ::99::99 u 1:2:("O3") with labels font "arial,10" offset 2,-.2 notitle, \
#datafile every ::99::99 u 1:3:("O2") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:4:("O1D") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:5:("NO2") with labels font "arial,10" offset 2,.1 notitle, \
#datafile every ::99::99 u 1:6:("NO3") with labels font "arial,10" offset 2,.2 notitle, \
#datafile every ::99::99 u 1:7:("CO") with labels font "arial,10" offset 2,.2 notitle, \
#datafile every ::99::99 u 1:8:("C2H6") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:9:("C3H8") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:10:("C2H4") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:11:("C3H6") with labels font "arial,10" offset 6,-.6 notitle, \
#datafile every ::99::99 u 1:12:("C5H8") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:13:("C10H16") with labels font "arial,10" offset 6 notitle, \
#datafile every ::99::99 u 1:14:("CH3CHO") with labels font "arial,10" offset 7 notitle, \
#datafile every ::99::99 u 1:15:("MACR") with labels font "arial,10" offset 2,-.2 notitle, \
#datafile every ::99::99 u 1:16:("OH") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:17:("HO2") with labels font "arial,10" offset 2,-.7 notitle, \
#datafile every ::99::99 u 1:18:("CH3O2") with labels font "arial,10" offset 3,.3 notitle, \
#datafile every ::99::99 u 1:19:("CH3COO2") with labels font "arial,10" offset 4 notitle, \
#datafile every ::99::99 u 1:20:("MACRO2") with labels font "arial,10" offset 3,-.2 notitle, \
#datafile every ::99::99 u 1:21:("H2O") with labels font "arial,10" offset 2,.1 notitle, \
#datafile every ::99::99 u 1:22:("CO2") with labels font "arial,10" offset 2,.1 notitle, \
#datafile every ::99::99 u 1:23:("H2") with labels font "arial,10" offset 2,-.4 notitle, \
#datafile every ::99::99 u 1:24:("CH3COOH") with labels font "arial,10" offset 7 notitle, \
#datafile every ::99::99 u 1:25:("HCOOH") with labels font "arial,10" offset 6 notitle
#Rural (dt=1e-4) label placement^^^

#datafile every ::99::99 u 1:2:("O3") with labels font "arial,10" offset 2,-.2 notitle, \
#datafile every ::99::99 u 1:3:("O2") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:4:("O1D") with labels font "arial,10" offset 2,-.1 notitle, \
#datafile every ::99::99 u 1:5:("NO2") with labels font "arial,10" offset 2,.1 notitle, \
#datafile every ::99::99 u 1:6:("NO3") with labels font "arial,10" offset 2,.2 notitle, \
#datafile every ::99::99 u 1:7:("CO") with labels font "arial,10" offset 2,.2 notitle, \
#datafile every ::99::99 u 1:8:("C2H6") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:9:("C3H8") with labels font "arial,10" offset 6 notitle, \
#datafile every ::99::99 u 1:10:("C2H4") with labels font "arial,10" offset 2,-.5 notitle, \
#datafile every ::99::99 u 1:11:("C3H6") with labels font "arial,10" offset 6,-.7 notitle, \
#datafile every ::99::99 u 1:12:("C5H8") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:13:("C10H16") with labels font "arial,10" offset 6 notitle, \
#datafile every ::99::99 u 1:14:("CH3CHO") with labels font "arial,10" offset 7 notitle, \
#datafile every ::99::99 u 1:15:("MACR") with labels font "arial,10" offset 2,-.2 notitle, \
#datafile every ::99::99 u 1:16:("OH") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:17:("HO2") with labels font "arial,10" offset 2 notitle, \
#datafile every ::99::99 u 1:18:("CH3O2") with labels font "arial,10" offset 2.3,.3 notitle, \
#datafile every ::99::99 u 1:19:("CH3COO2") with labels font "arial,10" offset 4 notitle, \
#datafile every ::99::99 u 1:20:("MACRO2") with labels font "arial,10" offset 3,-.2 notitle, \
#datafile every ::99::99 u 1:21:("H2O") with labels font "arial,10" offset 2,-.2 notitle, \
#datafile every ::99::99 u 1:22:("CO2") with labels font "arial,10" offset 2,.4 notitle, \
#datafile every ::99::99 u 1:23:("H2") with labels font "arial,10" offset 2,.2 notitle, \
#datafile every ::99::99 u 1:24:("CH3COOH") with labels font "arial,10" offset 7,-.3 notitle, \
#datafile every ::99::99 u 1:25:("HCOOH") with labels font "arial,10" offset 7,.5 notitle
#Remote Troposphere (dt=1e-4)  label placement ^^^
