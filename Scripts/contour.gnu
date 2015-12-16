#set terminal and output
set terminal postscript enhanced color
set output '../plots/fidepe/Compound_contour.ps'
 
# Set various features of the plot
#set key outside
#set pm3d
set contour
unset surface  # don't need surfaces
set view map
set key left top outside
#set hidden3d
#set logscale z
set dgrid3d #plot non-uniform gridded data
#set cntrparam cubicspline  # smooth out the lines
set cntrparam levels incremental 0,1,10    # sets the num of contour lines
set cntrlabel font ",20" start 2 interval 10
#set pm3d interpolate 5,5 # interpolate the color
#set cbrange [0:10]
#set style data lines
# Set a nice color palette
#set palette model RGB defined ( 0"black", 1"blue", 2"cyan",3"green",4"yellow",5"red",8"purple")
      # Axes
set xlabel 'cutoff\_d'
set ylabel 'nstddelta'
#set xlabel 'eps'
#set ylabel 'MinPts'
#set zlabel 'cluster number'
set format x '%.1f'
set format y '%.1f'
set format z '%.0f'
 
# Now plot
splot '../plots/fidepe/Compound.dat' using 1:2:3 title 'Compound' with lines lt 1,'../plots/fidepe/Compound.dat' using 1:2:3 notitle with labels
