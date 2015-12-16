#set terminal and output
set terminal postscript enhanced color
set output 'plot.ps'
 
 # Set various features of the plot
 set pm3d
 set surface  # don't need surfaces
 #set view map
 set contour
 set key outside
 #set logscale z
 set dgrid3d #plot non-uniform gridded data
 #set cntrparam cubicspline  # smooth out the lines
 set cntrparam levels 15 #incremental 3,10,53    # sets the num of contour lines
 set pm3d #interpolate 5,5 # interpolate the color
 #set cbrange [0:10]
  
  # Set a nice color palette
  set palette model RGB defined ( 0"black", 1"blue", 2"cyan",3"green",4"yellow",5"red",8"purple")
        
        # Axes
        set xlabel 'cutoff_d'
        set ylabel 'nstddelta'
        set format x '%.1f'
        set format y '%.1f'
        set format z '%.2f'
         
         # Now plot
         splot 'contour.txt' using 1:2:3 notitle with lines lt 1
