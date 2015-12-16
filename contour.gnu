#set terminal and output
set terminal postscript enhanced color
set output 'plot.ps'
 
 # Set various features of the plot
 set pm3d
 unset surface  # don't need surfaces
 set view map
 set contour
 set key outside
 #set cntrparam cubicspline  # smooth out the lines
 set cntrparam levels 20    # sets the num of contour lines
 set pm3d interpolate 20,20 # interpolate the color
  
  # Set a nice color palette
  set palette model RGB defined ( 0"black", 1"blue", 2"cyan",3"green",4"yellow",5"red",8"purple")
        
        # Axes
        set xlabel 'contour\_d'
        set ylabel 'nstddelta'
        set format x '%.1f'
        set format y '%.1f'
        set format z '%.2f'
         
         # Now plot
         splot 'contour.txt' using 1:2:3 notitle with lines lt 1
