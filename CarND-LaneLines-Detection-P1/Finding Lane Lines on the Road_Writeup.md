1. ***Pipeline Discription***

   My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied a Gaussian Noise kernel to suppress noise and spurious gradients by averaging ,next I applied the Canny transform to get the edges in the image. In order to get the the region of interest, I applied an image mask defined by the polygon formed from vertices. Then I can apply the Hough Transform to find the lines in the region of interest in the image. By applying the draw_lines() function I can draw the lines on a empty image. Finally I merge the image which contains the lines and the initial image with the weighted_img() function. 

   In order to draw a single line on the left and right lanes, I modified the hough_lines() function to hough_extendedlines() function. Firstly I add three new functions, slope(), lines_seperate() and lines_extend(). Slope() is used to calculate the slope of the lines. Lines_seperate() can separate the right and left lines by the slope of the lines. Lines_extend() function can average and extend the lines.In lines_extend() function I get the equation of the lines by calculating the average center coordinates and the average slope of all the lines in one side:*y-centery=slope*(x-centerx)*.Then I put y_max=539 and y_min=330 in the equation to get x_start and x_end. Finally I get the result line that is decided by two points (x_start, y_max)(x_end,y_min) and use the draw_lines() function to draw it. 

   ​

2.  **Potential Shortcomings with my current pipeline**

   When I use my pipline to detect the lines in the image, it seems OK. But when I apply it to the test videos, the line in the left side in several frames doesn't fit the lane lines on the road. Maybe the calculation of the slope of the lines is inaccurate. In addition when I apply my current pipeline to the challenge video, it can't distinguish the lane lines and other lines and doesn't work. That means it's not robust enough.

    

3. **Suggest possible improvements to your pipeline**

   I want to find the reason of inaccurate slope of the left lines.

   ​

   ​