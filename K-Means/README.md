# K-Means Algorithms

In the `K-Means.py` file you can find the implementation of the K-Means algorithm. <br/>
I implemented the k-means algorithm for image compression - on the image pixels and then replace each pixel by its centroid. <br/>

## Notes
- Originally, the initial centroids in K-means are randomly generated. For reproducible purposes we will use a centroids initialization values.
- In case when 2 centroids are evenly close to a certain point, the one with the lower index "wins".
- The code will run for 20 iterations or until convergence. We define convergence where all the centroids don't change.

##  Run
The code should get as input four arguments.<br/>

The run command to the program should be:<br/>
> $ python K-Means.py <image_path> <centroinds_init_path> <output_file_name> <br/>

### Parameters

Name | Meaning 
-----|-------
`image_path` | The path to the image
`centroinds_init_path` | The centroids initialization values
`output_file_name` | The output file name (will hold the results of each algorithm)

Each line in the output filem, will be in the format (for eaxmple): <br/>
[iter 0]: [0.1327 0.1135 0.1088] , [0.6819 0.6071 0.5152] <br/>
[iter 1]: [0.1022 0.0879 0.0899] , [0.6549 0.5801 0.4896] <br/>
... <br/>
[iter 7]: [0.0918 0.0793 0.0837] , [0.6435 0.569 0.4796] <br/>
[iter 8]: [0.0918 0.0793 0.0837] , [0.6435 0.569 0.4796] 
