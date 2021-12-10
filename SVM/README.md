# SVM Algorithms

In the `SVM.py` file you can find the implementation of the SVM algorithm (and several other algorithms, in seperated classes). <br/>
The implemntation is for the Multi Class problem of SVM algorithm.

##  Dataset
The dataset is the Iris flower classification. In this dataset, we have five features per instance (all the features are numerical) and three labels `0`,`1`,`2`, which are correspond to the Iris flower species.

##  Run
The code should get as input four arguments.<br/>

The run command to the program should be:<br/>
> $ python SVM.py <train_x_path> <train_y_path> <test_x_path> <output_file_name> <br/>

### Parameters

Name | Meaning 
-----|-------
`train_x_path` | The training examples
`train_y_path` | The training examples labels
`test_x_path` | The testing examples
`output_file_name` | The output file name (will hold the results of each algorithm)

Each line in the output filem, will be in the format: <br/>
knn: <`calculated_label`>, perceptron: <`calculated_label`>, svm: <`calculated_label`>, pa: <`calculated_label`>, <br/>
such that <`calculated_label`> = {0,1,2}
