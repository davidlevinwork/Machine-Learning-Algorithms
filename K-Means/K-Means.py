import sys
import numpy as np
import matplotlib.pyplot as plt


# <editor-fold desc="Variables">

_imagePath = sys.argv[1]
_centroidsInitPath = sys.argv[2]
_outputLogFileName = sys.argv[3]

_z = np.loadtxt(_centroidsInitPath)
_originalPixels = plt.imread(_imagePath)
_pixels = _originalPixels.astype(float)/255.
_pixels = _pixels.reshape(-1, 3)

_iteration = 20

# </editor-fold>


# <editor-fold desc="Auxiliary functions">

# Function role is to initiate a dictionary {[centroid_name] : [empty_list_of_pixels]}
def InitCentroidDictionary(centroids):
    dictionary = {}
    for index, centroid in enumerate(centroids):
        dictionary[index] = []
    return dictionary


# Function role is to calculate the new location of the centroids
def SetCentroids(dictionary, centroids):
    newCentroidsValues = []
    for centroid in dictionary.keys():
        pixelsDetails = dictionary[centroid]
        temp = np.zeros(3)                                              # Init an array filled with zeros

        for value in pixelsDetails:                                     # Iterate each pixel for the current centroid
            temp = np.add(temp, value)                                  # Sum all the values of the pixels into 'temp'
        # Calculate the mean of the values. If the sum is zero, then we will divide by 1
        points = np.divide(temp, max(len(pixelsDetails), 1))
        points = points.round(4)
        newCentroidsValues.append(points)

    if str(centroids) == str(newCentroidsValues):                       # Convergence
        return newCentroidsValues, True

    return newCentroidsValues, False

# </editor-fold>


# <editor-fold desc="K_Means Algorithm">

# Function role is to run the K_Means algorithm
def K_Means(centroids, pixels, counter, output):
    stop = False
    f = open(output, "w+")
    dictionary = InitCentroidDictionary(centroids)                      # Init an empty instance of dictionary

    for loop in range(counter):
        if stop:                                                        # If we found Convergence
            break
        for pixel in pixels:                                            # Iterate each of the pixels
            distances = []                                              # Init an array for distances from the centroids

            for centroid in centroids:                                  # Iterate each of the centroids
                # Calculate the distance between the current pixel and each of the centroids & add it to the array
                distances.append([np.linalg.norm(pixel - centroid)])

            classificationIndex = distances.index(min(distances))       # Find the index of the minimum value
            dictionary[classificationIndex].append(pixel)               # Insert the pixel to the list of centroids

        centroids, stop = SetCentroids(dictionary, centroids)           # Set the new location of the centroids
        message = f"[iter {loop}]:{','.join(str(i) for i in centroids)}"
        f.write(message + "\n")
        dictionary = InitCentroidDictionary(centroids)                  # Init an empty instance of dictionary
    f.close()

# </editor-fold>


if len(sys.argv) != 4:
    print("Error: Insufficient arguments!")
    sys.exit()
else:
    K_Means(_z, _pixels, _iteration, _outputLogFileName)
