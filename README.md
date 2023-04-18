# Unveiling-Facial-Structure-and-Correlations---EE-399-Spring-2023-Homework-2
**Author:** Brendan Oquist <br>
**Abstract:** This report delves into the analysis of facial images using correlation matrices, eigenvectors, and singular value decomposition (SVD). We examine a dataset containing faces under varying lighting conditions and preprocess the images into a suitable format for analysis. By computing correlation matrices, we identify similarities and differences between faces, and explore the underlying structure of the data through eigenvectors and SVD. Our investigation of the principal component directions highlights the significance of these techniques in dimensionality reduction and feature extraction for applications in computer vision and machine learning.

## I. Introduction and Overview
This project delves into the analysis of facial images using correlation matrices, eigenvectors, and singular value decomposition (SVD) techniques to uncover underlying structures and relationships within the data. We start by introducing the concept of correlation matrices, which provide a quantitative measure of the linear relationship between pairs of images in our dataset. We then explore eigenvectors and eigenvalues as a means of identifying key patterns in the data, highlighting their significance in understanding the underlying structure of facial images.

## II. Theoretical Background
In this section, we provide the necessary mathematical background for facial image analysis, including correlation matrices, eigenvectors, eigenvalues, and singular value decomposition (SVD). We also introduce the procedures we used, such as matrix operations and visualizations.

### 1. **Correlation Matrices** 
A correlation matrix is a square matrix that contains the Pearson correlation coefficients between pairs of variables (or images in our case). The Pearson correlation coefficient measures the strength of the linear relationship between two variables. It is computed as:

$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$,

where $x_i$ and $y_i$ are the individual data points, and $\bar{x}$ and $\bar{y}$ are the means of the respective variables.

In our context, we compute the dot product between pairs of images as the correlation, which simplifies the correlation calculation to:

$c_{jk} = x_j^T x_k$,

where $x_j$ and $x_k$ are columns of the matrix X.

### 2. **Eigenvectors and Eigenvalues** 
Eigenvectors and eigenvalues are fundamental concepts in linear algebra that provide insights into the underlying structure of a matrix. For a given square matrix A, an eigenvector v and its corresponding eigenvalue λ satisfy the following equation:

$A\textbf{v} = \lambda\textbf{v}$

Eigenvectors represent the directions in which the matrix A stretches or compresses the data, while eigenvalues indicate the magnitude of that stretching or compression. In our analysis, we compute the eigenvectors and eigenvalues of the matrix Y = XX^T to identify key patterns in the facial image dataset.

### 3. **Singular Value Decomposition (SVD)** 
Singular value decomposition (SVD) is a factorization technique that decomposes a given matrix X into three matrices U, S, and V^T:

$X = USV^T$

U and V are orthogonal matrices containing the left and right singular vectors of X, respectively, while S is a diagonal matrix containing the singular values of X. SVD can be used for dimensionality reduction, feature extraction, and data compression by selecting a subset of singular vectors that capture the most significant variations in the data.

In our analysis, we perform SVD on the facial image matrix X and examine the principal component directions to uncover the underlying structure of the data.

### 4. **Matrix Operations and Visualization** 
To effectively analyze the facial images, we employ various matrix operations such as dot products, transposes, and decompositions. Additionally, we use visualization techniques like pcolor and imshow to display correlation matrices and SVD modes, which provide a visual understanding of the relationships between images and the significant patterns captured by the SVD modes.

## III. Algorithm Implementation and Development
In this section, we detail the implementation of the facial correlation models and techniques, including the code and the steps taken to apply them to the Yale Faces dataset. We compute correlation matrices, identify the most and least correlated images, and perform eigenvalue and singular value decomposition analyses.

**Loading and Preprocessing the Dataset** <br>
We imported the Yale Faces dataset using the provided link and loaded the matrix X containing the grayscale images.

```
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

results=loadmat('yalefaces.mat')
X=results['X']
```

**Compute the 100x100 Correlation Matrix** <br>
We calculated the correlation matrix C for the first 100 images in the matrix X, with each element given by c_jk = x_j.T * x_k, where x_j is the jth column of the matrix.

```
# (a) Compute the 100 x 100 correlation matrix C
C = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        C[i, j] = np.dot(X[:, i].T, X[:, j])

# Plot the correlation matrix using pcolor
plt.pcolor(C)
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()
```

**Identifying Most and Least Correlated Images** <br>
We determined the most and least correlated images by finding the maximum and minimum off-diagonal elements in the correlation matrix. We then plotted these images.

```
# (b) Find the most highly correlated and most uncorrelated images
upper_triangle_indices = np.triu_indices(100, 1)
max_corr_index = np.argmax(C[upper_triangle_indices])
min_corr_index = np.argmin(C[upper_triangle_indices])

max_corr_i, max_corr_j = upper_triangle_indices[0][max_corr_index], upper_triangle_indices[1][max_corr_index]
min_corr_i, min_corr_j = upper_triangle_indices[0][min_corr_index], upper_triangle_indices[1][min_corr_index]

# Plot the most highly correlated images
plt.figure()
plt.subplot(121)
plt.imshow(X[:, max_corr_i].reshape(32, 32), cmap='gray')
plt.title('Most Highly Correlated Image 1')
plt.subplot(122)
plt.imshow(X[:, max_corr_j].reshape(32, 32), cmap='gray')
plt.title('Most Highly Correlated Image 2')
plt.show()

# Plot the most uncorrelated images
plt.figure()
plt.subplot(121)
plt.imshow(X[:, min_corr_i].reshape(32, 32), cmap='gray')
plt.title('Most Uncorrelated Image 1')
plt.subplot(122)
plt.imshow(X[:, min_corr_j].reshape(32, 32), cmap='gray')
plt.title('Most Uncorrelated Image 2')
plt.show()

```

**Repeat for 10x10 Correlation Matrix** <br>
We calculated the correlation matrix for a specific set of 10 images and plotted the resulting 10x10 correlation matrix.

```
# (c) Compute the 10 x 10 correlation matrix C for the specified images
selected_images = [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]  # Subtract 1 to adjust for Python's 0-based indexing
C = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        C[i, j] = np.dot(X[:, selected_images[i]].T, X[:, selected_images[j]])

# Plot the correlation matrix using pcolor
plt.pcolor(C)
plt.colorbar()
plt.title('10x10 Correlation Matrix')
plt.show()
```

**Create Matrix Y and Find First Six Eigenvectors** <br>
In this section, we create the matrix Y by multiplying X with its transpose. Then, we find the first six eigenvectors with the largest magnitude eigenvalues, which will be useful for further analysis.
```
from numpy.linalg import eig, norm
from scipy.linalg import svd

# (d) Create the matrix Y = XX^T and find the first six eigenvectors with the largest magnitude eigenvalue
Y = np.dot(X, X.T)
eigenvalues, eigenvectors = eig(Y)
sorted_indices = np.argsort(eigenvalues)[::-1]

first_six_eigenvectors = eigenvectors[:, sorted_indices[:6]]
```

**SVD and First Six Principal Component Directions** <br>
Here, we perform Singular Value Decomposition (SVD) on the matrix X to obtain the first six principal component directions. These directions represent the most significant variations in the data.

```
# (e) SVD the matrix X and find the first six principal component directions
U, S, Vt = svd(X, full_matrices=False)
first_six_principal_components = U[:, :6]

# Print the first six eigenvectors and the first six principal component directions
print("First six eigenvectors with the largest magnitude eigenvalue:")
print(first_six_eigenvectors)
print("\nFirst six principal component directions:")
print(first_six_principal_components)
```

**Compare First Eigenvector and First SVD Mode** <br>
We compare the first eigenvector obtained from matrix Y with the first SVD mode. The norm of the difference of their absolute values helps us understand their similarity or dissimilarity.

```
# (f) Compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the
# norm of difference of their absolute values
v1 = first_six_eigenvectors[:, 0]
u1 = first_six_principal_components[:, 0]

norm_diff = norm(np.abs(v1) - np.abs(u1))
print("Norm of difference of absolute values of the first eigenvector and the first SVD mode:", norm_diff)
```

**Compute the percentage of variance captured by each of the first 6 SVD modes** <br>
In this section, we calculate the percentage of variance captured by each of the first 6 SVD modes. This gives us an idea of how well these modes represent the variability in the dataset. The plotted SVD modes offer a visual representation of the primary variations in the face data.

```
total_variance = np.sum(S ** 2)
percentage_variance = (S[:6] ** 2) / total_variance * 100
print("\nPercentage of variance captured by each of the first 6 SVD modes:")
print(percentage_variance)

# Plot the first 6 SVD modes
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(U[:, i].reshape(32, 32), cmap='gray')
    plt.title(f'SVD Mode {i+1}')
plt.show()
```

## IV. Computational Results
**Correlation Matrix and Most (Un)correlated Images** <br>
We computed the 100x100 correlation matrix for the first 100 images in the dataset and identified the most correlated and most uncorrelated image pairs. Visualizing these pairs revealed similarities and differences in their facial features, lighting, and expressions. <br>
![image](https://user-images.githubusercontent.com/103399658/232252697-0cb68634-96b4-4904-bde9-2f0f3f0c6348.png)
![image](https://user-images.githubusercontent.com/103399658/232252711-7a3f5718-0ca3-49f8-93f1-d7d0c2b570ba.png)
![image](https://user-images.githubusercontent.com/103399658/232252714-acea5c40-b5d3-4c24-b053-0974483d4f2c.png)

As demonstrated, the most correlated images are clearly the same person, under subtly different lighting conditions. <br>

**Eigenvectors and Principal Components** <br>
We created the matrix Y and found the first six eigenvectors with the largest magnitude eigenvalues. Then, we performed Singular Value Decomposition (SVD) on the matrix X to find the first six principal component directions. These components represent the primary variations in the facial data.<br>

```
First six eigenvectors with the largest magnitude eigenvalue:
[[ 0.02384327  0.04535378  0.05653196  0.04441826 -0.03378603  0.02207542]
 [ 0.02576146  0.04567536  0.04709124  0.05057969 -0.01791442  0.03378819]
 [ 0.02728448  0.04474528  0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [ 0.02082937 -0.03737158  0.06455006 -0.01006919  0.06172201  0.03025485]
 [ 0.0193902  -0.03557383  0.06196898 -0.00355905  0.05796353  0.02850199]
 [ 0.0166019  -0.02965746  0.05241684  0.00040934  0.05757412  0.00941028]]

First six principal component directions:
[[-0.02384327  0.04535378  0.05653196 -0.04441826  0.03378603 -0.02207542]
 [-0.02576146  0.04567536  0.04709124 -0.05057969  0.01791442 -0.03378819]
 [-0.02728448  0.04474528  0.0362807  -0.05522219  0.00462854 -0.04487476]
 ...
 [-0.02082937 -0.03737158  0.06455006  0.01006919 -0.06172201 -0.03025485]
 [-0.0193902  -0.03557383  0.06196898  0.00355905 -0.05796353 -0.02850199]
 [-0.0166019  -0.02965746  0.05241684 -0.00040934 -0.05757412 -0.00941028]]
```
```
Norm of difference of absolute values of the first eigenvector and the first SVD mode: 9.640038007624114e-16
Percentage of variance captured by each of the first 6 SVD modes:
[72.92756747 15.28176266  2.56674494  1.87752485  0.63930584  0.59243144]
```

**Visualization of SVD Modes** <br>
We calculated the percentage of variance captured by each of the first six SVD modes, which helped us understand how well these modes represent the variability in the dataset. Additionally, we plotted the SVD modes to visualize the primary variations in the face data. <br>
```
Percentage of variance captured by each of the first 6 SVD modes:
[72.92756747 15.28176266  2.56674494  1.87752485  0.63930584  0.59243144]
```
![image](https://user-images.githubusercontent.com/103399658/232252896-24476293-f9b2-42f4-970d-2b14d0ada5c9.png)

The SVD modes clearly highlight important features in differentiating and recognizing faces, although they're not always components that humans may point out or recognize. SVD mode 1 displays the eyes, nostrils, and mouth prominently, as well as shading components of face shape. As the faces are displayed under different lighting conditions, some of the highlighted features are very lighting dependent. For example, SVD mode 4 clearly pulls from the lighting condition in which the light source was displayed prominently underneath the face. SVD mode 2, on the other hand, dispalys features of a face that are lit from the left. This demonstrates some of the flaws of relying on SVD modes entirely for facial recognition or differentiation. 

## V. Summary and Conclusions
In conclusion, our computational results showcase the importance of analyzing facial data using correlation matrices, eigenvectors, and principal components. These techniques help us understand the primary variations and relationships within the dataset, allowing for better representation and analysis of complex facial data.
