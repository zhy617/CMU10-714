#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void single_matrix_mult(const float *A, const float *B, float *C,
               size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            C[i * k + j] = 0;
            for (size_t l = 0; l < n; ++l) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

void first_transpose_matrix_mult(const float *A, const float *B, float *C,
               size_t m, size_t n, size_t k)
{
    // A with shape (n, m) and B with shape (n, k)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            C[i * k + j] = 0;
            for (size_t l = 0; l < n; ++l) {
                C[i * k + j] += A[l * m + i] * B[l * k + j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *Z = new float[batch * k];
    float *grad = new float[n * k];

    for(size_t i = 0; i < m; i += batch){
        size_t batch_size = std::min(batch, m - i);
        // Compute logits
        single_matrix_mult(X + i * n, theta, Z, batch_size, n, k);

        // Compute softmax
        for(size_t j = 0; j < batch_size; ++j){
            for(size_t l = 0; l < k; ++l){
                Z[j * k + l] = exp(Z[j * k + l]);
            }
            float sum = 0;
            for(size_t l = 0; l < k; ++l){
                sum += Z[j * k + l];
            }
            for(size_t l = 0; l < k; ++l){
                Z[j * k + l] /= sum;
            }
            for(size_t l = 0; l < k; ++l){
                if(l == y[i + j]){
                    Z[j * k + l] -= 1;
                }
            }
        }

        // Compute gradient
        first_transpose_matrix_mult(X + i * n, Z, grad, n, batch_size, k);
        for(size_t j = 0; j < n * k; ++j){
            grad[j] /= batch_size;
            grad[j] *= lr;
        }

        // Update theta
        for(size_t j = 0; j < n * k; ++j){
            theta[j] -= grad[j];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
