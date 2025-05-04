#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define BASE_BlOCK_X 16
#define BASE_BLOCK_Y 16

#define TILE 4
#define BLOCK_TILE 64
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t rows, size_t cols) {
  /**
   * Utility function to get cuda dimensions for 2D call
   */
  CudaDims dim;
  dim.block = dim3(BASE_BlOCK_X, BASE_BLOCK_Y, 1);
  dim.grid = dim3((rows + BASE_BlOCK_X - 1) / BASE_BlOCK_X,
                  (cols + BASE_BLOCK_Y - 1) / BASE_BLOCK_Y, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    // Calculate the index in the non-compact array a
    size_t a_index = offset;
    size_t gpu_id = gid;
    for(int i = shape.size - 1; i >= 0; i--) {
      size_t dim_index = gpu_id % shape.data[i];
      gpu_id /= shape.data[i];
      a_index += dim_index * strides.data[i];
    }
    out[gid] = a[a_index];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,
                                   CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the setitem operation.  This should effectively map a single entry in the 
   * compact input a, to the corresponding item (at location gid) in the non-compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of *out* array
   *   offset: offset of *out* array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    // Calculate the index in the non-compact array out
    size_t out_index = offset;
    size_t gpu_id = gid;
    for(int i = shape.size - 1; i >= 0; i--) {
      size_t dim_index = gpu_id % shape.data[i];
      gpu_id /= shape.data[i];
      out_index += dim_index * strides.data[i];
    }
    out[out_index] = a[gid];
  }
  /// END SOLUTION
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                     CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the setitem operation.  This should effectively map a single entry in the 
   * compact input a, to the corresponding item (at location gid) in the non-compact array out.
   * 
   * Args:
   *   val: scalar value to write to
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of *out* array
   *   offset: offset of *out* array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    // Calculate the index in the non-compact array out
    size_t out_index = offset;
    size_t gpu_id = gid;
    for(int i = shape.size - 1; i >= 0; i--) {
      size_t dim_index = gpu_id % shape.data[i];
      gpu_id /= shape.data[i];
      out_index += dim_index * strides.data[i];
    }
    out[out_index] = val;
  }
  /// END SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Multiply two CUDA arrays elementwise.
   * Args:
   *   a: Input array 'a' to be multiplied
   *   b: Input array 'b' to be multiplied
   *   out: Output array to store the result of 'a * b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Multiply a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be multiplied
   *   out: Output array to store the result of 'a * val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarMulKernel that will multiply each element of array 'a' by the scalar 'val', 
  // and store the result in array 'out'.
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Divide two CUDA arrays elementwise.
   * Args:
   *   a: Input array 'a' to be divided
   *   b: Input array 'b' to be divided
   *   out: Output array to store the result of 'a / b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be divided
   *   out: Output array to store the result of 'a / val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarDivKernel that will divide each element of array 'a' by the scalar 'val', 
  // and store the result in array 'out'.
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val,
                                   scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Raise each element of a CUDA array to the power of a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be raised to
   *   out: Output array to store the result of 'a ^ val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarPowerKernel that will raise each element of array 'a' to the power of 
  // the scalar 'val', and store the result in array 'out'.
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b,
                                    scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmax(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Take the maximum of two CUDA arrays elementwise.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'max(a, b)'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val,
                                      scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmax(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Take the maximum of a CUDA array and a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be compared
   *   out: Output array to store the result of 'max(a, val)'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarMaximumKernel that will take the maximum of each element of array 'a' 
  // and the scalar 'val', and store the result in array 'out'.
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Compare two CUDA arrays elementwise for equality.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'a == b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val,
                                 scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare a CUDA array with a scalar value for equality.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be compared
   *   out: Output array to store the result of 'a == val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarEqKernel that will compare each element of array 'a' with the scalar 
  // 'val', and store the result in array 'out'.
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b,
                               scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Compare two CUDA arrays elementwise for greater than or equal to.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'a >= b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val,
                                 scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare a CUDA array with a scalar value for greater than or equal to.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be compared
   *   out: Output array to store the result of 'a >= val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarGeKernel that will compare each element of array 'a' with the scalar 
  // 'val', and store the result in array 'out'.
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Take the natural logarithm of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'log(a)'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Take the exponential of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'exp(a)'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Take the hyperbolic tangent of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'tanh(a)'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define L BLOCK_TILE
#define S BLOCK_TILE
#define V TILE

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
                   uint32_t P) {
  /**
   * The CUDA kernel for the matmul operation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   b: CUDA point to b array
   *   out: CUDA point to out array
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  
  /// BEGIN SOLUTION
  
  // S: the step size of N dimension
  // L: the step size of M dimension
  // 

  __shared__ float sA[L][S], sB[S][L];
  float c[V][V] = {0.};
  float a_slice[V], b_slice[V];
  int xblock = blockIdx.x;
  int yblock = blockIdx.y;

  for(int ko=0; ko < N; ko += S) {
    __syncthreads();// 
    // Load A and B tiles into shared memory
    int nthreads = blockDim.x * blockDim.y;
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    for(int j = 0; j < (L*S + nthreads-1)/nthreads; j++){
      int y = (j * nthreads + tid) / L;
      int x = (j * nthreads + tid) % L;
      int Aidx = (xblock * L + x) * N + ko + y;
      sA[x][y] = (xblock * L + x < M && ko + y < N) ? a[Aidx] : 0.0f;
      int Bidx = (ko + y) * P + (yblock * L + x);
      sB[y][x] = (ko + y < N && yblock * L + x < P) ? b[Bidx] : 0.0f;
    }
    __syncthreads();

    // Compute the product of sA and sB tiles
    for(int ki = 0; ki < S; ki++) {
      for(int i = 0; i < V; i++) {
        a_slice[i] = sA[threadIdx.x * V + i][ki];
        b_slice[i] = sB[ki][threadIdx.y * V + i];
      }

      for(int x = 0; x < V; x++) {
        for(int y = 0; y < V; y++) {
          c[x][y] += a_slice[x] * b_slice[y];
        }
      }
    }
    int xbase = xblock * blockDim.x + threadIdx.x;
    int ybase = yblock * blockDim.y + threadIdx.y;
    for(int i = 0; i < V; i++) {
      for(int j = 0; j < V; j++) {
        int idx_x = (xbase * V + i), idx_y = (ybase * V + j);
        if (idx_x < M && idx_y < P) {
          out[idx_x * P + idx_y] = c[i][j];
        }
      }
    }
  }
  /// END SOLUTION
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  CudaDims dim = CudaTwoDim(M, P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size,
                                 size_t size) {
  /**
   * The CUDA kernel for the reduction operation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   reduce_size: size of the dimension to reduce over
   *   size: size of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    // Calculate the index in the non-compact array a
    size_t a_index = gid * reduce_size;
    float max_val = a[gid * reduce_size];
    for(int i = 1; i < reduce_size; i++) {
      max_val = fmax(max_val, a[a_index + i]);
    }
    out[gid] = max_val;
  }
  /// END SOLUTION
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size,
                                 size_t size) {
  /**
   * The CUDA kernel for the reduction operation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   reduce_size: size of the dimension to reduce over
   *   size: size of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    // Calculate the index in the non-compact array a
    size_t a_index = gid * reduce_size;
    float sum_val = 0.0f;
    for(int i = 0; i < reduce_size; i++) {
      sum_val += a[a_index + i];
    }
    out[gid] = sum_val;
  }
  /// END SOLUTION
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
