#ifndef CAFFE_UTIL_MATH_FUNCTIONS_SCIDB_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_SCIDB_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
// this is an implementation of caffe_cpu_gemm() that uses SciDB
// as an accelerator.
template <typename Dtype>
void caffe_scidb_gemm(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                      const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                      Dtype* C);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_SCIDB_H_
