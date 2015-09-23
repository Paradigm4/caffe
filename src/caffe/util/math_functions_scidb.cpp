
#include <unistd>

#include "caffe/util/math_functions_scidb.hpp"

namespace caffe {

//
// curl to interface with SciDB "shim" via URLs
//
void stuff_to_do() {
    easyhandle = curl_easy_init()
    curl_easy_setopt(handle, CURLOPT_URL, "http://domain.com/")
    // TODO: what is the value for SciDB at draper?
    // TODO: this will probalby need to be passed in by env. var.
    
    curl_easy_setopt(handle, CURLOPT_VERBOSE, 1L)

    //curl_easy_setopt(handle, CURLOPT_HEADER, 1L)
    //curl_easy_setopt(handle, CURLOPT_DEBUGFUNCTION, )
    
    // NOTE: string can be released after setopt
    // curl_easy_reset()
    // curl_easy_dupheandle()

    //
    // (SDF) send a data file
    //

    // SDF.1 make a callback to supply the data
    size_t supplyData(char *bufptr, size_t size, size_t nitems, void * userp)
    {
        blah blah blah
    }
    // set the read callback
    curl_easy_setopt(easyhandle, CURLOPT_READDATA, &filedata);
    // indicate desire to upload
    curl_easy_steopt(easyhandle, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(easyhandle, CURLOPT_INITFILESIZE_LARGE, file_size);
    success = curl_easy_perform(easyhandle);

    //
    // RQR Read Query Result
    //
    // formulate query:  domain.com/ + query string
    curl_easy_setopt(handle, CURLOPT_URL, "http://domain.com/+querystuff")

    //
    // Read Query Result
    //
    size_t writeData(char *bufptr, size_t size, size_t nitems, void * userp)
    {
        blah blah blah
    }
    curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION, writeData);
    curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA, &internal_struct); // 4th arg to writeData callback
    success = curl_easy_perform(easyhandle);
}


//
// execute scidb query
//
void scidbQuery(std::string query)
{
    abort();      // TODO: replace with code that uses the client library
                  //       see our iquery, for example.
                  //       for an example of using shim, see
                  //       github implementations for R and Python
                  //       scidbR uses libcurl-devel (ubuntu libcur14-gnutls-dev)
}

//
// create array for matrix
//
void create_matrix(std::string name, int64_t rows, int64_t cols)
{
    stringstream query;
    query << "create tmp array " << name << " <val:double>[r=0:"<<rows<<",1024,0"
                                                      << ",c=0:"<<cols<<",1024,0]" ;
    scidbQuery(query.str());
}

//
// send matrix data to scidb and load it
//
void send_and_load_matrix(std::string name, int64_t rows, int64_t cols, double* data)
{
    // TODO: how will it be sent?  copy a file by scp? libcurl to shim?
    abort();  // must implement
    
    // load the tmp file
    stringstream query;
    query << "store(load(<tmpfile>,<name>))"
    scidbQuery(query.str());
}

//
// NOTE: send_load is optional due to the utility of empty "C" matrices
//       (see gemm)
//
void create_send_load_matrix(std::string name, int64_t rows, int64_t cols, double* send_data)
{
    create_matrix(name, rows, cols);
    tmpfilename;
    if (send_data) {
        send_and_load_matrix(data, rows, cols, tmpfilename);
    }
}

void gemm(std::string nameA, std::string nameB, std::string nameC)
{
    query("gemm(A,B,C,'TRANSA=<TransA>; TRANSB=<TransB>; ALPHA=<alpha>; BETA=<beta>')");
}

template<>
void caffe_scidb_gemm<double>(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                              const double alpha, const double* A, const double* B, const double beta,
                              double* C)
{
    if (getenv("CAFFE_SCIDB_GEMM")) {   // TODO: see below should this policy be in here or where called from caffe_{cpu,gpu}_gemm
        // without tranpositions, A is MxK, B is KxN, and C is MxN (row major)
        int lda = (TransA == CblasNoTrans) ? M : K;  // correct order?
        int tda = (TransA == CblasNoTrans) ? K : M;

        int ldb = (TransB == CblasNoTrans) ? K : N;
        int tdb = (TransB == CblasNoTrans) ? N : K;

        //
        // get connection to SciDB
        // if its down, should we use the cblas?
        //

        // TODO: good temporary names for A,B,C, and product
        Type conn = get_cached_scidb_connection();

        // A
        scidb_create_send_load_matrix();

        // B
        scidb_create_send_load_matrix();

        // C
        // only send data if it is both
        // present and alpha/beta is non-zero
        double *c_data = NULL;
        if (xxxx) {
            c_data = blah blah blah;
        }
        
        result = gemm(std::string nameA, std::string nameB, std::string nameC);
        
        // pull result and copy into array at argument C 
        copy(result, C);

    } else {
        // TODO: decide about fall-back
        // fall back to clbas_dgemm?  or gpu?
        // or give an error?
        // or should caffe_cpu_gemm() and caffe_gpu_gemm()
        // both try scidb_dgemm() first and fall back
        // if an exception is raised?
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    } 
}

template<>
void caffe_scidb_gemm<float>(const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                             const float alpha, const float* A, const float* B, const float beta,
                             float* C)
{
    abort();
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                ldb, beta, C, N);
}

}  // namespace caffe
