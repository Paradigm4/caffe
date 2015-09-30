#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

#include <assert.h>
#include <time.h>
#include <unistd.h>

#include <curl/curl.h>       // curl
#include <curl/easy.h>       // curl

#include "caffe/util/math_functions_scidb.hpp"

namespace scidb {

//
// timing helpers
//

double clock_getsecs(long clock)
{
    struct timespec timeSpec;
    clock_gettime(clock, &timeSpec);
    // TODO: check for bad return and return NaN or throw. which?
    return double(timeSpec.tv_sec) + 1e-9*timeSpec.tv_nsec;
}

double getsecs()
{
    return clock_getsecs(CLOCK_MONOTONIC_RAW);
}

//
// curl to interface with SciDB "shim" via URLs
//
// FYI: this could potentially be done faster via the client library directly
//      however, there is no way to upload data that way
//


//
// a detailed debug function (the built in one doesn't show what is sent)
//
int my_trace(CURL *handle, curl_infotype type, const char *data, size_t size, void *userp)
{
    const char *infoTypeText = NULL;
    std::string printString("[not printable]");
    bool printEndl = true;

    switch (type) {
    default: /* in case a new one is introduced to shock us */
        return 0;
    case CURLINFO_TEXT:
        // infoTypeText = "== Info: " ;
        printString = std::string(data, size);
        printEndl = false;
        break;
    case CURLINFO_HEADER_OUT:
        infoTypeText = "=> Send header";
        printString = std::string(data, size);
        printEndl = false;
        break;
    case CURLINFO_DATA_OUT:
        infoTypeText = "=> Send data";
        break;
     case CURLINFO_SSL_DATA_OUT:
        infoTypeText = "=> Send SSL data";
        break;
     case CURLINFO_HEADER_IN:
        infoTypeText = "<= Recv header";
        printString = std::string(data, size);
        printEndl = false;
        break;
     case CURLINFO_DATA_IN:
        infoTypeText = "<= Recv data";
        break;
     case CURLINFO_SSL_DATA_IN:
        infoTypeText = "<= Recv SSL data";
        break;
   }

   if(infoTypeText) {
       std::cerr << "@ " << infoTypeText << " " << size << " b" << std::endl;
   }
   std::cerr << "@ " << printString ;
   if (printEndl) {
       std::cerr << std::endl;
   }
   return 0;
}
 

//
// a callback for receiving text data via curl
//

size_t receiveString(char *bufptr, size_t size, size_t nitems, void * userp)
{
    size_t bytesConsumed = size*nitems;
    // cast (yuck) the callback (yuck) data to its presumed type
    std::string* retStr = reinterpret_cast<std::string*>(userp);

    bool cerrDebug=false;
    if(cerrDebug) {
        std::string debugStr(bufptr, bytesConsumed);      // full string, untrimmed
        std::cerr << "@@@@@@@@@@@@@@ receiveString: received text start." << std::endl;
        std::cerr << debugStr << std::endl;
        std::cerr << "@@@@@@@@@@@@@@ receiveString: received text end." << std::endl;
    }

    // so now what we'll do is return a string to userp [aka retStr], unless it is NULL
    if (!retStr) {
        std::cerr << "receiveString: warning: caller discarded " << size*nitems << " bytes outright." << std::endl;
    } else {

        // turn it into a string (drop one or two chars of whitespace frequently added at end)
        ssize_t trimmedLen = bytesConsumed;
        for( ; trimmedLen > 0 && isspace(bufptr[trimmedLen-1]) ; --trimmedLen);
        
        *retStr = std::string(bufptr, trimmedLen);
    }

    return bytesConsumed; // how much consumed, not how much returned
}

//
// and another for receiving binary data
//
struct RecvStuff {
    char *       recvData;
    size_t       bufSize;
    size_t       dataSize;
};

size_t receiveData(char *bufptr, size_t size, size_t nitems, void * userp)
{
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "@@@@@@@@@@@@@@ receiveData: size: " << size << std::endl;
        std::cerr << "@@@@@@@@@@@@@@ receiveData: nitems: " << nitems << std::endl;
        std::cerr << "@@@@@@@@@@@@@@ receiveData: returning: " << size*nitems << std::endl;
    }


    size_t inBytes = size * nitems;
    // cast (yuck) the callback (yuck) data to its presumed type
    RecvStuff * output(reinterpret_cast<RecvStuff*>(userp));
    //TODO remove std::string ** const retStr(reinterpret_cast<std::string**const>(userp));

    // so now what we'll do is return a string to userp [aka retStr], unless it is NULL
    if (!output) {
        std::cerr << "receiveData: warning, caller discarded the full " << size*nitems << " bytes." << std::endl;
    } else {
        // transfer no more than allowed (receiveData maybe called repeatedly)
        size_t copyBytes = std::min(inBytes, output->bufSize - output->dataSize); // note may be part filled already
        memcpy(output->recvData + output->dataSize, bufptr, copyBytes);
        output->dataSize += copyBytes;

        size_t remainder = inBytes - copyBytes;
        if (remainder > 0) {
            // some was left over and not copied
            std::cerr << "receiveData: overflow of " << size*nitems << " bytes." << std::endl;
            // TODO: should this raise an exception?
        }
    }

    return inBytes; // TODO: figure out if this shold be copyBytes vs inBytes
}



//
// a callback for sending data via curl
//
struct SendStuff {
    const char * sendSrc;
    size_t sizeLeft;
};

size_t sendData(char *bufptr, size_t size, size_t nitems, void * userp)
{
    //std::string * const dataStr(reinterpret_cast<std::string*const>(userp));
    struct SendStuff* sendStuff = reinterpret_cast<SendStuff*>(userp);

    size_t  minSize = std::min(sendStuff->sizeLeft, size*nitems);
    if(minSize < 1) return 0;

    memcpy(bufptr, sendStuff->sendSrc, minSize);
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "%%%%%%%%%%% sendData: size " << size << std::endl;
        std::cerr << "%%%%%%%%%%% sendData: nitems " << nitems << std::endl;
        std::cerr << "%%%%%%%%%%% sendData: sent " << minSize << " bytes." << std::endl;
        std::cerr << "%%%%%%%%%%% sendData: from " << (void*)(sendStuff->sendSrc) << " bytes." << std::endl;
    }
    sendStuff->sendSrc += minSize;
    sendStuff->sizeLeft -= minSize;


    return minSize;
}

//
// run a simple query via curl (no array uploads or downloads)
//
std::string doHTMLGetString(CURL* easyHandle, const std::string& URL, bool resultExpected=true)
{
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "XXXXXXX DEBUG doHTMLGetString, URL is: " << URL << std::endl; 
    }

    // TODO: make this a modifiable, empty string
    std::string receivedStr;

    // receive anything it sends
    curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, receiveString);
    if (resultExpected) {
        curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, &receivedStr);
    } else {
        curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, NULL);
    }
    curl_easy_setopt(easyHandle, CURLOPT_HTTPGET, 1L);

    curl_easy_setopt(easyHandle, CURLOPT_URL, URL.c_str());
    CURLcode code = curl_easy_perform(easyHandle);
    if(code != CURLE_OK) {
        throw std::runtime_error("ERROR doHTMLGetString: perform failed");
    }
    if(resultExpected) {
        if(receivedStr.size() == 0) {
            throw std::runtime_error("ERROR doHTMLGetString: no data received");
        }
    } else {
        if(receivedStr.size() != 0) {
            throw std::runtime_error("ERROR doHTMLGetString: received unexpected data");
        }
    }

    // TODO: possible that whitespace trimming should be done here,
    // TODO: possbile that whitespace trimming should be option-controlled

    if (cerrDebug && resultExpected) {
            std::cerr << "XXXXXXX DEBUG doHTMLGetString, received string is: " << receivedStr << std::endl; 
    }
    return receivedStr;
}

//
// Data version of the above, for getting binary data back
//  (caller interprets the bytes)
//
size_t doHTMLGetData(CURL* easyHandle, const std::string& URL, char * data, size_t dataMax)
{
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "XXXXXXX DEBUG doHTMLGetData, URL is: " << URL << std::endl; 
        std::cerr << "XXXXXXX DEBUG doHTMLGetData, dataMax is: " << dataMax << std::endl; 
    }

    RecvStuff recvStuff;
    recvStuff.recvData = data;
    recvStuff.bufSize = dataMax;
    recvStuff.dataSize = 0;

    // TODO: how to handle the case where more is received that the length of data?
    //       throwing an exception might be a problem

    // receive anything it sends
    curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, receiveData);
    curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, &recvStuff);
    curl_easy_setopt(easyHandle, CURLOPT_HTTPGET, 1L);

    curl_easy_setopt(easyHandle, CURLOPT_URL, URL.c_str());
    CURLcode code = curl_easy_perform(easyHandle);
    if(code != CURLE_OK) {
        if(cerrDebug) {
            std::cerr << "XXXXXXX DEBUG doHTMLGetData, perform failed" << std::endl; 
        }
        throw std::runtime_error("ERROR doHTMLGetData: perform failed");
    }
    if(!recvStuff.dataSize) {
        throw std::runtime_error("ERROR doHTMLGetData: received no data");
    }

    // TODO: possible that whitespace trimming should be done here,
    // TODO: possbile that whitespace trimming should be option-controlled

    if(cerrDebug) {
        std::cerr << "XXXXXXX DEBUG doHTMLGetData, recvStuff.dataSize is: " << recvStuff.dataSize << std::endl; 
    }
    return recvStuff.dataSize;
}

//
// above here, only knowledge is that the protocol is HTTP-based
// below here, we are aware that we are connecting to SciDB via shim protocol
//

//
// run a SciDB query via a particular shim session
//
// TODO: becomes a method of Shim
//
std::string executeQuery(Shim& shim, const std::string& query, const std::string& scalarName=std::string())
{
    // paste together the base URL stuff with the URL-escaped query
    std::string URL = shim.baseURL + "/execute_query?id=" + shim.session
                                   + "&query=" + curl_easy_escape(shim.curlHandle, query.c_str(), 0);
    if (scalarName.size()) {
        URL += "&save=" ;
        URL += "(" ;
        URL += scalarName;
        URL += ")" ;
    }

    // TODO: add a catch here and extend the exception with a from:
    std::string qid = doHTMLGetString(shim.curlHandle, URL);
    return qid;
}
//
//
// run a SciDB query via a particular shim session
//
// TODO: becomes a method of Shim
//
template<typename scalar_tt>
void readBytesMatrix(Shim& shim, scalar_tt* array, size_t nRow, size_t nCol)
{
    // paste together the query and the base URL stuff
    std::stringstream ssURL;
    ssURL << shim.baseURL << "/read_bytes?id=" << shim.session << "&n=" << nRow*nCol*sizeof(scalar_tt);

    size_t bytesToRead = nRow*nCol*sizeof(scalar_tt);
    size_t bytesRead = doHTMLGetData(shim.curlHandle, ssURL.str(), reinterpret_cast<char*>(array), bytesToRead);

    // TODO: add a catch here?  what if too long? too short? 
    if(bytesRead != bytesToRead) {
        abort();
    }
}


//
// scan_matrix
//
void scan_matrix(Shim& shim, const std::string& nameA, const std::string& scalarName)
{
    if (scalarName.size() == 0) {
        throw std::runtime_error("gemm: scalarName must be specified");
    }

    //
    // save and download?
    //
    std::stringstream ss;
    
    // TODO:  MUST PUT a RESHAPE here as bryan told me so that
    //        the array comes back streaming in the right order
    //        actually the reshape here obviates the need for scan
    //
    if(shim.verbose) {
        std::cerr << "XXXXX ********************** FAILED (scan_matrix) to add the required RESHAPE to get the data to come back in order ********************" << std::endl;
    }

    ss << "scan(" << nameA << ")";
    //
    // TODO: close the reshape
    //
                                                  
    if(shim.verbose) {
        std::cerr << "XXXXX scan_matrix query is '" << ss.str() << "'" << std::endl; 
        std::cerr << "XXXXX scan_matrix scalarName is: " << scalarName << std::endl; 
    }
    std::string qid = executeQuery(shim, ss.str(), scalarName);
}

//
// API: create_temp_matrix
//
std::string create_temp_matrix(Shim& shim, size_t nrow, size_t ncol, const std::string& resultName, const std::string& scalarTypeName)
{
    char createQuery[1000];
    sprintf(createQuery, "create TEMP array %s <v:%s>[r=0:%ld-1,1000,0,c=0:%ld-1,1000,0]",
                          resultName.c_str(), scalarTypeName.c_str(), nrow, ncol);
    if(shim.verbose) {
        shim.tsos << "XXXXX create_temp_matrix: createQuery is '" << createQuery << "'" << std::endl; 
    }
    std::string qid = executeQuery(shim, createQuery);
    if(shim.verbose) {
        shim.tsos << "@@@@@ create_temp_matrix: input QID: " << qid << std::endl;
    }
    // FACTOR
    if(shim.verbose) {
        shim.tsos << "@@@@@@ create_temp_matrix: dropping createQuery session" << std::endl;
    }
    std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
    doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
    shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "@@@@@@ create_temp_matrix: new session: " << shim.session << std::endl;
    }
    return resultName;
}

//
// helper, because typeid(T).name() returns "f" and "d", not float and double
//
template<typename scalar_tt>
const char* typeStr(scalar_tt val) { return "unknown type"; }

template<> const char* typeStr<float >(float  val) { return "float" ;}
template<> const char* typeStr<double>(double val) { return "double";}

//
// API: send_matrix
//
// TODO: add matrix source data pointer
// NOTE: shim gets a new session as a side-effect
//
template<typename scalar_tt>
std::string send_matrix(Shim& shim, const scalar_tt* data, size_t nrow, size_t ncol, const std::string& resultName)
{
    const char* scalarName=typeStr(data[0]);

    //
    // upload matrix data
    //
    const size_t dataBytes=nrow*ncol*sizeof(scalar_tt);

    // may not be needed when CURLFORM_BUFFER,BUFFERPTR used
    SendStuff sendStuff;
    sendStuff.sendSrc = reinterpret_cast<const char*>(data);
    sendStuff.sizeLeft = dataBytes;

    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: nrow " << nrow << " ncol " << ncol << std::endl; 
        shim.tsos << "XXXXX send_matrix: uploading '" << sendStuff.sizeLeft << "bytes" << std::endl; 
        shim.tsos << "XXXXX send_matrix: sendSrc " << (void*)sendStuff.sendSrc << std::endl; 
    }
    
    //
    // creates to make the TEMP arrays
    //
    char createLoadQuery[1000];
    sprintf(createLoadQuery, "create TEMP array TMPLOAD123 <v:%s>[i=0:%ld-1,1000,0]", scalarName, nrow * ncol);
    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: createLoadQuery is '" << createLoadQuery << "'" << std::endl; 
    }
    std::string qid = executeQuery(shim, createLoadQuery);
    if(shim.verbose) {
        shim.tsos << "@@@@@ send_matrix: input QID: " << qid << std::endl;
    }
    // FACTOR
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: dropping createLoadQuery session" << std::endl;
    }
    std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
    doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
    shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: new session: " << shim.session << std::endl;
    }

    char createReshapeQuery[1000];
    sprintf(createReshapeQuery, "create TEMP array %s <v:%s>[r=0:%ld-1,1000,0,c=0:%ld-1,1000,0]",
                                resultName.c_str(), scalarName, nrow, ncol);
    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: createReshapeQuery is '" << createReshapeQuery << "'" << std::endl; 
    }
    qid = executeQuery(shim, createReshapeQuery);
    if(shim.verbose) {
        shim.tsos << "@@@@@ send_matrix: input QID: " << qid << std::endl;
    }
    // FACTOR
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: dropping createReshapeQuery session" << std::endl;
    }
    URL = shim.baseURL + "/release_session?id=" + shim.session ;
    doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
    shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: new session: " << shim.session << std::endl;
    }

    //
    // can factor this to doHTMLPost -- similar to doHTMLGetData?
    //
    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~POST~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }
    std::string filenameStr;
    {
    
        // URL
        char uploadURL[1000];
        sprintf(uploadURL, "http://localhost:8080/upload_file?id=%s", shim.session.c_str());
        if(shim.verbose) {
            shim.tsos << "XXXXX send_matrix: uploadURL is '" << uploadURL << "'" << std::endl; 
        }


        // curl POST, multipart, custom headers for binary file upload
        //curl_easy_setopt(shim.curlHandle, CURLOPT_POST, 1L); // want to POST here, then receive the filename back
        //curl_easy_setopt(shim.curlHandle, CURLOPT_POSTFIELDS, NULL); // want to POST here, then GET the filename back
        //curl_easy_setopt(shim.curlHandle, CURLOPT_POSTFIELDSIZE_LARGE, dataBytes); // want to POST here, then GET the filename back
        //curl_easy_setopt(shim.curlHandle, CURLOPT_READFUNCTION, sendData); // send this data
        //curl_easy_setopt(shim.curlHandle, CURLOPT_READDATA, &sendStuff);
        //headers=curl_slist_append(headers, "Content-Disposition: form-data; name=\"fileupload\"; filename=\"data\"");

        struct curl_slist* headers=NULL;
        headers= curl_slist_append(headers, "Expect:");
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPHEADER, headers);

        struct curl_slist* formHeaders=NULL;
        formHeaders= curl_slist_append(formHeaders, "Content-Type: application/octet-stream");
        struct curl_httppost* post=NULL;
        struct curl_httppost* last=NULL;
        CURLFORMcode formcode = curl_formadd(&post, &last,
                                             CURLFORM_COPYNAME,"fileupload",
                                             CURLFORM_BUFFER,  "data",
                                             CURLFORM_BUFFERPTR, data,
                                             CURLFORM_BUFFERLENGTH, dataBytes,
                                             CURLFORM_CONTENTHEADER, formHeaders,
                                             CURLFORM_END);
        if(formcode != CURL_FORMADD_OK) {
            shim.tsos << "XXXXX send_matrix: curl_formadd failed with code: " << formcode << std::endl;
            throw std::runtime_error("send_matrix: curl_formadd failed");
        }
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPPOST, post); // want to POST here, then GET the filename back

        // curl receive pathname of the tmp file back
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEFUNCTION, receiveString);
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEDATA, &filenameStr);

        // run the page
        curl_easy_setopt(shim.curlHandle, CURLOPT_URL, uploadURL);
        CURLcode code = curl_easy_perform(shim.curlHandle);
        if(shim.verbose) {
            shim.tsos << "@@@@@ send_matrix: curl_easy_perform done '" << filenameStr << "'" << std::endl;       // ### need this for the following 'load' query
        }
        curl_formfree(post);
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPHEADER, NULL);    // cancel the header change, it messes up the load query
        curl_slist_free_all(headers);
        curl_slist_free_all(formHeaders);

        if(code != CURLE_OK) {
            throw std::runtime_error("send_matrix: upload_file failed");
        }
        if(filenameStr.size() == 0) {
            throw std::runtime_error("send_matrix: upload_file no remote filename returned");
        }

        if(shim.verbose) {
            shim.tsos << "@@@@@ send_matrix: filename: '" << filenameStr << "'" << std::endl;       // ### need this for the following 'load' query
        }
    }


    //
    // load the data file into tmp ARRAY
    // TODO: don't use a hard-coded name like TMPLOAD123
    // TODO: test for a collision with an existing array
    //       rather than risk it being overwritten or incompatible
    //
    

    // "load" the datafile
    char loadQuery[1000];
    sprintf(loadQuery, "store(input(<v:%s>[i=0:%ld-1,1000,0],'%s',-2,'(%s)'),TMPLOAD123)",
                        scalarName, nrow * ncol, filenameStr.c_str(), scalarName);
    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: loadQuery is '" << loadQuery << "'" << std::endl; 
    }
    qid = executeQuery(shim, loadQuery);
    if(shim.verbose) {
        shim.tsos << "@@@@@ send_matrix: input QID: " << qid << std::endl;
    }

    //
    // TODO: drop session to drop uploaded file, or find way to do it explicitly
    //
    // FACTOR
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: dropping session to release the uploaded file" << std::endl;
    }
    URL = shim.baseURL + "/release_session?id=" + shim.session ;
    doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release

    shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "@@@@@@ send_matrix: new session: " << shim.session << std::endl;
    }

    const int CHECK_DATA_SIZE=1200;
    if(shim.check && (nrow * ncol <= CHECK_DATA_SIZE)) {
        for(size_t c=0; c < 1; c++) {   // NOCHECKIN: remove this loop
            if(shim.verbose) {
                shim.tsos << "@@@@@ send_matrix: CHECK A " << c << " : issuing scan of the uploaded matrix" << std::endl;
            }
            scan_matrix(shim, "TMPLOAD123", scalarName);

            scalar_tt checkData[CHECK_DATA_SIZE];
            memset(checkData, 0, nrow*ncol*sizeof(scalar_tt));
            readBytesMatrix(shim, checkData, nrow, ncol); 
            if(shim.verbose) {
                shim.tsos << "@@@@@ send_matrix: check A " << c << ": results received" << std::endl;
            }
            if(memcmp(data, checkData, nrow*ncol*sizeof(scalar_tt))) {
                shim.tsos << "@@@@@ send_matrix: check A " << c << ": results differ" << std::endl;
                for(size_t i =0; i < nrow*ncol; i++) {
                    if (data[i] != checkData[i]) {
                        shim.tsos << "@@@@@ data["<<i<<"]=" << data[i] << " checkData["<<i<<"]=" << checkData[i] << std::endl;
                    }
                }
            } else {
                if(shim.verbose) {
                    shim.tsos << "TIMING send_matrix: check A " << c << ": passed" << std::endl;
                }
            }

            // FACTOR
            {
                if(shim.verbose) {
                    shim.tsos << "@@@@@@ send_matrix: releasing session from last check A: " << shim.session << std::endl;
                }
                std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
                doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
                shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
                if(shim.verbose) {
                    shim.tsos << "@@@@@@ send_matrix: new session for next query: " << shim.session << std::endl;
                }
            }
        } // for
    } // if

    //
    // reshape to rectangular from a simple vector of data
    // NOTE: this interprets the linear stream of data as row-major, which is useful for c-style matrices.
    // if this C++ code were to be used from, e.g. R, Fortran, (Python/Numpy?), then we would
    // need an argument to this routine that we need a transpose here instead of a simple reshape
    // (or more control over the reshape)
    //
    char reshapeQuery[1000];
    sprintf(reshapeQuery, "store(reshape(TMPLOAD123,<v:%s>[r=0:%ld-1,1000,0,c=0:%ld-1,1000,0]),%s)",
                           scalarName, nrow, ncol, resultName.c_str());
    if(shim.verbose) {
        shim.tsos << "XXXXX send_matrix: reshapeQuery is '" << reshapeQuery << "'" << std::endl; 
    }

    qid = executeQuery(shim, reshapeQuery);
    if(shim.verbose) {
        shim.tsos << "@@@@@ send_matrix: reshape QID: " << qid << std::endl;
    }

    // FACTOR
    {
        if(shim.verbose) {
            shim.tsos << "@@@@@@ send_matrix: releasing session from store(reshape()): " << shim.session << std::endl;
        }
        std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
        doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
        shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
        if(shim.verbose) {
            shim.tsos << "@@@@@@ send_matrix: new session for next query: " << shim.session << std::endl;
        }
    }

    // so lets check the info in the rectangular (resultName) array
    if(shim.check && (nrow * ncol <= CHECK_DATA_SIZE)) {
        if(shim.verbose) {
            shim.tsos << "@@@@@ send_matrix: CHECK2: issuing scan of the reshaped  matrix" << std::endl;
        }
        scan_matrix(shim, resultName, scalarName);
        scalar_tt checkData[CHECK_DATA_SIZE];
        memset(checkData, 0, nrow*ncol*sizeof(scalar_tt));
        readBytesMatrix(shim, checkData, nrow, ncol); 
        if(shim.verbose) {
            shim.tsos << "@@@@@ send_matrix: check2 results received" << std::endl;
        }
        if(memcmp(data, checkData, nrow*ncol*sizeof(scalar_tt))) {
            shim.tsos << "@@@@@ send_matrix: check2 results differ" << std::endl;
            for(size_t i =0; i < nrow*ncol; i++) {
                if (data[i] != checkData[i]) {
                    shim.tsos << "@@@@@ data["<<i<<"]=" << data[i] << " checkData["<<i<<"]=" << checkData[i] << std::endl;
                }
            }
        } else {
            if(shim.verbose) {
                shim.tsos << "@@@@@ send_matrix: check2 passed" << std::endl;
            }
        }

        // FACTOR
        {
            if(shim.verbose) {
                shim.tsos << "@@@@@@ send_matrix: releasing session from scan [check2]: " << shim.session << std::endl;
            }
            std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
            doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
            shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
            if(shim.verbose) {
                shim.tsos << "@@@@@@ send_matrix: new session for next query: " << shim.session << std::endl;
            }
        }
    }
    
    //
    // finally, remove array TMPLOAD123
    //
    qid = executeQuery(shim, "remove(TMPLOAD123)");
    if(shim.verbose) {
        shim.tsos << "@@@@@ remove QID: " << qid << std::endl;
    }

    // FACTOR
    {
        if(shim.verbose) {
            shim.tsos << "@@@@@@ send_matrix: releasing session from remove(TMPLOAD123): " << shim.session << std::endl;
        }
        std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
        doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
        shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
        if(shim.verbose) {
            shim.tsos << "@@@@@@ send_matrix: new session for next query: " << shim.session << std::endl;
        }
    }

    return resultName;
}

//
// GEMM MATRIX
//
// TODO: or should this generate the string, but not execute it?
//
template<typename scalar_tt>
void queryGemm(Shim& shim, const std::string& nameA, const std::string& nameB, const std::string& nameC,
                           bool transA, bool transB, scalar_tt alpha, scalar_tt beta,
                           const std::string& storeArrayName, const std::string& scalarName)
{
    if (storeArrayName.size() && scalarName.size()) {
        throw std::runtime_error("queryGemm: only one of storeArrayName and scalarName may be specified");
    }
    if (storeArrayName.size() + scalarName.size() == 0) {
        throw std::runtime_error("queryGemm: storeArrayName or scalarName must be specified");
        // not 100% true, it would run, but what would be the value of not recording the answer
    }

    if(shim.verbose) {
        std::cerr << "XXXXX storeArrayName: '" << storeArrayName << "', .size(): " << storeArrayName.size() << std::endl;
    }

    //
    // save and download?
    //
    std::stringstream ss;
    if (storeArrayName.size()) {                // storing option
        ss << "store(" ;
    }
    // TODO:  MUST PUT a RESHAPE here as bryan told me so that
    //        the array comes back streaming in the right order
    //
    if(shim.verbose) {
        std::cerr << "XXXXX ********************** FAILED to add the required RESHAPE to get the data to come back in order ********************" << std::endl;
    }

    //
    // there is no gemm for float.  if scalarName == "float"
    // then we need to use project(apply()) to increase array to double precision.
    // and we need to do the reverse at the end
    // NOTE: we could have a fgemm macro that woudl do this
    std::string exprA = nameA;
    std::string exprB = nameB;
    std::string exprC = nameC;
    if (scalarName == std::string("float")) {
        exprA = "project(apply(" + nameA + ",vDbl,double(v)), vDbl)" ;
        exprB = "project(apply(" + nameB + ",vDbl,double(v)), vDbl)" ;
        exprC = "project(apply(" + nameC + ",vDbl,double(v)), vDbl)" ;
    }

    std::stringstream gg;
    gg << "gemm(" << exprA << "," << exprB << "," << exprC ;
    gg << ",'TRANSA=" << int(transA) << ";TRANSB=" << int(transB); 
    gg << ";ALPHA=" << alpha << ";BETA=" << beta << "')";

    if (scalarName == std::string("float")) {
        ss << "project(apply(" << gg.str() << ", vFlt, float(gemm)), vFlt)";
    } else {
        ss << gg.str();
    }

    //
    // TODO: close the reshape
    //
    if (storeArrayName.size()) {                // storing option
        ss << "," << storeArrayName << ")";
    }
                                                  
    if(shim.verbose) {
        std::cerr << "XXXXX queryGemm: gemmQuery is '" << ss.str() << "'" << std::endl; 
    }
    std::string qid = executeQuery(shim, ss.str(), scalarName);
}

//
// dividing line for breaking the file into pieces
// below depends on the above
//

//
// get an active connection to the scidb "shim" process
// to communication with SciDB database.
// TODO: may want to rename this as SciDBProxy or similar
//
Shim& getShim()
{
    static Shim * cachedShim = NULL;   // what we return, hopefully correctly initialized
                   

    if(!cachedShim) {
        // curl init
        CURL* easyHandle = curl_easy_init();
        if (!easyHandle) {
            throw std::runtime_error("getShim: curl init failed");
        }
        
        //
        // to enable extra debugging
        //
        bool protoDebug=bool(getenv("SCIDB_SHIM_PROTOCOL_DEBUG"));       // traces the HTTP requests
        if(protoDebug) {
            curl_easy_setopt(easyHandle, CURLOPT_DEBUGFUNCTION, my_trace);
            curl_easy_setopt(easyHandle, CURLOPT_VERBOSE, 1L);             // requried to enable my_trace
        }

        // sketch of authorization for use with https (unverified).
        // e.g. use
        // https://localhost:8083/login?username=root&password=Paradigm4
        // to get an authentication token of root:Paradigm4, and then 
        // add [?&]auth=root:Paradigm4 at the end of each query
        // (this latter option would need to be implemented throughout)
        //
        if(false) {
            curl_easy_setopt(easyHandle, CURLOPT_URL, "http://localhost:8080");
            curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, receiveData);
            curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, NULL); // we don't want the data back
            CURLcode code = curl_easy_perform(easyHandle);
            if(code != CURLE_OK) {
                throw std::runtime_error("getShim: initial contact failed");
            }
            // NOTE: string can be released after setopt
            // TODO: this will probalby need to be passed in by env. var.

            // curl_easy_setopt(easyHandle, CURLOPT_HEADER, 1L)
            // curl_easy_setopt(easyHandle, CURLOPT_DEBUGFUNCTION, )
            // curl_easy_reset()
            // curl_easy_dupheandle()
        }

        //
        // get the URL
        //
        std::string baseURL;
        const char * envURL = getenv("SCIDB_SHIM_URL");
        if (!envURL) {
            baseURL = std::string("http://localhost:8080");  // a reasonable default, works if running on a coordinator
                                                             // TODO when authentication implemented, change default to https:
        } else {
            baseURL = std::string(envURL);
        }
        
        //
        // get a session
        //
        std::string fullURL = baseURL + std::string("/new_session");
        std::string session;
        // TODO TODO switch to doHTMLGetString
        if (false) { // TODO, remove me
            curl_easy_setopt(easyHandle, CURLOPT_URL, fullURL.c_str());
            curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, receiveString);
            std::string sessionStr;
            curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, &sessionStr);
            CURLcode code = curl_easy_perform(easyHandle);
            if(code != CURLE_OK) {
                throw std::runtime_error("new session: new session failed");
            }
            if(sessionStr.size()==0) {
                throw std::runtime_error("new session: could not receive QID");
            }
            session = sessionStr;
        } else {
            // active point
            session  = doHTMLGetString(easyHandle, fullURL.c_str());
        }

        // TODO: check the session and if not obtained, throw a std::runtime_error()

        //
        // NOTE: we define the setting of the handle that the shim
        // is initialized.
        // The caller should not have to check the initial session.
        //
        cachedShim = new Shim(easyHandle, baseURL, session, std::cerr);
        if(cachedShim->verbose) {
            cachedShim->tsos << "XXXXX getsShim: created new Shim(,\"" << baseURL << " ," << session << ")" << std::endl;
        }

    }

    return *cachedShim;
}

//
// helper for dgemmScidbServer
//
bool boolFromTransposeFlag(char flag)
{
    // seeming oddness of this mapping is straight from the specification of
    // dgemm(), where n presumably means none
    // and t or c presumably mean transpose or conjugate-transpose
    // if its anything else, this should raise an exception
    if (flag == 'N' || flag == 'n') {
        return false;
    } else if (flag == 'T' || flag == 't' || flag == 'C' || flag == 'c') {
        return true;
    } else {
        throw std::runtime_error("boolFromTransposeFlag: only [NnTtCc] permitted");
    }
}


template<typename scalar_tt>
void dgemmScidbServer(const char& TRANSA, const char& TRANSB,
                      long M, long N, long K,
                      scalar_tt ALPHA, const scalar_tt* aData, long LDA,
                                       const scalar_tt* bData, long LDB,
                      scalar_tt BETA,        scalar_tt* cData, long LDC,
                      Shim& shim)
{
    const char* scalarName=typeStr(aData[0]);

    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: Start" << std::endl;
        shim.tsos << "TIMING dgemmScidbServer: TRANSA:" << TRANSA << " TRANSB: " << TRANSB << std::endl; 
        shim.tsos << "TIMING dgemmScidbServer: ALPHA: " << ALPHA  << " BETA:   " << BETA << std::endl; 
        shim.tsos << "TIMING dgemmScidbServer: M:     " << M      << " N:      " << N    << " K:      " << K << std::endl; 
        shim.tsos << "TIMING dgemmScidbServer: LDA:   " << LDA    << " LDB:    " << LDB  << " LDC:    " << LDC << std::endl; 
    }

    // load dense linear algebra for dgemm
    std::string qid = executeQuery(shim, "load_library('dense_linear_algebra')");
    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: load_library('dense_linear_algebra') done" << std::endl;
    }
    // FACTOR
    if(shim.verbose) {
        shim.tsos << "@@@@@@ dgemmScidbServer: dropping load_library session" << std::endl;
    }
    std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
    doHTMLGetString(shim.curlHandle, URL, false); // nothing returned on a release
    shim.session  = doHTMLGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "@@@@@@ dgemmScidbServer: new session: " << shim.session << std::endl;
    }

    bool transA=(boolFromTransposeFlag(TRANSA));
    long aRow = (transA==false)? M : K;
    long aCol = (transA==false)? K : M;

    bool transB=(boolFromTransposeFlag(TRANSB));
    long bRow = (transB==false)? K : N;
    long bCol = (transB==false)? N : K;

    // there is no transC flag
    long cRow = M; long cCol = N;

    //
    // TODO: maybe assign random names as a default parameter
    // TODO: have the send return the name of the result (errors become exceptions)
    //      1. change errors to exceptions in send_matrix
    //      2. catch them here to print them
    //      3. return the name of the result (text "handle"
    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: calling sendMatrix()" << std::endl;
    }

    //
    // TODO: generate unique temporary names, e.g. GUID-based
    //
    
    std::string aName = send_matrix(shim, aData, aRow, aCol, "TMPA");
    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: A sent" << std::endl;
    }

    std::string bName = send_matrix(shim, bData, bRow, bCol, "TMPB");
    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: B sent" << std::endl;
    }

    std::string cName;
    if(BETA==0) {
        // TODO, make analogue that does not send any data
        cName = create_temp_matrix(shim, cRow, cCol, "TMPC", scalarName);
        if(shim.verbose) {
            shim.tsos << "TIMING dgemmScidbServer: C created empty" << std::endl;
        }
    } else {
        cName = send_matrix(shim, cData, cRow, cCol, "TMPC");
        if(shim.verbose) {
            shim.tsos << "TIMING dgemmScidbServer: C sent" << std::endl;
        }
    }

    bool debugUsingShow=false;
    if(debugUsingShow) {
        scan_matrix(shim, aName, scalarName);
        if(shim.verbose) {
            shim.tsos << "TIMING dgemmScidbServer: scan_matrix called" << std::endl;
        }
    } else {
        queryGemm(shim, aName, bName, cName, transA, transB, /*alpha*/ALPHA, /*beta*/BETA, "", scalarName);
        // answer should be
        // [22.5 49.5]
        // [28.5 64.5]
        if(shim.verbose) {
            shim.tsos << "TIMING dgemmScidbServer: queryGemm called" << std::endl;
        }
    }

    //and read the output back into an array
    readBytesMatrix(shim, cData, cRow, cCol); 
    if(shim.verbose) {
        shim.tsos << "TIMING dgemmScidbServer: results received" << std::endl;
    }

    // and remove the arrays
    qid = executeQuery(shim, "remove(TMPA)");
    if(shim.verbose) {
        shim.tsos << "@@@@@ remove QID: " << qid << std::endl;
    }
    qid = executeQuery(shim, "remove(TMPB)");
    if(shim.verbose) {
        shim.tsos << "@@@@@ remove QID: " << qid << std::endl;
    }
    qid = executeQuery(shim, "remove(TMPC)");
    if(shim.verbose) {
        shim.tsos << "@@@@@ remove QID: " << qid << std::endl;
    }
}

//
// the following is a test program for validating the code above
// to use it, uncomment the implementation of main() that
// follows and compile this file as the entire program
//

template<typename scalar_tt>
int mainTest(scalar_tt value)
{
    Shim& shim = getShim();     // with active session
    shim.verbose= true;         // timed tracing

    const char* scalarName=typeStr(value);


    std::cerr << "main: dense_linear_algebra loaded" << std::endl;
    shim.tsos << "main: dense_linear_algebra loaded" << std::endl;
    shim.tsos << "main: 2" << std::endl;
    shim.tsos << "main: 3" << std::endl;

    // fake data:
    const size_t aRow = 2;
    const size_t aCol = 3;
    scalar_tt aData[aRow*aCol];
    for(size_t i=0; i < aRow*aCol; i++) {
        aData[i] = 1+i ;
    }
    shim.tsos << "main: a filled" << std::endl;

    const size_t bRow = 3;
    const size_t bCol = 2;
    scalar_tt bData[bRow*bCol];
    for(size_t i=0; i < bRow*bCol; i++) {
        bData[i] = 1+i ;
    }
    shim.tsos << "main: b filled" << std::endl;

    const size_t cRow = 2;
    const size_t cCol = 2;
    scalar_tt cData[cRow*cCol];
    for(size_t i=0; i < cRow*cCol; i++) {
        cData[i] = 1 ; 
    }
    shim.tsos << "main: c filled" << std::endl;

    bool doUnitTests = false;
    if(doUnitTests) {
        //
        // TODO: maybe assign random names as a default parameter
        // TODO: have the send return the name of the result (errors become exceptions)
        //      1. change errors to exceptions in send_matrix
        //      2. catch them here to print them
        //      3. return the name of the result (text "handle"
        shim.tsos << "XXXXX UNIT main calling send_matrix(,aData," << aRow << " x " << aCol << ",)" << std::endl;
        std::string aName = send_matrix(shim, aData, aRow, aCol, "UNITA");

        shim.tsos << "XXXXX UINIT main calling send_matrix(,bData," << bRow << " x " << bCol << ",)" << std::endl;
        std::string bName = send_matrix(shim, bData, bRow, bCol, "UNITB");

        shim.tsos << "XXXXX UNIT main calling send_matrix(,bData," << cRow << " x " << cCol << ",)" << std::endl;
        std::string cName = send_matrix(shim, cData, cRow, cCol, "UNITC");

        // need 1 to 3 arrays
        // TODO: when send_matrix done, change this to 3 different arrays
        bool transA=false;
        bool transB=false;
        // run query outputting in binary scalar_tt that can be retrieved with /read_bytes
        bool debugWithShow=false;
        if(debugWithShow) {
            scan_matrix(shim, aName, scalarName);
        } else {
            queryGemm(shim, aName, bName, cName, transA, transB, /*alpha*/1.0, /*beta*/0.5, "", scalarName);
            // answer should be
            // [22.5 49.5]
            // [28.5 64.5]
        }

        //and read the output back into an array D
        const size_t dRow = cRow;
        const size_t dCol = cCol;
        double dData[dRow*dCol];
        readBytesMatrix(shim, dData, dRow, dCol); 
        for(size_t i=0; i< dRow*dCol; i++) {
            shim.tsos << "XXXXX dgemm result["<<i<<"] = " << dData[i] << std::endl;
        }

        // TODO:
        // REMOVE UNITA, UNITB, UNITC
    }

    shim.tsos << "main: calling dgemmScidbServer" << std::endl;

    dgemmScidbServer('N', 'N', aRow, bCol, bRow,
                      1.0, aData, /*LDA*/aRow,
                           bData, /*LDB*/bRow,
                      0.5, cData, /*LDC*/cRow,
                      shim);

    for(size_t i=0; i< cRow*cCol; i++) {
        shim.tsos << "dgemm result["<<i<<"] = " << cData[i] << std::endl;
    }

    // repeat the test with a different beta.
    // and a reset CData
    shim.tsos << "repeating with beta=0.1:" << std::endl;
    for(size_t i=0; i < cRow*cCol; i++) {
        cData[i] = 1 ; 
    }

    dgemmScidbServer('N', 'N', aRow, bCol, bRow,
                      1.0, aData, /*LDA*/aRow,
                           bData, /*LDB*/bRow,
                      0.1, cData, /*LDC*/cRow,
                      shim);

    for(size_t i=0; i< cRow*cCol; i++) {
        shim.tsos << "dgemm result["<<i<<"] = " << cData[i] << std::endl;
    }

    return 0;
}

} // namespace scidb

//
// uncomment the following to turn this into a self-test program
//
#if 0
int main() {
    int a = scidb::mainTest(0.0f);
    int b = scidb::mainTest(0.0);
    return a+b;
}
#endif

namespace caffe {

//
// Caffe gemm provides its own version of the gemm functions which
// are then redirected to a BLAS or GPU implementations.
// This version performs the computation on SciDB, to off-load the multiplication
// onto a parallel cluster, much like using a much like using a GPU
//

char charFromCblasTrans(CBLAS_TRANSPOSE flag)
{
    // Cblas uses the following type of flags
    switch(flag)
    {
    case CblasNoTrans:   return 'N';
    case CblasTrans:     return 'T';
    case CblasConjTrans: return 'C';

    default:
    case AtlasConj:
        throw std::runtime_error("caffe_scidb_gemm only supports NoTrans, Trans, and ConjTrans");
    }
}

template<typename scalar_tt>
bool significantDifference(scalar_tt cData[], scalar_tt cCheck[], size_t numVals)
{
    for(size_t i =0; i < numVals; i++) {
        if (abs(cData[i] - cCheck[i]) > 1e-10 ) { // TODO: fix this to be relative error
            return true;
        }
    }
    return false;
}

template<typename scalar_tt>
void dumpError(scidb::TimeStampedStream& tsos, const scalar_tt* data, size_t nRow, size_t nCol, const std::string& label)
{
    tsos << "caffe_scidb_gemm error: " << label << " nRow " << nRow << " nCol " << nCol << std::endl;
    
    for(size_t i=0; i < nRow*nCol; i++) {
        tsos << "caffe_scidb_gemm error: " << "[" << i << "]= " << data[i] << std::endl;
    }
}

template<typename scalar_tt>
void cblas_gemm(const CBLAS_ORDER CblasRowMajor, CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const scalar_tt alpha, const scalar_tt* aData, const int& lda,
                                        const scalar_tt* bData, const int& ldb, const scalar_tt beta,
                 scalar_tt* cData, const int& ldc);

template<>
void cblas_gemm<float>(const CBLAS_ORDER CblasRowMajor, CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                       const int M, const int N, const int K,
                       const float alpha, const float* aData, const int& lda,
                                          const float* bData, const int& ldb, const float beta,
                       float* cData, const int& ldc)
{
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, aData, lda, bData, ldb, beta, cData, N);
}

template<>
void cblas_gemm<double>(const CBLAS_ORDER CblasRowMajor, CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                        const int M, const int N, const int K,
                        const double alpha, const double* aData, const int& lda,
                                            const double* bData, const int& ldb, const double beta,
                        double* cData, const int& ldc)
{
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, aData, lda, bData, ldb, beta, cData, N);
}

template<typename scalar_tt>
void caffe_scidb_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const scalar_tt alpha, const scalar_tt* aData, const int& lda,
                                             const scalar_tt* bData, const int& ldb, const scalar_tt beta,
                                                   scalar_tt* cData, const int& ldc)
{
    //
    // get connection to SciDB
    // if its down, should we use the cblas?
    //
    scidb::Shim& shim = scidb::getShim();     // with active session

    // set verbosity
    shim.verbose=bool(getenv("SCIDB_SHIM_TRACE"));       // traces the major steps

    // set checking (vs cblas)
    shim.check=  bool(getenv("SCIDB_SHIM_CHECK"));

    // set size limit past which gemm is sent to scidb
    // (since the overhead is too high for small gemms
    // set at 10M flops (assume machine does at lest 10Mflops/sec)
    // and query will take at least one second
    // but we will often need to lower this limit when testing for correctness
    // so it can be adjusted by environment variable
    long localLimit=10*1000*1000; // 10M by default
    const char* limitStr = getenv("SCIDB_SHIM_GEMM_LIMIT");
    if(limitStr) {
        localLimit = atol(limitStr);
    }

    bool doTiming = bool(getenv("SCIDB_SHIM_TIME"));

    // env SCIDB_SHIM_URL was not set
    // or if the matrix is small enough
    if (shim.baseURL.size()==0 || M*N*K <= localLimit) {
        // local will be faster
        double start = scidb::getsecs();
        cblas_gemm(CblasRowMajor,
                   TransA, TransB, M, N, K,
                    alpha, aData, lda,
                           bData, ldb,
                    beta,  cData, N);
        double end = scidb::getsecs();
        // NOCHECKIN ... resolve the && next line
        if(doTiming /*&& M*N*K >= localLimit/10*/ ) { // only print in the magnitude of localLimit
            shim.tsos << "caffe_scidb_gemm: cblas: "<<M<<" * "<<K<<" * "<<N<<" , " << end-start << " s, "
                      << 1e-6*M*K*N/(end-start) << " MFLOP/s" << std::endl; 
        }
        return;
    }


    if(shim.verbose) {
        shim.tsos << "caffe_scidb_gemm: alpha  " << alpha  << " beta   " << beta    << std::endl; 
        shim.tsos << "caffe_scidb_gemm: TransA " << TransA << " TransB " << TransB << std::endl; 
        shim.tsos << "caffe_scidb_gemm: M      " << M      << " N      " << N      << " K      " << K << std::endl; 
        shim.tsos << "caffe_scidb_gemm: lda    " << lda    << " ldb    " << ldb    << " ldc    " << ldc << std::endl; 
    }



    // pre execution, have to save C when checking
    const int MAX_CHECK=1024;
    scalar_tt cDataCopy[MAX_CHECK];
    if (shim.check && (M*N <= MAX_CHECK)) {
        // save a copy of cData for the check calculation
        memcpy(cDataCopy, cData, M*N*sizeof(scalar_tt));
    }

    //
    // TODO: optimize away the sending of the C array
    //       when it is completely zero
    //
    {
        double start = scidb::getsecs();
        // without tranpositions, A is MxK, B is KxN, and C is MxN (row major)
        // not yet clear what lda, ldb, ldc should be
        // int lda = (TransA == CblasNoTrans) ? M : K;  // correct order?
        // int ldb = (TransB == CblasNoTrans) ? K : N;
        // int ldc = N;
        scidb::dgemmScidbServer(charFromCblasTrans(TransA), charFromCblasTrans(TransB),
                                M, N, K,
                                alpha, aData, lda /*M?*/, 
                                     bData, ldb /*K?*/,
                                beta, cData, ldc /*N?*/,
                                shim);
        double end = scidb::getsecs();
        if(doTiming) {
            shim.tsos << "caffe_scidb_gemm: scidb: "<<M<<" * "<<K<<" * "<<N<<" , " << end-start << " s, "
                      << 1e-6*M*K*N/(end-start) << " MFLOP/s" << std::endl; 
        }
    }
            
    if (shim.check && (M*N <= MAX_CHECK)) { // TODO: see below should this policy be in here or where called from caffe_{cpu,gpu}_gemm
        if(shim.verbose) {
            shim.tsos << "caffe_scidb_gemm: checking" << std::endl;
        }
        // TODO: decide about fall-back
        // fall back to clbas_dgemm?  or gpu?
        // or give an error?
        // or should caffe_cpu_gemm() and caffe_gpu_gemm()
        // both try scidb_dgemm() first and fall back
        // if an exception is raised?
        //int lda = (TransA == CblasNoTrans) ? K : M; // reverse
        //int ldb = (TransB == CblasNoTrans) ? N : K; // reverse
        scalar_tt cCheck[MAX_CHECK];
        memcpy(cCheck, cDataCopy, M*N*sizeof(scalar_tt));

        cblas_gemm(CblasRowMajor,
                   TransA, TransB, M, N, K,
                   alpha, aData, lda,
                          bData, ldb,
                   beta,  cCheck, N);

        // TODO now compare original cData with cDataCheck

        if(significantDifference(cData, cCheck, M*N)) {
            // they differ
            shim.tsos << "caffe_scidb_gemm error: -----------------------------------------" << std::endl;
            shim.tsos << "caffe_scidb_gemm error:  ERROR, not same result as cblas_dgemm"    << std::endl;
            shim.tsos << "caffe_scidb_gemm error:                                          " << std::endl;
            shim.tsos << "caffe_scidb_gemm error: alpha  " << alpha  << " beta   " << beta   << std::endl; 
            shim.tsos << "caffe_scidb_gemm error: TransA " << TransA << " TransB " << TransB << std::endl; 
            shim.tsos << "caffe_scidb_gemm error: M      " << M      << " N      " << N      << " K      " << K << std::endl; 
            shim.tsos << "caffe_scidb_gemm error: lda    " << lda    << " ldb    " << ldb    << " ldc    " << ldc << std::endl; 
            shim.tsos << "caffe_scidb_gemm error:                                          " << std::endl;

            dumpError(shim.tsos, aData, M, K, "aData");
            shim.tsos << "caffe_scidb_gemm error:                                          " << std::endl;

            dumpError(shim.tsos, bData, K, N, "bData");
            shim.tsos << "caffe_scidb_gemm error:                                          " << std::endl;

            dumpError(shim.tsos, cDataCopy, M, N, "cDataCopy -- original");
            shim.tsos << "caffe_scidb_gemm error:                                          " << std::endl;

            for(size_t i =0; i < M*N; i++) {
                if (abs(cData[i] - cCheck[i]) > 1e-10 ) {
                    shim.tsos << "@@@@@ cData["<<i<<"]=" << cData[i] << " != cCheck["<<i<<"]=" << cCheck[i] << std::endl;
                }
            }
        } else {
            if(shim.verbose) {
                shim.tsos << "caffe_scidb_gemm error: check passed." << std::endl;
            }
        }
    } 
}

// force instantiation since template not exposed to caller
template
void caffe_scidb_gemm<float >(const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                              const int M, const int N, const int K,
                              const float  alpha, const float * aData, const int& lda,
                                                  const float * bData, const int& ldb, const float  beta,
                                                        float * cData, const int& ldc);
template
void caffe_scidb_gemm<double>(const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                              const int M, const int N, const int K,
                              const double alpha, const double* aData, const int& lda,
                                                  const double* bData, const int& ldb, const double beta,
                                                        double* cData, const int& ldc);


}  // namespace caffe
