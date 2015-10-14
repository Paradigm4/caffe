#include <cmath>       // fplassify et. al.
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

// for implementing getsecs()
double clock_getsecs(long clock)
{
    struct timespec timeSpec;
    clock_gettime(clock, &timeSpec);
    // TODO: check for bad return and return NaN or throw. which?
    return double(timeSpec.tv_sec) + 1e-9*timeSpec.tv_nsec;
}

// to use in code
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
// TODO: user userp to indicate if the
// data currently expected is ascii or floats or doubles or unknown
// and fix the "unknown printabilty" below
int my_trace(CURL *handle, curl_infotype type, const char *data, size_t size, void *userp)
{
    static size_t counter=0;
    counter++;

    const char *infoTypeText = NULL;
    std::string printString("");
    bool printEndl = true;

    switch (type) {
    default: /* in case a new one is introduced to shock us */
        return 0;
    case CURLINFO_TEXT:
        infoTypeText = "== Info, " ;
        printString = std::string(data, size);
        printEndl = false;
        break;
    case CURLINFO_HEADER_OUT:
        infoTypeText = "=> Send header,";
        printString = std::string(data, size);
        printEndl = false;
        break;
    case CURLINFO_DATA_OUT:
        infoTypeText = "=> Send data,";
        break;
     case CURLINFO_SSL_DATA_OUT:
        infoTypeText = "=> Send SSL data,";
        break;
     case CURLINFO_HEADER_IN:
        infoTypeText = "<= Recv header,";
        printString = std::string(data, size);
        break;
     case CURLINFO_DATA_IN:
        infoTypeText = "<= Recv data,";
        break;
     case CURLINFO_SSL_DATA_IN:
        infoTypeText = "<= Recv SSL data,";
        break;
   }

   if(infoTypeText) {
       std::cerr << counter << ". " << infoTypeText << " " << size << "b:"; 
   }
   if(printString.size()>0) {
       std::cerr << " [[" << printString << "]]" << std::endl;
   } else {
       std::cerr << " ??"<< std::endl;
   }
   //if (printEndl) {
       //std::cerr << std::endl;
   //}
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
        std::cerr << "[cerrDebug] receiveString: received text start." << std::endl;
        std::cerr << debugStr << std::endl;
        std::cerr << "[cerrDebug] receiveString: received text end." << std::endl;
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

const bool DoReshape=false ;// expermental, use reshape to reorder
                            // the data rather than the receiving code
                            // suggested by Bryan, but it performs poorly
                            // setting it to false which enables code
                            // that re-orders the data on the client
                            // side.
                            // NOTE this is possible because my case
                            // is strictly dense ... in Bryan's case
                            // according to Alex, getting the vector
                            // is one way to determine its dense.

//
// and another for receiving binary data
//

#define myassert(e) { \
    if(!(e)) { \
        std::cerr << "myassert " << #e \
                  << " " << __FILE__ << " " << __LINE__ << std::endl; \
        abort(); \
    } \
}

//
// TODO: Design thought for factoring
// MatrixShimAdapter to a part that controls the sequencing
// and separate classes that have only
// recvData or sendData and the extra data needed
// to support them
//
// squentially generates the next index in a row-major
// global array, for a SciDB-canonical-order set of blocks that
// tile it.
// In the general case, SciDB will use 4 size of block to tile
// the matrix
// A. full square bock, e.g. the first block when matrix is larger than block order
// n. narrow block, on right most column when (F) won't fit
// s. short block , on bottom row when (F) won't fit
// z, narrow, short block, last block when no other will fit
//
// we can generalize the above by allowing the N,S,andNS blocks to equal
// the size of (F) and using their names for the right-most, bottom-most,
// and south-east most blocks
// This way we can set the side of the four block types and just
// advance the state, eg.
//   AAAAAAAAn    n
//   AAAAAAAAn    n 
//   ssssssssz    z   ssssssz
//
// when a dimension is exactly a multiple of the chunk order,
// the notation will be
//   AA                 An
//   AA    rather than  Az
//
// this should allow the A state to have more optimal transfer and
// less involved code when iteration through indices belonging to
// that state
//
// TODO Design the tool and factor RecvMatrix
//

class MatrixShimAdapter {
public:
    MatrixShimAdapter(size_t matRow,  size_t matCol,
               size_t blkRows, size_t blkCols,
               void*  matrix, size_t tSize)
    :
        _matrix(reinterpret_cast<char*>(matrix)),
        _tSize(tSize),
        _matRows(matRow),  _matCols(matCol),
        _blkRows(blkRows), _blkCols(blkCols),
        _matOffRow(0),     _matOffCol(0),       // where cur block is in matrix
        _blkCurRow(0),     _blkCurCol(0),        // where we are in the block
                                                // (and matrix) from Off[set]
        _nBytesSoFar(0),
        _extraBytes(0),
        _recursing(false)
    {
        myassert(sizeof(_extraBuf) >= _tSize);
        memset(_extraBuf, 0, sizeof(_extraBuf));
    }

    void recvData(const char* dataPtr, size_t dataLen) {
        if(DBG) std::cerr << "recvData, dataLen " << dataLen << " nVals " << dataLen/_tSize << std::endl;
        
        // first have to deal with the fact that the amoutn supplied is not always
        // a multiple of _tSize
        // if there was any partial val left over from last time, try to complete it first
        if(_extraBytes && !_recursing) {
            std::cerr << "recvData, _extraBytes " << _extraBytes << std::endl;
            size_t takeBytes = std::min(_tSize-_extraBytes, dataLen);
            memcpy(_extraBuf+_extraBytes, dataPtr, takeBytes);
            _extraBytes += takeBytes;
            myassert(_extraBytes <= _tSize);
            if(_extraBytes >= _tSize) {   // enough to process
                _recursing=true; // skip _extraBytes handling on this call
                recvData(_extraBuf, _extraBytes);
                _recursing=false;
                _extraBytes = 0;  // should be consumed now (CAREFUL?)
            }
            dataPtr += takeBytes;       // advance
            dataLen -= takeBytes;
            if (!dataLen) {
                std::cerr << "recvData, _extraBytes left no more data, early return " << std::endl;
                return;                 // done
            }
        }
        if(_recursing) { myassert(_extraBytes==_tSize && dataLen == _tSize); }

        // data comes in blocks.  each time we are called
        // we may be starting mid-segment
        // by running out of bytes at an arbitrary point we may end
        // mid-segment
        // when less than _tSize bytes remain, we store
        // them in a tmp buffer and return, and next
        // time will augment them until a single value
        // can be consumed. (see above)

        // transfers broken up by row in chunk, as addresses in
        // _matrix are not continuous across these boundaries
        while(dataLen >= _tSize && _nBytesSoFar < _matRows*_matCols*_tSize ) {
            if(DBG) std::cerr << "recvData, @ " << _matOffRow+_blkCurRow
                                         <<"," << _matOffCol+_blkCurCol << std::endl;
            // NOTE, after first pass _blkCurCol == 0
            size_t takeVals = blkColsTrimmed() - _blkCurCol; // Trimmed() changes with state
            // but not more than we have data for
            takeVals = std::min(takeVals, dataLen/_tSize);
            size_t bytes = takeVals * _tSize;
            myassert(bytes <= dataLen); //  must not take more than given

            memcpy(dstAddr(), dataPtr, bytes);

            _nBytesSoFar += bytes;        // advance dstAddr()
            dataPtr += bytes;
            dataLen -= bytes;
            if(DBG) std::cerr << "recvData, takeVals " << takeVals
                              << " leaving " << dataLen/_tSize << std::endl;

            blockEndUpdates(takeVals);
        }
        if(DBG) std::cerr << "recvData, Bend, @ " << _matOffRow+_blkCurRow
                                           <<"," << _matOffCol+_blkCurCol << std::endl;
        myassert(_blkCurCol==0 || dataLen < _tSize); // on a fresh row, unless out of data

        // the only leftovers should be less than _tSize
        // handle it via the _extraBuf
        if(dataLen) {
            if(DBG) std::cerr << "recvData, dataLen " << dataLen
                              << " being added to _extraBytes " << _extraBytes << std::endl;
            myassert(dataLen < _tSize);
            myassert(dataLen + _extraBytes < _tSize);   // it will all fit
            memcpy(_extraBuf+_extraBytes, dataPtr, dataLen);
            _extraBytes += dataLen;
            dataLen = 0;
            assert(_extraBytes < _tSize);
        }
        myassert(dataLen == 0);

        // check -- did we copy dataLen?
        if(DBG) std::cerr << "recvData END: _matOff: " << _matOffRow << ","
                                                      << _matOffCol << std::endl; 
        if(DBG) std::cerr << "recvData END: _blkCur: " << _blkCurRow << ","
                                                      << _blkCurCol << std::endl; 
    }

    //
    // SAME as above, but the memcopy direction is reversed.
    // We control how much to return each time, so
    // we don't have to deal with _extraBytes
    //
    size_t sendData(char* dataPtr, size_t dataLen) {
        if(DBG) std::cerr << "sendData, dataLen " << dataLen << " nVals " << dataLen/_tSize << std::endl;

        size_t bytesThisPass=0; // vs nBytesSoFar which is over multiple calls
        
        // data comes in blocks.  each time we are called
        // 1. we may be starting mid-segment
        // [We don't necesarily want to assume dataLen is ever greater than
        //  the size of a segment]
        // 2. by running out of bytes at an arbitrary point we may end
        // mid-segment

        // transfers broken up by row in chunk, as addresses in
        // _matrix are not continuous across these boundaries

        // for send: assert we have something left to send precondtion
        // (vs for receive, that we took every bit of data they gave us)
        myassert(_nBytesSoFar <= _matRows*_matCols*_tSize );
        if(_nBytesSoFar == _matRows*_matCols*_tSize ) {
            std::cerr << "-------------sendData, sent _nBytesSoFar: " << _nBytesSoFar << std::endl;
            return 0;   // tell them we're done
        }
        
        // send: while room to send && we have not sent everything we have
        while(dataLen >= _tSize && _nBytesSoFar < _matRows*_matCols*_tSize ) {
            if(DBG) std::cerr << "sendData, @ " << _matOffRow+_blkCurRow
                                         <<"," << _matOffCol+_blkCurCol << std::endl;
            // NOTE, after first pass _blkCurCol == 0
            size_t takeVals = blkColsTrimmed() - _blkCurCol; // Trimmed() changes with state
            // but not more than we have data for
            takeVals = std::min(takeVals, dataLen/_tSize);
            size_t bytes = takeVals * _tSize;
            myassert(bytes <= dataLen); //  must not take more than room for

            // send reversed
            memcpy(dataPtr, dstAddr(), bytes);

            bytesThisPass += bytes;
            _nBytesSoFar += bytes;        // advance dstAddr()
            dataPtr += bytes;
            dataLen -= bytes;
            // send modified
            if(DBG) std::cerr << "sendData, takeVals " << takeVals
                              << " leaving nVals" << _matRows*_matCols-_nBytesSoFar/_tSize << std::endl;
            myassert(_nBytesSoFar <= _matRows*_matCols*_tSize); // send added

            blockEndUpdates(takeVals);
        }
        if(DBG) std::cerr << "sendData, Bend, @ " << _matOffRow+_blkCurRow
                                           <<"," << _matOffCol+_blkCurCol << std::endl;
        myassert(_blkCurCol==0 || dataLen < _tSize); // on a fresh row, unless out of data

        // check -- did we copy dataLen?
        if(DBG) std::cerr << "sendData END: _matOff: " << _matOffRow << ","
                                                      << _matOffCol << std::endl; 
        if(DBG) std::cerr << "sendData END: _blkCur: " << _blkCurRow << ","
                                                      << _blkCurCol << std::endl; 
        return bytesThisPass;
    }

    bool isComplete() {
          return (_nBytesSoFar == _matRows * _matCols * _tSize);
    }

    size_t bytesSoFar() { return _nBytesSoFar; }

private:
    enum dummy { DBG=0 };

    size_t blkColsTrimmed() {
        return std::min(_blkCols, _matCols-_matOffCol);
    }
    size_t blkRowsTrimmed() {
        return std::min(_blkRows, _matRows-_matOffRow);
    }

    size_t toEndOfBlkRow() {
        return blkColsTrimmed() - _blkCurCol;
    }

    size_t rowMajIdx() {
        size_t row = _matOffRow + _blkCurRow; 
        size_t col = _matOffCol + _blkCurCol; 
        return _matCols * row + col;
    }

    char* dstAddr() {
        return rowMajIdx() * _tSize + _matrix;
    }

    void assertConsistent() {
        // whole values transmitted
        myassert(_nBytesSoFar%_tSize == 0);
        // compare bytes processed with matOff blkCur
        size_t vals = 0;
        vals += _matCols*_matOffRow;   // num vals in full rows of full blocks
        vals += _blkRows*_matOffCol;  // full blocks to left of current offset

        vals += blkColsTrimmed()*_blkRows; // full rows of current block
        vals += _blkCols;                   // current partial block row

        if (vals != _nBytesSoFar/_tSize) {
            if(DBG) std::cerr << " myassert vals " << vals << " byte so far " << _nBytesSoFar << std::endl;
        }
        myassert(vals == _nBytesSoFar*_tSize);
    }

    void blockEndUpdates(size_t nVals) {
        _blkCurCol += nVals;
        if(DBG) std::cerr << "blockEndUpdates("<<nVals<<"), _blkCurCol now" << _blkCurCol << std::endl;

        myassert(_blkCurRow <= blkRowsTrimmed());
        myassert(_blkCurCol <= blkColsTrimmed());  // how?  _blkCurcol =1000, _matCols=15002, _matOffcol=1500

        if(_blkCurCol == blkColsTrimmed()) {    // goto next row in block
            _blkCurCol = 0;
            _blkCurRow++;
            if(DBG) std::cerr << "blockEndUpdates: fresh row _matOff: " << _matOffRow << ","
                                                                        << _matOffCol << std::endl; 
            if(DBG) std::cerr << "blockEndUpdates: fresh row _blkCurr" << _blkCurRow << ","
                                                                       << _blkCurCol << std::endl; 
        }
        myassert(_blkCurCol <= _blkCols);

        if(_blkCurRow >= blkRowsTrimmed()) { // next block
            _blkCurRow=0; // starting point of next block
            if(DBG) std::cerr << "blockEndUpdates: next block" << std::endl; 

            // change _matOff
            size_t newCol = _matOffCol + _blkCols;
            if ( newCol < _matCols ) { // next block is to the right
                _matOffCol = newCol;   // as of this happening
                // no change to _matOffRow 
            } else { // next block is first in next row
                _matOffCol = 0;
                _matOffRow += _blkRows;
            }
            if(DBG) std::cerr << "blockEndUpdates: offset " << _matOffCol << ","
                                                            << _matOffRow << std::endl; 
        }
    }

    char * const _matrix;                       // TODO: make class not-copyable
    const size_t _tSize;
    const size_t _matRows, _matCols;
    const size_t _blkRows, _blkCols;

    size_t       _matOffRow, _matOffCol;
    size_t       _blkCurRow, _blkCurCol;
    size_t       _nBytesSoFar;

    char         _extraBuf[sizeof(double)];
    size_t       _extraBytes;
    bool         _recursing;
};

size_t CBRecvData(char *bufptr, size_t size, size_t nitems, void * userp)
{
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "[cerrDebug] CBRecvData: size: " << size << std::endl;
        std::cerr << "[cerrDebug] CBRecvData: nitems: " << nitems << std::endl;
        std::cerr << "[cerrDebug] CBRecvData: returning: " << size*nitems << std::endl;
    }


    size_t inBytes = size * nitems;
    // cast (yuck) the callback (yuck) data to its presumed type
    MatrixShimAdapter* output(reinterpret_cast<MatrixShimAdapter*>(userp));

    // so now what we'll do is return a string to userp [aka retStr], unless it is NULL
    if (!output) {
        std::cerr << "CBRecvData: warning, caller discarded the full " << size*nitems << " bytes." << std::endl;
    } else {
        output->recvData(bufptr, inBytes);
    }

    // question: can we take only as many as we want?
    return inBytes; // TODO: figure out if this shold be copyBytes vs inBytes
}

size_t CBSendData(char *bufptr, size_t size, size_t nitems, void * userp)
{
    MatrixShimAdapter* cbMatrix = reinterpret_cast<MatrixShimAdapter*>(userp);
    size_t bytesSent = cbMatrix->sendData(bufptr, size*nitems);

    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "[cerrDebug] CBSendData: size " << size << std::endl;
        std::cerr << "[cerrDebug] CBSendData: nitems " << nitems << std::endl;
        std::cerr << "[cerrDebug] CBSendData: bytesSent " << bytesSent << std::endl;
    }
    return bytesSent;
}

//
// run a simple query via curl (no array uploads or downloads)
//
std::string curlGetString(CURL* easyHandle, const std::string& url, bool resultExpected=true)
{
    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "[cerrDebug] curlGetString, url is: " << url << std::endl; 
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

    curl_easy_setopt(easyHandle, CURLOPT_URL, url.c_str());
    CURLcode code = curl_easy_perform(easyHandle);
    if(code != CURLE_OK) {
        throw std::runtime_error("ERROR curlGetString: perform failed");
    }
    if(resultExpected) {
        if(receivedStr.size() == 0) {
            throw std::runtime_error("ERROR curlGetString: no data received");
        }
    } else {
        if(receivedStr.size() != 0) {
            throw std::runtime_error("ERROR curlGetString: received unexpected data");
        }
    }

    // TODO: possible that whitespace trimming should be done here,
    // TODO: possbile that whitespace trimming should be option-controlled

    if (cerrDebug && resultExpected) {
        std::cerr << "[cerrDebug] curlGetString, received string is: " << receivedStr << std::endl; 
    }
    return receivedStr;
}

const size_t BlkRows=1000, BlkCols=1000;

//
// Data version of the above, for getting binary data back
//  (caller interprets the bytes)
//
template <typename scalar_tt>
void curlGetMatrix(CURL* easyHandle, const std::string& url, scalar_tt* data, size_t nRow, size_t nCol)
{
    size_t tSize = sizeof(scalar_tt);

    bool cerrDebug=false;
    if(cerrDebug) {
        std::cerr << "[cerrDebug] curlGetMatrix, URL is: " << url << std::endl; 
        std::cerr << "[cerrDebug] curlGetMatrix, nRow: " << nRow << std::endl; 
        std::cerr << "[cerrDebug] curlGetMatrix, nCol: " << nCol << std::endl; 
        std::cerr << "[cerrDebug] curlGetMatrix, tSize: " << tSize << std::endl; 
    }

    MatrixShimAdapter recvMatrix(nRow,  nCol, BlkRows, BlkCols, (void*)data, tSize);

    // TODO: how to handle the case where more is received that the length of data?
    //       throwing an exception might be a problem

    // receive anything it sends
    curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, CBRecvData);
    curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, &recvMatrix);
    curl_easy_setopt(easyHandle, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(easyHandle, CURLOPT_URL, url.c_str());

    CURLcode code = curl_easy_perform(easyHandle);
    if(code != CURLE_OK) {
        std::cerr << "[cerrDebug] curlGetMatrix, perform fail code" << code << std::endl; 
        throw std::runtime_error("ERROR curlGetMatrix: perform failed");
    }

    if(!recvMatrix.isComplete()) {
        std::cerr << "error: recvMatrix not complete nRow " << nRow
                  << " nCol " << nCol << " tSize " << tSize
                  << " bytesSoFar " << recvMatrix.bytesSoFar() << std::endl;
        throw std::runtime_error("ERROR curlGetMatrix: recvMatrix incomplete");
    }

    // TODO: possible that whitespace trimming should be done here,
    // TODO: possbile that whitespace trimming should be option-controlled
}

//
// above here, only knowledge is that the protocol is HTTP-based
// below here, we are aware that we are connecting to SciDB via shim protocol
//

//
// run a SciDB query via a particular shim session
//
// TODO: becomes a method of Shim
// TODO: return void
//
std::string executeQuery(Shim& shim,
                         const std::string& nickName,
                         const std::string& query,
                         const std::string& binaryFormat=std::string())
{
    // need a new session for a new query
    // except between an upload(not a query) and
    // the query that uses it
    // so maybe we only need new sessions just before uploads
    
    // paste together the base URL stuff with the URL-escaped query
    std::string url = shim.baseURL + "/execute_query?id=" + shim.session
                                   + "&query=" + curl_easy_escape(shim.curlHandle, query.c_str(), 0);
    if (binaryFormat.size()) {
        url += "&save=" ;
        url += "(" ;
        url += binaryFormat;    // e.g. "float" "double"
        url += ")" ;
    }

    double start=getsecs();
    std::string qid = curlGetString(shim.curlHandle, url);
    double secs = getsecs() - start;
    if(shim.verbose) {
        shim.tsos << "[v]                     executeQuery: " << nickName << ": QID: " << qid << std::endl;
    }
    if(secs > shim.timing) {
        shim.tsos << "[t] " << secs << " .... executeQuery: " << nickName << std::endl;
    }
    return qid;
}

//
// newSession, needed after[before?] file uploads, maybe other places
//
void new_session(Shim& shim)
{
    if(shim.verbose) {
        shim.tsos << "new_session: releasing " << shim.session << std::endl;
    }
    std::string URL = shim.baseURL + "/release_session?id=" + shim.session ;
    curlGetString(shim.curlHandle, URL, false); // nothing returned on a release
    shim.session  = curlGetString(shim.curlHandle, shim.baseURL + "/new_session");
    if(shim.verbose) {
        shim.tsos << "new_session: acquired " << shim.session << std::endl;
    }
}

//
// TODO: becomes a method of Shim
//
template<typename scalar_tt>
void readResultMatrix(Shim& shim, std::string nickName,
                      scalar_tt* array, size_t nRow, size_t nCol)
{
    // paste together the query and the base URL stuff
    std::stringstream ssURL;
    ssURL << shim.baseURL << "/read_bytes?id=" << shim.session << "&n=" << nRow*nCol*sizeof(scalar_tt);

    double start=getsecs();
    curlGetMatrix(shim.curlHandle, ssURL.str(), array, nRow, nCol);
    double secs = getsecs() - start;
    if(shim.verbose) {
        shim.tsos << "[v]                     readResultMatrix: " << nickName << std::endl;
    }
    if(secs > shim.timing) {
        shim.tsos << "[t] " << secs << " .... readResultMatrix: " << nickName << std::endl;
    }

    new_session(shim); // after invoking a curl read, we need a new session
    // preventative for a hanging read of a query
    // when beta is non-zero (a C matrix sent)
    // or maybe this should be before the scan query?
}



//
// scan_matrix
//
void scan_matrix(Shim& shim, const std::string& nameA,
                 const std::string& scalarName, size_t reshapeRows, size_t reshapeCols)
{
    // scalarName is "float" or "double"
    if (scalarName != std::string("float") &&
        scalarName != std::string("double")) {
        throw std::runtime_error("gemm: scalarName 'float' or 'double' required");
    }

    //
    // save and download?
    //
    std::stringstream query;

    if(DoReshape) {
        // reshape to a a vector, this will serialize the data in row-major order
        // and the client will not have to relocate chunks by position
        query << "reshape(";
    }
    
    const bool scidbGemmDoubleOnly = getenv("SCIDB_GEMM_DOUBLE_ONLY");
    if(scidbGemmDoubleOnly && scalarName == std::string("float")) {
        // expecting flaot, but we have only double
        query << "project(apply(" << nameA << ",vDbl,double(v)), vDbl)" ;
    } else {
        query << "scan(" << nameA << ")";
    }
    
    if(DoReshape) {
        // close the reshape expression
        query << ", <v:" << scalarName << ">[i=0:"<<reshapeRows*reshapeCols<<"-1,1000,0])";
    }
                                                  
    if(shim.verbose) {
        std::cerr << "[v]               scan_matrix query is '" << query.str() << "'" << std::endl; 
        std::cerr << "[v]               scan_matrix scalarName is: " << scalarName << std::endl; 
    }
    std::string qid = executeQuery(shim, "scan matrix", query.str(), scalarName);
    // no new_session() here ever ... the query result is not read yet!
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
        shim.tsos << "[v]                 create_temp_matrix: createQuery is '" << createQuery << "'" << std::endl; 
    }
    std::string qid = executeQuery(shim, "create_temp_array", createQuery);
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
// NOTE: shim may get a new session as a side-effect
// result is an expression or name of array containing the result,
// depending on the setting of shim.lazyEval
//
template<typename scalar_tt>
std::string send_matrix(Shim& shim, const scalar_tt* data, size_t nrow, size_t ncol, const std::string& resultName)
{
    std::string result;

    // parameters

    if(shim.verbose) { shim.tsos << "[v]              send_matrix: entered" << std::endl; }

    const char* scalarName=typeStr(data[0]);
    size_t      dataBytes = nrow*ncol*sizeof(scalar_tt);

    // hack, until MatrixShimAdapater split into sender and receiver
    scalar_tt* s_data = const_cast<scalar_tt*>(data);
    char*      c_data = reinterpret_cast<char*>(s_data);
    MatrixShimAdapter cbMatrix(nrow,  ncol, BlkRows, BlkCols, c_data, sizeof(scalar_tt));

    if(shim.verbose) { shim.tsos << "[v]               send_matrix: nrow " << nrow << " ncol " << ncol << std::endl; }

    bool doDebugSeq=false;
    if(doDebugSeq) {
        // debug pre-numbering: overwrite data with something easy to track
        float* fltData= reinterpret_cast<float*>(c_data);
        for(size_t i=0; i<dataBytes/sizeof(float); i++) {
            fltData[i] = float(i);
        }
    }

    //
    // PART A: create the temp file
    // can't avoid the temp array until the shim session can be
    // preserved across file uploads to run the final gemm
    //
    create_temp_matrix(shim, nrow, ncol, resultName, scalarName);

    //
    // PART B: upload the file
    //
    new_session(shim); // before an upload we NEED a fresh session
    if(shim.verbose) { shim.tsos << "[v]              send_matrix: ~~~~~~~POST~~~~~~~~" << std::endl; }
    const size_t MAX_UPLOAD_ATTEMPTS=3; // sometimes upload times out or gets error

    double uploadSecs=0;

    std::string filePath;
    for(size_t attempts=1; attempts < MAX_UPLOAD_ATTEMPTS; attempts++) {
        // TODO: get URL from shim
        char uploadURL[1000];
        sprintf(uploadURL, "%s/upload_file?id=%s", shim.baseURL.c_str(), shim.session.c_str());
        if(shim.verbose) {
            shim.tsos << "[v]                         send_matrix: uploadURL is '" << uploadURL << "'" << std::endl; 
            shim.tsos << "[v]                         send_matrix: form _CONTENTS_LENGTH=" << dataBytes << "'" << std::endl;
        }

        // curl POST, not this way because need multipart, custom headers for binary file
        // curl_easy_setopt(shim.curlHandle, CURLOPT_POST, 1L);
        // curl_easy_setopt(shim.curlHandle, CURLOPT_POSTFIELDS, NULL);
        // curl_easy_setopt(shim.curlHandle, CURLOPT_POSTFIELDSIZE_LARGE, dataBytes);
        // curl_easy_setopt(shim.curlHandle, CURLOPT_READFUNCTION, CBSendData); // callback
        // curl_easy_setopt(shim.curlHandle, CURLOPT_READDATA, &cbMatrix); // or @CFM_STREAM?
        // headers=curl_slist_append(headers, "Content-Disposition: form-data; name=\"fileupload\"; filename=\"data\"");

        // disable the "Expect: 100-continue", it causes 1-sec timeout delays with shim's mongoose web server
        struct curl_slist* headers=NULL;
        headers= curl_slist_append(headers, "Expect:");
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPHEADER, headers);

        // manually add binary data header
        // struct curl_slist* formHeaders=NULL;
        // formHeaders= curl_slist_append(formHeaders, "Content-Type: application/octet-stream");

#define CURLFORM_STREAM_WORKAROUND 1
#if CURLFORM_STREAM_WORKAROUND
        // note: the CURLFORM_STREAM stuff does not succeed in getting file data
        // to shim, but the BUFFPERPTR, BUFFERLENGTH does
        // so as a work-around, we ourselves call the cbMatrix to re-order
        // the ENTIRE matrix into a tmp buffer, and then send that with BUFFERPTR.
        shim.tsos << "warning: matrix upload workaround active" << std::endl;
        std::vector<char> chunkedData(dataBytes,0);
        cbMatrix.sendData(&chunkedData[0], dataBytes);
#endif

        struct curl_httppost* post=NULL;
        struct curl_httppost* last=NULL;
        CURLFORMcode formcode = curl_formadd(&post, &last,
                                             CURLFORM_COPYNAME,"fileupload",
                                             CURLFORM_BUFFER,  "data",  // if 1 buffer
#if CURLFORM_STREAM_WORKAROUND
                                             CURLFORM_BUFFERPTR, &chunkedData[0],
                                             CURLFORM_BUFFERLENGTH, dataBytes,
#else
                                             CURLFORM_STREAM, &cbMatrix, // CBSendData 4th arg
                                             // also set CURLOPT_READFUNCTIION
                                             CURLFORM_CONTENTSLENGTH, dataBytes
#endif
                                             CURLFORM_CONTENTTYPE, "binary",    // poorly documented!
                                             //CURLFORM_CONTENTHEADER, formHeaders,
                                             CURLFORM_END);
        if(formcode != CURL_FORMADD_OK) {
            shim.tsos << "ERROR send_matrix: curl_formadd failed with code: " << formcode << std::endl;
            throw std::runtime_error("send_matrix: curl_formadd failed");
        }

        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPPOST, post); // want to POST here, then GET the filename back
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEDATA, &filePath);          // to get filePath back
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEFUNCTION, receiveString);  // ditto
        curl_easy_setopt(shim.curlHandle, CURLOPT_URL, uploadURL);

        // run the page
        double start = getsecs();
        CURLcode code = curl_easy_perform(shim.curlHandle);
        uploadSecs+= getsecs()-start;
        if(shim.verbose) { shim.tsos << "[v]                    send_matrix: shim filepath '" << filePath << "'" << std::endl; }

        //
        // cleanups in reverse order of their use, to make it easier to track
        //
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(shim.curlHandle, CURLOPT_WRITEDATA, NULL);
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPPOST, NULL);
        curl_formfree(post);
        curl_easy_setopt(shim.curlHandle, CURLOPT_HTTPHEADER, NULL);    // not clearing this makes severe trouble later
        curl_slist_free_all(headers);
        curl_easy_setopt(shim.curlHandle, CURLOPT_READFUNCTION, NULL);


        // success yet?
        if(code != CURLE_OK) {
            // getting these with shim server every once in a while
            if (attempts == MAX_UPLOAD_ATTEMPTS) {
                throw std::runtime_error("send_matrix: upload_file failed");
            }
            size_t retrySecs = 1 << (attempts-1);  // exponential back-off
            shim.tsos << "send_matrix: upload_file failed, retrying in " << retrySecs << "secs" << std::endl;
            sleep(retrySecs);
            continue;
        }

        if(filePath.size() == 0) {      // happens with session problems or other failures
            throw std::runtime_error("send_matrix: upload_file no remote filename returned");
        }

        if(shim.verbose) { shim.tsos << "[v]            send_matrix: upload " << filePath << " success" << std::endl; }

        if(uploadSecs > shim.timing) {
            shim.tsos << "[t] " << uploadSecs << " .... send_matrix: upload " << filePath << std::endl;
        }
        break; // don't retry
    }

    //
    // PART C: load the array
    //
    // TODO: test for a collision with an existing array name?
    //       right now it is the callers responsibility
 
    char loadQuery[1000];
    sprintf(loadQuery,
            "store(input(<v:%s>[i=0:%ld-1,1000,0,c=0:%ld-1,1000,0],'%s',-2,'(%s)'),%s)",
            scalarName, nrow, ncol, filePath.c_str(), scalarName,
            resultName.c_str());
    if(shim.verbose) { shim.tsos << "[v]               send_matrix: loadQuery is '" << loadQuery << "'" << std::endl; }

    if(true || !shim.lazyEval) {
        // add store, return array's name
        result = resultName;
        std::string qid = executeQuery(shim, "send_matrix store(input())", loadQuery);
    
        if(shim.check) { // check the info in the resulting array
            scan_matrix(shim, resultName, scalarName, nrow, ncol);
            std::vector<scalar_tt> checkData(nrow*ncol,0);
            if(shim.verbose) { shim.tsos << "[v]               send_matrix: reading back for check'" << std::endl; }
            readResultMatrix<scalar_tt>(shim, "check uploaded matrix", &(checkData[0]), nrow, ncol);
            if(memcmp(data, &(checkData[0]), dataBytes)) {
                shim.tsos << "ERROR send_matrix: round-trip check fails" << std::endl;
                for(size_t i =0; i < nrow*ncol; i++) {
                    if (data[i] != checkData[i]) {
                        shim.tsos << "ERROR val["<<i<<"]=" << data[i] << " returned as " << checkData[i] << std::endl;
                    }
                }
            }
        }
    } else {
        sprintf(loadQuery, "input(<v:%s>[i=0:%ld-1,1000*1000,0],'%s',-2,'(%s)')",
                            scalarName, nrow * ncol, filePath.c_str(), scalarName);
        result = loadQuery; // use the query, not the array name
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
                           const std::string& scalarName, size_t cRow, size_t cCol)
{
    //
    // start the query
    //
    std::string exprA = nameA;
    std::string exprB = nameB;
    std::string exprC = nameC;
    const bool scidbGemmDoubleOnly = getenv("SCIDB_GEMM_DOUBLE_ONLY");
    if (scidbGemmDoubleOnly && scalarName == std::string("float")) {
        // there is no gemm for float.  if scalarName == "float"
        // then we need to use project(apply()) to increase array to double precision.
        // and we need to do the reverse at the end
        // NOTE: we could have a fgemm macro that would do this
        exprA = "project(apply(" + nameA + ",vDbl,double(v)), vDbl)" ;
        exprB = "project(apply(" + nameB + ",vDbl,double(v)), vDbl)" ;
        exprC = "project(apply(" + nameC + ",vDbl,double(v)), vDbl)" ;
    }

    //
    // gemm portion of the query
    //
    std::stringstream gemmQuery;
    if (scidbGemmDoubleOnly && scalarName == std::string("float")) {
        // have to convert to double with an apply :(
        gemmQuery << "project(apply(" ;
    }

    gemmQuery << "gemm(" << exprA << "," << exprB << "," << exprC ;
    gemmQuery << ",'TRANSA=" << long(transA) << ";TRANSB=" << long(transB); 
    gemmQuery << ";ALPHA=" << alpha << ";BETA=" << beta << "')";

    if (scidbGemmDoubleOnly && scalarName == std::string("float")) {
        // have to close the apply
        gemmQuery << ", vFlt, float(gemm)), vFlt)";
    }

    if(shim.verbose) {
        std::cerr << "[v] queryGemm: gemmQuery is '" << gemmQuery.str() << "'" << std::endl; 
    }
    // as root of expression tree currenly,
    // gemm's don't lazyEval
    std::string qid = executeQuery(shim, "queryGemm", gemmQuery.str(), scalarName);
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

        // sketch of authorization for use with https (unverified).
        // e.g. use
        // https://localhost:8083/login?username=root&password=Paradigm4
        // to get an authentication token of root:Paradigm4, and then 
        // add [?&]auth=root:Paradigm4 at the end of each query
        // (this latter option would need to be implemented throughout)
        //
        if(false) {
            curl_easy_setopt(easyHandle, CURLOPT_URL, baseURL.c_str());
            curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, CBRecvData);
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
        // get a session
        //
        std::string fullURL = baseURL + std::string("/new_session");
        std::string session;
        // TODO TODO switch to curlGetString
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
            session  = curlGetString(easyHandle, fullURL.c_str());
        }

        // TODO: check the session and if not obtained, throw a std::runtime_error()

        //
        // NOTE: we define the setting of the handle that the shim
        // is initialized.
        // The caller should not have to check the initial session.
        //
        cachedShim = new Shim(easyHandle, baseURL, session, std::cerr);
        if(cachedShim->verbose) {
            cachedShim->tsos << "[v]                getsShim: created new Shim(,\"" << baseURL << " ," << session << ")" << std::endl;
        }
    
        // load dense linear algebra for dgemm
        executeQuery(*cachedShim, "getShim load_library dense_linear_algebra",
                     "load_library('dense_linear_algebra')");
    }

    return *cachedShim;
}

//
// helper for gemmScidbServer
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
void gemmScidbServer(const char& TRANSA, const char& TRANSB,
                     long M, long N, long K,
                     scalar_tt ALPHA, const scalar_tt* aData, long LDA,
                                      const scalar_tt* bData, long LDB,
                     scalar_tt BETA,        scalar_tt* cData, long LDC,
                     Shim& shim)
{
    const char* scalarName=typeStr(aData[0]);

    if(shim.verbose) {
        shim.tsos << "TIMING gemmScidbServer: Start" << std::endl;
        shim.tsos << "TIMING gemmScidbServer: TRANSA:" << TRANSA << " TRANSB: " << TRANSB << std::endl; 
        shim.tsos << "TIMING gemmScidbServer: ALPHA: " << ALPHA  << " BETA:   " << BETA << std::endl; 
        shim.tsos << "TIMING gemmScidbServer: M:     " << M      << " N:      " << N    << " K:      " << K << std::endl; 
        shim.tsos << "TIMING gemmScidbServer: LDA:   " << LDA    << " LDB:    " << LDB  << " LDC:    " << LDC << std::endl; 
    }

    // load dense linear algebra for dgemm
    std::string qid = executeQuery(shim, "gemmScidbServer load_library", "load_library('dense_linear_algebra')");
    //new_session(shim);

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

    //
    // TODO: generate unique temporary names, e.g. GUID-based
    //

    double start = getsecs();
    std::string aName = send_matrix(shim, aData, aRow, aCol, "TMPA");
    double secs = getsecs() - start;
    if(std::isfinite(shim.timing) || shim.verbose) {
        shim.tsos << "[t] "<< secs << " =SUBTOTAL gemmScidbServer send_matrix A: " <<aRow<< " x " <<aCol<< std::endl;
    }

    start = getsecs();
    std::string bName = send_matrix(shim, bData, bRow, bCol, "TMPB");
    secs = getsecs() - start;
    if(std::isfinite(shim.timing) || shim.verbose) {
        shim.tsos << "[t] "<< secs << " =SUBTOTAL gemmScidbServer send_matrix B: " <<bRow<< " x " <<bCol<<  std::endl;
    }

    std::string cName;
    if(BETA==0.0) {
        // TODO, make analogue that does not send any data
        double start = getsecs();
        cName = create_temp_matrix(shim, cRow, cCol, "TMPC", scalarName);
        double secs = getsecs() - start;
        if(std::isfinite(shim.timing) || shim.verbose) {
            shim.tsos << "[t] " << secs << " =SUBTOTAL gemmScidbServer create empty C: " <<cRow<< " x " <<cCol << std::endl;
        }
    } else {
        double start = getsecs();
        cName = send_matrix(shim, cData, cRow, cCol, "TMPC");
        double secs = getsecs() - start;
        if(std::isfinite(shim.timing) || shim.verbose) {
            shim.tsos << "[t] " << secs << " =SUBTOTAL gemmScidbServer send_matrix C: " <<cRow<< " x " <<cCol<< std::endl;
        }
    }

    // run the query
    {
        bool debugUsingShow=false;
        if(debugUsingShow) {
            scan_matrix(shim, aName, scalarName, cRow, cCol);
        } else {
            queryGemm(shim, aName, bName, cName, transA, transB, /*alpha*/ALPHA, /*beta*/BETA, scalarName, cRow, cCol);
            // answer should be
            // [22.5 49.5]
            // [28.5 64.5]
        }

        //and read the output back into the "C" array
        if(shim.verbose) { shim.tsos << "[v]               gemmScidbServer: reading result C'" << std::endl; }
        readResultMatrix<scalar_tt>(shim, "read result 'C' array", cData, cRow, cCol); 
    }

    // and remove the arrays
    {
        std::string qidA = executeQuery(shim, "gemmScidbServer remove A", "remove(TMPA)");
        std::string qidB = executeQuery(shim, "gemmScidbServer remove B", "remove(TMPB)");
        std::string qidC = executeQuery(shim, "gemmScidbServer remove C", "remove(TMPC)");
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
        shim.tsos << "unit_test main calling send_matrix(,aData," << aRow << " x " << aCol << ",)" << std::endl;
        std::string aName = send_matrix<scalar_tt>(shim, aData, aRow, aCol, "UNITA");

        shim.tsos << "unit_test main calling send_matrix(,bData," << bRow << " x " << bCol << ",)" << std::endl;
        std::string bName = send_matrix<scalar_tt>(shim, bData, bRow, bCol, "UNITB");

        shim.tsos << "unit_test main calling send_matrix(,bData," << cRow << " x " << cCol << ",)" << std::endl;
        std::string cName = send_matrix<scalar_tt>(shim, cData, cRow, cCol, "UNITC");

        // need 1 to 3 arrays
        // TODO: when send_matrix done, change this to 3 different arrays
        bool transA=false;
        bool transB=false;
        // run query outputting in binary scalar_tt that can be retrieved with /read_bytes
        bool debugWithShow=false;
        if(debugWithShow) {
            scan_matrix(shim, aName, scalarName, cRow, cCol);
        } else {
            queryGemm(shim, aName, bName, cName, transA, transB, /*alpha*/1.0, /*beta*/0.5, scalarName, cRow, cCol);
            // answer should be
            // [22.5 49.5]
            // [28.5 64.5]
        }

        //and read the output back into an array D
        const size_t dRow = cRow;
        const size_t dCol = cCol;
        double dData[dRow*dCol];
        readResultMatrix<scalar_tt>(shim, "unit tests", dData, dRow, dCol); 
        for(size_t i=0; i< dRow*dCol; i++) {
            shim.tsos << "unit_test dgemm result["<<i<<"] = " << dData[i] << std::endl;
        }

        // TODO:
        // REMOVE UNITA, UNITB, UNITC
    }

    shim.tsos << "main: calling gemmScidbServer" << std::endl;

    gemmScidbServer('N', 'N', aRow, bCol, bRow,
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

    gemmScidbServer('N', 'N', aRow, bCol, bRow,
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
bool significantDifference(scalar_tt cData[], scalar_tt cResultBlas[], size_t numVals)
{
    for(size_t i =0; i < numVals; i++) {
        if (abs(cData[i] - cResultBlas[i]) > 1e-10 ) { // TODO: fix this to be relative error
            return true;
        }
    }
    return false;
}

template<typename scalar_tt>
void dumpError(scidb::TimeStampedStream& tsos, const scalar_tt* data, size_t nRow, size_t nCol, const std::string& label)
{
    tsos << "ERROR caffe_scidb_gemm: " << label << " nRow " << nRow << " nCol " << nCol << std::endl;
    
    for(size_t i=0; i < nRow*nCol; i++) {
        tsos << "ERROR caffe_scidb_gemm: " << "[" << i << "]= " << data[i] << std::endl;
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
                      const long M, const long N, const long K,
                      const scalar_tt alpha, const scalar_tt* aData, const long& lda,
                                             const scalar_tt* bData, const long& ldb, const scalar_tt beta,
                                                   scalar_tt* cData, const long& ldc)
{

    //
    // get connection to SciDB
    // if its down, should we use the cblas?
    //
    scidb::Shim& shim = scidb::getShim();     // with active session

    //
    // set verbosity
    //
    shim.verbose=bool(getenv("SCIDB_SHIM_TRACE"));       // traces the major steps
    if(shim.verbose) {
        shim.tsos << "caffe_scidb_gemm: entered type: " << scidb::typeStr(aData[0]) << std::endl;
    }

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

    const char* doTimingStr = getenv("SCIDB_SHIM_TIME");
    if (doTimingStr) {
        shim.timing = atof(doTimingStr);
    }

    shim.lazyEval = true;               // return queries, not array names

    static bool dotsNeedNewline=false;

    // env SCIDB_SHIM_URL was not set
    // or if the matrix is small enough
    if(shim.baseURL.size()==0 || M*N*K <= localLimit) {
        // local will be faster
        double start = scidb::getsecs();
        cblas_gemm(CblasRowMajor,
                   TransA, TransB, M, N, K,
                    alpha, aData, lda,
                           bData, ldb,
                    beta,  cData, N);
        double secs = scidb::getsecs() - start;
        if (std::isfinite(shim.timing)) {
            if(secs >= shim.timing) {
                if(dotsNeedNewline) {
                    std::cerr << std::endl;
                    dotsNeedNewline=true;
                }
                shim.tsos << "[t] " << secs << " ==TOTAL"
                          << " caffe_scidb_gemm: cblas: " << scidb::typeStr(aData[0])
                          << " " << M << "*" << K << "*" << N
                          << " beta " << beta
                          << ", " << 1e-6*M*K*N/(secs) << " MFLOP/s" << std::endl; 
            } else {
                std::cerr << "c" ;
                dotsNeedNewline=true;
            }
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
    // using a big stack array causes debugging problems
    // (they may not expect frame offsets to be larger than 16 or 32 bit?
    // so as a workaround, we new the array whenever it is smaller
    // than the M*N requested.  We keep it in a static
    // since its goign to be used over and over.
    // note we never bother to free it... this is for debuggging only
    static size_t cDataSize=0;
    static scalar_tt* cOriginal = NULL; // original C
    static scalar_tt* cResultBlas = NULL; // C' computed by BLAS as a check against SciDB

    if(shim.check && (M*N > cDataSize)) {
        if(cOriginal) { delete[] cOriginal; }
        if(cResultBlas) { delete[] cResultBlas; }
        cDataSize=M*N;
        cOriginal = new scalar_tt[cDataSize];
        cResultBlas = new scalar_tt[cDataSize];
        shim.tsos << "caffe_scidb_gemm: cDataSize increased to " << cDataSize <<std::endl;
    }
    if (shim.check) {
        assert(M*N <= cDataSize);
        assert(cOriginal);
        assert(cResultBlas);
        // save a copy of cData for the check calculation
        memcpy(cOriginal, cData, M*N*sizeof(cOriginal[0]));
    }

    //
    // TODO: optimize away the sending of the C array
    //       when it is completely zero
    //
    bool timingPrinted=false;
    {
        double start = scidb::getsecs();
        // without tranpositions, A is MxK, B is KxN, and C is MxN (row major)
        // not yet clear what lda, ldb, ldc should be
        // long lda = (TransA == CblasNoTrans) ? M : K;  // correct order?
        // long ldb = (TransB == CblasNoTrans) ? K : N;
        // long ldc = N;
        scidb::gemmScidbServer(charFromCblasTrans(TransA), charFromCblasTrans(TransB),
                               M, N, K,
                               alpha, aData, lda /*M?*/, 
                                      bData, ldb /*K?*/,
                               beta,  cData, ldc /*N?*/,
                               shim);
        double secs = scidb::getsecs() - start;
        if(std::isfinite(shim.timing) || shim.check) {
            if(secs >= shim.timing) {
                if(dotsNeedNewline) {
                    std::cerr << std::endl;
                    dotsNeedNewline=true;
                }
                shim.tsos << "[t] " << secs << " ==TOTAL"
                          << " caffe_scidb_gemm: scidb: " << scidb::typeStr(aData[0])
                          << " " << M << "*" << K << "*" << N
                          << " beta " << beta
                          << ", " << 1e-6*M*K*N/(secs) << " MFLOP/s" << std::endl; 
                timingPrinted=true;
            } else {
                std::cerr << "s" ;
                dotsNeedNewline=true;
            }
        }
    }
            
    if (shim.check) { // TODO: see below should this policy be in here or where called from caffe_{cpu,gpu}_gemm
        if(shim.verbose) {
            shim.tsos << "caffe_scidb_gemm: checking" << std::endl;
        }
        // TODO: decide about fall-back
        // fall back to clbas_dgemm?  or gpu?
        // or give an error?
        // or should caffe_cpu_gemm() and caffe_gpu_gemm()
        // both try scidb_dgemm() first and fall back
        // if an exception is raised?
        //long lda = (TransA == CblasNoTrans) ? K : M; // reverse
        //long ldb = (TransB == CblasNoTrans) ? N : K; // reverse
        memcpy(cResultBlas, cOriginal, M*N*sizeof(scalar_tt));

        double start = scidb::getsecs();
        cblas_gemm(CblasRowMajor,
                   TransA, TransB, M, N, K,
                   alpha, aData, lda,
                          bData, ldb,
                   beta,  cResultBlas, N);
        double secs = scidb::getsecs() - start;
        if(timingPrinted) {
            shim.tsos << "[t] " << secs << " ==TOTAL"
                      << " caffe_scidb_gemm: cblas check: " << scidb::typeStr(aData[0])
                      << " MKN: " << M << "*" << K << "*" << N
                      << " beta " << beta << ", "
                      << 1e-6*M*K*N/(secs) << " MFLOP/s" << std::endl; 
        }

        // TODO now compare original cData with cDataCheck

        if(significantDifference(cData, cResultBlas, M*N)) {
            // they differ
            shim.tsos << "ERROR caffe_scidb_gemm: ------------------------------" << std::endl;
            shim.tsos << "ERROR caffe_scidb_gemm: not same result as cblas_dgemm"    << std::endl;
            shim.tsos << "ERROR caffe_scidb_gemm:                                          " << std::endl;
            shim.tsos << "ERROR caffe_scidb_gemm: alpha  " << alpha  << " beta   " << beta   << std::endl; 
            shim.tsos << "ERROR caffe_scidb_gemm: TransA " << TransA << " TransB " << TransB << std::endl; 
            shim.tsos << "ERROR caffe_scidb_gemm: M      " << M      << " N      " << N      << " K      " << K << std::endl; 
            shim.tsos << "ERROR caffe_scidb_gemm: lda    " << lda    << " ldb    " << ldb    << " ldc    " << ldc << std::endl; 
            shim.tsos << "ERROR caffe_scidb_gemm:                                          " << std::endl;

            dumpError(shim.tsos, aData, M, K, "aData");
            shim.tsos << "ERROR caffe_scidb_gemm:                                          " << std::endl;

            dumpError(shim.tsos, bData, K, N, "bData");
            shim.tsos << "ERROR caffe_scidb_gemm:                                          " << std::endl;

            dumpError(shim.tsos, cOriginal, M, N, "cOriginal -- original");
            shim.tsos << "ERROR caffe_scidb_gemm:                                          " << std::endl;

            for(size_t i =0; i < M*N; i++) {
                if (abs(cData[i] - cResultBlas[i]) > 1e-10 ) {
                    shim.tsos << "ERROR cData["<<i<<"]=" << cData[i] << " != cResultBlas["<<i<<"]=" << cResultBlas[i] << std::endl;
                }
            }
        } else {
            if(shim.verbose) {
                shim.tsos << "caffe_scidb_gemm: check passed." << std::endl;
            }
        }
    } 
}

// force instantiation since template not exposed to caller
template
void caffe_scidb_gemm<float >(const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                              const long M, const long N, const long K,
                              const float  alpha, const float * aData, const long& lda,
                                                  const float * bData, const long& ldb, const float  beta,
                                                        float * cData, const long& ldc);
template
void caffe_scidb_gemm<double>(const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                              const long M, const long N, const long K,
                              const double alpha, const double* aData, const long& lda,
                                                  const double* bData, const long& ldb, const double beta,
                                                        double* cData, const long& ldc);


}  // namespace caffe
