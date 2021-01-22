#include "../zfp/include/zfp.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <random>

using namespace std; 
typedef std::chrono::high_resolution_clock Clock;

int size; 

int main(int argc, char* argv[]) {
	if(argc < 2)
		return -1; 
	size = atoi(argv[1]);
	
	void * array = 	malloc(sizeof(double)*size);
        void * decompressed = malloc(sizeof(double)*size);
        double * ptr = (double *)array;
	for(int i = 0; i < size; i++) {
		ptr[i] = std::rand();			
	}

	for(int i = 0; i < size; i++) {
		cout << ptr[i] << ",";
	}

	cout << endl;
	
        void * array2;
	int code = cudaMalloc((void**)&array2,sizeof(double) * size);
        cudaMemcpy(array2,
                       array,
                       sizeof(double)*size, cudaMemcpyHostToDevice);	
        cout << "Code: " << code << endl;
	std::cout << "Input size" << sizeof(double) * size << std::endl;
	// initialize metadata for the 3D array a[nz][ny][nx]
	zfp_type type = zfp_type_double;                          // array scalar type
	zfp_field* field = zfp_field_1d(array2, type, size); // array metadata
	cout << "Done 1" << endl;
	zfp_stream* zfp = zfp_stream_open(NULL);                  // compressed stream and parameters
	zfp->maxbits = 16; 	
	// initialize metadata for a compressed stream
	cout << zfp_stream_set_rate(zfp, 12, type, 1, 0) << endl;


	cout << "Done 2" << endl;
	// zfp_stream_set_accuracy(zfp, tolerance);                  // set tolerance for fixed-accuracy mode
	//  zfp_stream_set_precision(zfp, precision);             // alternative: fixed-precision mode
	//  zfp_stream_set_rate(zfp, rate, type, 3, 0);           // alternative: fixed-rate mode

	// allocate buffer for compressed data
	size_t bufsize = zfp_stream_maximum_size(zfp, field);     // capacity of compressed buffer (conservative)
	void* buffer; 
	cudaMalloc(&buffer, bufsize);                           // storage for compressed stream

	cout << "Done 3" << endl;
	// associate bit stream with allocated buffer
	bitstream* stream = stream_open(buffer, bufsize);         // bit stream to compress to
	zfp_stream_set_bit_stream(zfp, stream);                   // associate with compressed stream
	zfp_stream_rewind(zfp);                                   // rewind stream to beginning


	cout << "Done 4" << endl;
	if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
		// size_t zfpsize = 0;
		auto t1 = Clock::now();
		cout << "Done 5" << endl;
		size_t zfpsize = zfp_compress(zfp, field);                // return value is byte size of compressed stream
		auto t2 = Clock::now();
		cout << "Done 6" << endl;
		std::cout << zfpsize << " Final output size" << std::endl;
		std::cout << "Time taken to compress and decompress is: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns"  << std::endl;
                void* buffer2;
	        cudaMalloc(&buffer2, bufsize);
		cudaMemcpy(buffer2, buffer, zfpsize, cudaMemcpyDeviceToDevice);
	         /* associate bit stream with allocated buffer */
	        stream = stream_open(buffer, bufsize);
  		zfp_stream_set_bit_stream(zfp, stream);
  		zfp_stream_rewind(zfp);	
		
                if(zfp_decompress(zfp,field)) {
			cout << "Decompression done" << endl;
		} else {
			cout << "Decompression failed" << endl;
		}                
	} else {
		std::cout << "Nope" << std::endl;
	} 
        cudaMemcpy(decompressed,
                       array2,
                       sizeof(double)*size, cudaMemcpyDeviceToHost);	
        
	double * ptr2 = (double *) decompressed;
	
	for(int i = 0; i < size; i++) {
		if(ptr[i] - ptr2[i] > 0.00001) {
			cout << ptr[i] << " " << ptr2[i] << endl;
		}		
	}
	free(array);
        free(decompressed);
	cudaFree(array2);
	cudaFree(buffer);
}
