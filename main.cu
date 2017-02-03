// Defs
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

using namespace std;
using namespace glm;

#define MIN_WIDTH 1
#define MAX_WIDTH 65535
#define MIN_HEIGHT 1
#define MAX_HEIGHT 65535
#define MIN_SAMPLES 1
#define MAX_SAMPLES 65535

struct PixelBuffer{
	glm::vec3 color;
	unsigned int samples;

	__host__ __device__ PixelBuffer(){
		color = glm::vec3(0.0f, 0.0f, 0.0f);
		samples = 0;
	}

	__host__ __device__  glm::vec3 getAverage(){
		// return result
		if(samples != 0){
			return color / (float)samples;
		}else{
			return color;
		}
	}

};
__device__ void render(PixelBuffer *pixel){
	// return a red pixel
	pixel->color = glm::vec3(1.0f, 0.0f, 0.0f);
	pixel->samples = 0;
}

__global__ void KERNEL_RENDER(PixelBuffer *buffer, int w, int h){

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= w * h) return;

	// render
	render(&buffer[threadId]);
}

unsigned char floatToShort(float f){
	return (unsigned char)(f*255.0f);
}

void saveBuffer(PixelBuffer* data, unsigned int w, unsigned int h, std::string filename){

	if(data == NULL){
		cout << " Invalid data... no file saved" << "\n";
		return;
	}

	ofstream outputFile;
	outputFile.open(filename, ofstream::out | ofstream::binary);

	// PPM header, P6 = binary
	outputFile << "P6\n" << w << " " << h << "\n" << "255" << "\n";

	unsigned char r, g, b;

	// loop over each pixel
	long long i;
	long long j;

	// reverse image write
	j = (w*h);
	for (i=0; i < j; i++) {
		PixelBuffer a;
		a = data[i];

		glm::vec3 t = a.getAverage();

		r = floatToShort(t.x);
		g = floatToShort(t.y);
		b = floatToShort(t.z);
		outputFile << r << g << b;
	}

	// Close the file
	outputFile.close();

}

void checkCudaError(cudaError_t e, unsigned short *c){
	if(e != cudaSuccess){
		const char *errorString = cudaGetErrorString(e);
		cout << "ERROR: "<< errorString << std::endl;
		*c += 1;
	}
}

// main
int main(int argc, const char * argv[]){

	// error flag
	cudaError_t error;
	unsigned short errorcount = 0;

	// the "samples" variable is not used in this simplified example
	int width, height, samples;

	// Pointers to buffers
	PixelBuffer *host_pixels;
	PixelBuffer *device_pixels;

	if(argc == 4){
		width = atoi(argv[1]);
		height = atoi(argv[2]);
		samples = atoi(argv[3]);

		if (width < MIN_WIDTH) width = MIN_WIDTH;
		if (width > MAX_WIDTH) width = MAX_WIDTH;

		if (height < MIN_HEIGHT) height = MIN_HEIGHT;
		if (height > MAX_HEIGHT) height = MAX_HEIGHT;

		if (samples < MIN_SAMPLES) height = MIN_SAMPLES;
		if (samples > MAX_SAMPLES) height = MAX_SAMPLES;
	}else{
		cout << "Use: ./mt -width -height -samples" << std::endl;
		return 0;
	}

	size_t buffersize = sizeof(PixelBuffer)*width*height;
	cout << "Pixel buffer size: " << buffersize / (1024 * 1024) << " Mb" << std::endl;

	// Allocate the pixel buffer on the host
	error = cudaHostAlloc((void**)&host_pixels, buffersize, cudaHostAllocWriteCombined);
	checkCudaError(error, &errorcount);

	// Allocate the pixel buffer on the device
	error = cudaMalloc((void**)&device_pixels, buffersize);
	checkCudaError(error, &errorcount);

	// Do some stuff
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
	KERNEL_RENDER <<<gridSize, blockSize>>>(device_pixels, width, height);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	checkCudaError(error, &errorcount);

	// Copy pixel buffer device to host, error checking omitted for brevity
	error = cudaMemcpy(host_pixels, device_pixels, buffersize, cudaMemcpyDeviceToHost);
	checkCudaError(error, &errorcount);

	// Write the output file
	cout << "Number of errors: " << errorcount << std::endl;
	if(errorcount == 0){
		saveBuffer(host_pixels, width, height, "out.ppm");
		cout << "Output saved..." << std::endl;
	}

	// Delete device buffer
	if( device_pixels != NULL ){
		cudaFree(device_pixels);
	}

	// Delete host buffer
	if( host_pixels != NULL ){
		cudaFreeHost(host_pixels);
	}

	// Reset and exit
	cudaDeviceReset();
	return 0;
}
