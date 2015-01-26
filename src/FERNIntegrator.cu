
#include "FERNIntegrator.cuh"
#include "Globals.cuh"
#include "kernels.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>


FERNIntegrator::FERNIntegrator()
{
	// TODO
	// Hardcoded for now. Make this the default for a command line
	// argument or something.
	
	blocks.x = 1;
	threads.x = 512;
}


FERNIntegrator::~FERNIntegrator()
{
	// TODO
	// Implement the freeing methods
	
	// devReactionData.cudaFree();
	// devNetworkData.cudaFree();
	
	// reactionData.free();
	// networkData.free();
}


void FERNIntegrator::initializeCuda()
{
	// Ensure that a valid device (GPU) exists
	printf("Checking for valid device...\n");
	printf("\n");
	devcheck(0);
	
	// Check available memory on the GPU
	size_t msizeFree;
	size_t msizeTotal;
	cudaMemGetInfo(&msizeFree, &msizeTotal);
	printf("GPU total memory: %d\n", (int)msizeTotal);
	printf("GPU free memory: %d\n", (int)msizeFree);
	
	// Following memory queries not supported on GF 8600 GT
	size_t msize;
	cudaDeviceGetLimit(&msize, cudaLimitMallocHeapSize);
	printf("GPU heap size: %d\n", (int)msize);
	cudaDeviceGetLimit(&msize, cudaLimitStackSize);
	printf("GPU stack size: %d\n", (int)msize);
	
	// Set and print the printf FIFO size
	
	size_t printfSize = 1 << 20; // 1 MiB
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfSize);
	cudaDeviceGetLimit(&printfSize, cudaLimitPrintfFifoSize);
	printf("printf FIFO size: %lu\n", printfSize);
	
	// Set the shared memory size
	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	printf("\n");
}


void FERNIntegrator::prepareKernel()
{
	printf("Preparing kernel...\n");
	
	// The network should be copied to the device
	// before the integration kernel is launched.
	// The memory blocks will last the lifetime of the FERNIntegrator.
	
	devNetwork.setSizes(network);
	devNetwork.cudaAllocate();
	checkCudaErrors();
	devNetwork.cudaCopy(network, cudaMemcpyHostToDevice);
	checkCudaErrors();
}


void FERNIntegrator::integrate(IntegrationData &integrationData)
{
	// Set up stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	// The IntegrationData should be copied to the device upon each
	// integration.
	
	printf("Copying integration data...\n");
	
	IntegrationData devIntegrationData;
	devIntegrationData.cudaAllocate(network.species);
	devIntegrationData.cudaCopy(integrationData, cudaMemcpyHostToDevice);
	checkCudaErrors();
	
	Globals devGlobals;
	devGlobals.cudaAllocate(network);
	
	Globals *devGlobalsPtr;
	cudaMalloc(&devGlobalsPtr, sizeof(Globals));
	cudaMemcpy(devGlobalsPtr, &devGlobals, sizeof(Globals), cudaMemcpyHostToDevice);
	checkCudaErrors();
	
	// Set up shared memory
	
	size_t sharedSize = integrateNetwork_sharedSize(network);
	printf("%d bytes of shared memory allocated.\n", (int) sharedSize);
	
	// Set up timer
	cudaEvent_t start, end;
	float duration;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	printf("Launching integration kernel...\n");
	cudaEventRecord(start);
	
	integrateNetwork<<<blocks, threads, sharedSize, stream>>>(
		devNetwork,
		devIntegrationData,
		devGlobalsPtr
	);
	
	cudaStreamSynchronize(stream);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	checkCudaErrors();
	cudaEventElapsedTime(&duration, start, end);
	printf("Kernel has finished in %f seconds\n", duration / 1000.0);
	
	integrationData.cudaCopy(devIntegrationData, cudaMemcpyDeviceToHost);
	checkCudaErrors();
	
	// TODO
	// Clean up the device IntegrationData
	
	// devIntegrationData.cudaFree();
	// devGlobals.cudaFree();
	
	cudaStreamDestroy(stream);
}


void FERNIntegrator::checkCudaErrors()
{
	// Sync the default stream before getting the last error
	cudaDeviceSynchronize();
	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("***CUDA error: %s\n", cudaGetErrorString(error));
		
		// Crash
		abort();
	}
}


// Copied from http://www.ncsa.illinois.edu/UserInfo/Training/Workshops/
// CUDA/presentations/tutorial-CUDA.html

void FERNIntegrator::devcheck(int gpudevice)
{
	int device_count;
	int device;
	
	// Get the number of non-emulation devices detected
	cudaGetDeviceCount(&device_count);
	if (gpudevice > device_count)
	{
		printf("gpudevice >= device_count ... exiting\n");
		exit(1);
	}
	
	cudaError_t cudareturn;
	cudaDeviceProp deviceProp;
	
	// cudaGetDeviceProperties() is also demonstrated in the deviceQuery example
	// of the sdk projects directory
	
	cudaGetDeviceProperties(&deviceProp, gpudevice);
	printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n",
		   deviceProp.major, deviceProp.minor);
	
	if (deviceProp.major > 999)
	{
		printf("warning, CUDA Device  Emulation (CPU) detected, exiting\n");
		exit(1);
	}
	
	// choose a cuda device for kernel execution
	cudareturn = cudaSetDevice(gpudevice);
	if (cudareturn == cudaErrorInvalidDevice)
	{
		printf("cudaSetDevice returned cudaErrorInvalidDevice\n");
		exit(1);
	}
	else
	{
		// double check that device was properly selected
		cudaGetDevice(&device);
		printf("cudaGetDevice()=%d\n", device);
	}
}
