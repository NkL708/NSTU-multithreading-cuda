#include <ctime>
#include <iostream>

#include <stdio.h>
//#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ bool isPalindrome(int num) 
{
	// Get number lenght
	int length = 0;
	for (int i = num; i > 0; i /= 10) {
		length++;
	}
	int* digits = new int[length];
	// Fill digits array
	int tempNum = num;
	for (int i = 0; i < length; i++) {
		digits[i] = tempNum % 10;
		tempNum /= 10;
	}
	// Comparison mirrored elements
	for (int begin = 0, end = length - 1; begin != end; begin++, end--) {
		if (digits[begin] != digits[end]) {
			return false;
		}
	}
	return true;
}

int getMaxPalindrome(int n)
{
	int maxPalindrome = 0;
	for (int num = 0; num < n; num++) {
		for (int firstPower = 0; pow(firstPower, 2) <= num; firstPower++) {
			int sum = 0;
			for (int numberOfSeq = firstPower; sum < num; numberOfSeq++) {
				sum += (int) pow(numberOfSeq, 2);
			}
			if (sum == num && isPalindrome(num)) {
				maxPalindrome = num;
			}
		}
	}
	return maxPalindrome;
}

void getSequence(int num)
{
	int lastPower = 0;
	for (int firstPower = 0; pow(firstPower, 2) <= num; firstPower++) {
		int sum = 0;;
		for (int numberOfSeq = firstPower; sum < num; numberOfSeq++) {
			lastPower = numberOfSeq;
			sum += (int)pow(numberOfSeq, 2);
		}
		if (sum == num && isPalindrome(num)) {
			break;
		}
	}
	for (int numberOfSeq = lastPower, sum = 0; sum < num; numberOfSeq--) {
		sum += (int)pow(numberOfSeq, 2);
		std::cout << numberOfSeq << "^2 + ";
	}
	std::cout << " = " << num << "\n\n";
}

int getCores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 8: // Ampere
		if (devProp.minor == 0) cores = mp * 64;
		else if (devProp.minor == 6) cores = mp * 128;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

__global__ void getMaxPalindromeCUDA(int n, int result)
{
	for (int num = threadIdx.x; num < n; num += threadIdx.x) {
		for (int firstPower = 0; firstPower * firstPower <= num; firstPower++) {
			int sum = 0;
			for (int numberOfSeq = firstPower; sum < num; numberOfSeq++) {
				sum += numberOfSeq * numberOfSeq;
			}
			if (sum == num && isPalindrome(num)) {
				result = num;
			}
		}
	}
}

int main(int argc, char* argv[]) {
	double durationL, durationP;
	clock_t timeBegin, timeEnd;
	int num, result, cores;
	// Get CUDA cores
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device, 0);
	cores = getCores(device);
	if (argc > 1) {
		num = atoi(argv[1]);
	}
	else {
		num = 100000;
	}
	timeBegin = clock();
	result = getMaxPalindrome(num);
	timeEnd = clock();
	std::cout << "Linear result: " << result << std::endl;
	durationL = (double)(timeEnd - timeBegin) / CLOCKS_PER_SEC;
	timeBegin = clock();
	getMaxPalindromeCUDA << <1, cores >> > (num, result);
	timeEnd = clock();
	std::cout << "Parallel result: " << result << std::endl;
	durationP = (double)(timeEnd - timeBegin) / CLOCKS_PER_SEC;
	std::cout << "Linear time: " << durationL << std::endl;
	std::cout << "Parallel time: " << durationP << std::endl;
	std::cout << "Parallel faster than Linear on: " << durationL - durationP << std::endl;
	getSequence(result);
	return 0;
}