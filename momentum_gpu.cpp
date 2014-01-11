/*
Copyright (C) 2013  iruu@10g.pl

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <ctime>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#define SUCCESS 0
#define FAILURE 1

#ifdef _MSC_VER
typedef long long unsigned uint64_t;
#else
typedef unsigned int DWORD;
#endif

typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

//#define DEBUG_PRINT
#define OPENCL_KERNEL_SOURCE
#include "OpenCLSha512.hpp"

using namespace std;

#define BIRTHDAYS_PER_HASH 8
#define MAX_MOMENTUM_NONCE  (1<<26)
#define SEARCH_SPACE_BITS 50
#define BIRTHDAYS_PER_HASH 8

#include <openssl/sha.h>

#ifdef _MSC_VER
long long getMS() {
	return GetTickCount64();
}
#else
#include <sys/time.h>

long long getMS() {
	struct timeval te;
	gettimeofday(&te, NULL); // get current time
	long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
	// printf("milliseconds: %lld\n", milliseconds);
	return milliseconds;
}
#endif

std::vector<std::pair<uint32_t, uint32_t> > momentum_search3(unsigned int *midHash, uint32_t *noncesBuf) {
	std::unordered_map<uint64_t, uint32_t> hash;
	std::vector<std::pair<uint32_t, uint32_t> > results;

	char hash_tmp[36];
	memcpy((char*) &hash_tmp[4], (char*)midHash, 32);
	uint32_t* index = (uint32_t*) hash_tmp;

	uint32_t noncesCount = noncesBuf[0];
	uint32_t *nonces = &noncesBuf[1];
	for (uint32_t n = 0; n < noncesCount; n++) {
		uint32_t i = nonces[n];
		*index = i;
		uint64_t result_hash[8];
		SHA512((unsigned char*) hash_tmp, sizeof(hash_tmp), (unsigned char*) result_hash);

		for (uint32_t x = 0; x < 8; ++x) {
			uint64_t birthday = result_hash[x] >> (64 - SEARCH_SPACE_BITS);
			uint32_t nonce = i + x;
			auto f = hash.find(birthday);
			if(f != hash.end()) {
				results.push_back(std::make_pair(f->second, nonce));
			}
			else
				hash[birthday] = nonce;
		}
	}
	return results;
}

OpenCL::OpenCLSha512 *_worker;

unsigned int testMine(uint32_t *midHash, unsigned int &dGpu) {
	DWORD t1, t2;
	auto &worker = *_worker;
	unsigned int *output = new unsigned int[worker.collisionListCount];
	if(!worker.setArgument(midHash))
		return -1;
	t1 = getMS();
	if(!worker.run())
		return -1;
	t2 = getMS();
	//printf("gpu time %dms\n", t2-t1);
	dGpu = t2-t1;

	if(!worker.readOutput((unsigned char*)output, worker.collisionListCount*sizeof(unsigned int))) return -1;
	//printf("%d %d\n", output[0], output[1]);

	t1 = getMS();
	auto results = momentum_search3(midHash, output);
	t2 = getMS();
	//printf("cpu search time: %u ms, found %u\n", t2-t1, (unsigned int)results.size());
	/*for( auto itr = results.begin(); itr != results.end(); ++itr )
	{
		printf("%u, %u\n", itr->first, itr->second);
	}
	printf("RESULTS END\n");*/
	delete output; 
	return (unsigned int)results.size();
}

int printSystemInfo() {
	try {
		std::vector< cl::Platform > platformList;
		cl::Platform::get(&platformList);
		if (platformList.size() == 0)
			throw std::exception("Getting platforms");
		std::cout << platformList.size() << " OpenCL computing platforms found" << std::endl;
		for(unsigned int i = 0; i < platformList.size(); i++) {
			std::string name;
			if(platformList[i].getInfo(CL_PLATFORM_NAME, &name) != CL_SUCCESS)
				throw std::exception("Getting platform info");
			std::cout << "[" << i << "] " << name << std::endl;
			std::cout << "GPU OpenCl Devices on this platform: " << std::endl;

			for(unsigned int p = 0; p < platformList.size(); p++) {
				std::vector<cl::Device> devices;
				//Query the platform and list gpu devices
				if(platformList[p].getDevices(CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS)
					throw std::exception("Getting device info");
				for(unsigned int d = 0; d < devices.size(); d++) {
					std::string deviceName;
					cl_ulong maxMemSize, maxBufSize;
					if(devices[d].getInfo(CL_DEVICE_NAME, &deviceName) != CL_SUCCESS)
						throw std::exception("Getting device name");
					if(devices[d].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &maxMemSize) != CL_SUCCESS)
						throw std::exception("Getting device max buffer size");
					if(devices[d].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxBufSize) != CL_SUCCESS)
						throw std::exception("Getting device global memory size");
					unsigned int maxMemoryMB = 0;
					if(maxBufSize >= 256*1024*1024 && maxMemSize >= 300*1024*1024)
						maxMemoryMB = 288;
					if(maxBufSize >= 512*1024*1024 && maxMemSize >= 620*1024*1024)
						maxMemoryMB = 544;
					if(maxBufSize >= 1024*1024*1024 && maxMemSize >= 1120*1024*1024)
						maxMemoryMB = 1056;
					if(maxBufSize >= 2048u*1024*1024 && maxMemSize >= 2150u*1024*1024)
						maxMemoryMB = 2080;
					std::cout << "\t[" << d << "] " << deviceName << ", maximum memory setting " << maxMemoryMB << "MB" << std::endl;
				}
			}
		}
	}
	catch (std::exception &e) {
		std::cerr << "ERROR: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int main(int argc, char* argv[]) {
	std::cout << "OpenCL PTS MINER version 0.0.2" << endl;
	try {
		unsigned int deviceId, platformId, memoryToUse, ilePetli;

		namespace po = boost::program_options; 
		std::string appName = boost::filesystem::basename(argv[0]); 
		po::options_description desc("Options"); 
		desc.add_options() 
			("memory,m", po::value<unsigned int>(&memoryToUse)->required(),
			"GPU memory to use (must be lower or equal maximum from systeminfo). Either 288, 544 or 1056 (MB).\n"\
			"Generally, higher is better. However, if significant amount of gpu memory is already in use, higher amounts may force copying to RAM, which will slow the algorithm by order(s) of magnitude.\n"\
			"On Vista and onwards, this may even cause the restart of the graphics driver (just wait a few seconds in case of a lockup). On Windows XP a restart may be probably required (or ctrl-c+patience).\n"\
			"Example application that uses significant amount of gpu memory is Adobe Reader.\n") 
			("platform,p", po::value<unsigned int>(&platformId)->default_value(0), "platform id to use")
			("benchmarkiterations,b", po::value<unsigned int>(&ilePetli)->default_value(3),
			"how many benchmark iterations")
			("device,d", po::value<unsigned int>(&deviceId)->default_value(0), "device id to use")
			("systeminfo", "list available platforms and opencl devices"); 

		po::variables_map vm;
		try {
			//throws on error
			po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
			if(vm.count("systeminfo"))
				return printSystemInfo();
			po::notify(vm); //throw for lack of required args
			if(memoryToUse != 288 && memoryToUse != 544 && memoryToUse != 1056)
				throw po::error("Invalid memory size. Must be 288, 544, or 1056");
		}
		catch(po::error& e) 
		{
			std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
			std::cout << desc << std::endl;
			return -1; 
		}

		std::cout << "If you're getting low performance, try either lower memory setting or freeing gpu memory" << endl;

		std::cout << ilePetli << " iterations" << std::endl;

		unsigned int midHash[8] = {0};

		//uint32_t a = 7553595; //81c4c712419b
		//uint32_t a = 14639133; //81c4c712419b
		//printf("%llx\n", getBirthdayHash(midHash, a)); 
		/*t1 = GetTickCount();
		momentum_search(midHash);
		t2 = GetTickCount();
		printf("time %dms\n", t2-t1);
		return 0;*/
		/*
		dla midhash i*33
		7553595, 14639133
		1738047, 42191680
		31159256, 55256412
		46181524, 59517431
		660418, 66458937
		*/

		/*
		dla midhash i*32 wg gpu:
		20929156, 31804625
		21984980, 41939391
		36471473, 45374303
		*/


		cl_int err;
		_worker = new OpenCL::OpenCLSha512(err, deviceId, platformId, memoryToUse);
	
		if(err == -1)
			return -1;
		printf("gpu initialized\n");
	
		DWORD t1 = getMS();
		unsigned int collisions = 0;
		double srednia = 0;
		for(unsigned int j = 0; j < ilePetli; j++) {
			for(int i = 0; i < 8; i++)
				midHash[i] = i*(32+j);

			unsigned int gpuTime = 0;
			int ret = testMine(midHash, gpuTime);
			srednia = (srednia*j + gpuTime)/(j+1);
			if(ret == -1) {
				std::cout << "blad!" << std::endl;
				break;
			}
			collisions += ret;
		}
		DWORD t2 = getMS();
		std::cout << ilePetli << " repeats, average " << srednia << "ms, " << 2*60*double(collisions)/(double(t2-t1)/1000) << " col/m" << std::endl;

		delete _worker;
		return SUCCESS;
	}
	catch (std::exception &e) {
		std::cerr << "Top level exception " << e.what() << std::endl;
		return -1;
	}
}