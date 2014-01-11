#include <string>
#include <fstream>
#include <memory>

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
  
#include "OpenCLSha512.hpp"

OpenCLSha512 *worker;
static bool initialized = false;
const unsigned int outputCount = 1<<22;
unsigned int *output;

#include <sys/time.h>

long long getMS() {
	struct timeval te;
	gettimeofday(&te, NULL); // get current time
	long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
	// printf("milliseconds: %lld\n", milliseconds);
	return milliseconds;
}

bool initialize() {
	if(!initialized) {
		cl_int err;
		worker = new OpenCLSha512(err);
		if(err != CL_SUCCESS)
			return false;
		output = new unsigned int[outputCount];
		initialized = true;
	}
	return true;
}

void cleanup() {
  if(initialized) {
    printf("cleaning opencl\n");
    delete [] output;
    delete worker;
  }
}

#include <openssl/sha.h>
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

#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
boost::mutex minerMutex;

/*
midHash - midHash do obliczen
zwraca: znalezione pary
*/
std::vector< std::pair<uint32_t,uint32_t> > mine(unsigned int *midHash) {
  //uint64_t t1, t2, gt1, gt2;
	if(!initialized) {
		printf("Miner not initialized!\n");
		return std::vector< std::pair<uint32_t,uint32_t> >();
	}

	const unsigned int midHashSize = 32;
	boost::mutex::scoped_lock lock(minerMutex);

	//t1 = getMS();
	static unsigned long long lastMidHashes[2][midHashSize/sizeof(unsigned long long)] = {{0}};
	static unsigned int lastIndex = 0;

	//check if midhash is not repeated
	for(int i = 0; i < 2; i++) {
		if(memcmp(&lastMidHashes[(lastIndex+i)%2][0], midHash, midHashSize) == 0) {
			printf("OpenCL Miner: ignoring duplicated midHash!\n");
			return std::vector< std::pair<uint32_t,uint32_t> >();
		}
	}
	lastIndex = (lastIndex+1)%2;
	memcpy(&lastMidHashes[lastIndex][0], midHash, midHashSize);

	//gt1 = getMS();
	worker->setArgument(midHash);
	worker->run();
	worker->readOutput((unsigned char*)output, outputCount*sizeof(unsigned int));
	//gt2 = getMS();
	auto results = momentum_search3(midHash, output);
	//t2 = getMS();
	//printf("gpu time %dms, time %dms\n", gt2-gt1, t2-t1);
	return results;
}

namespace OpenCLMiner {
  extern void momentum_search(unsigned int *midHash, std::vector<std::pair<uint32_t, uint32_t> > &results) {
    results = ::mine(midHash);
  }

  extern bool initializeOpenClMiner() {
    return ::initialize();
  }

  extern void clean() {
    ::cleanup();
  }
}
