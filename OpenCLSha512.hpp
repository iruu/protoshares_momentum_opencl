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

#pragma once
//#ifdef _MSC_VER
//#include <CL/cl.h>
//#endif
//#define CL_USE_DEPRECATED_OPENCL_1_1_APIS //dla nvidii
//#undef CL_VERSION_1_2
#include <CL/cl.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <memory>
#include <climits>

#ifdef _MSC_VER
#include <intrin.h>
#define BSWAP64(x) (_byteswap_uint64((uint64_t)(x)))
#else
#define BSWAP64(x) (__builtin_bswap64((uint64_t)(x)))
#endif

namespace OpenCL {

typedef union {
    uint8_t                     mem_08[8];
    uint16_t                    mem_16[4];
    uint32_t                    mem_32[2];
    uint64_t                    mem_64[1];
} buffer_64;

typedef struct {
    uint64_t                    H[8];           //512 bits
    uint32_t                    total;
    uint32_t                    buflen;
	union {
		buffer_64                   buffer[16];     //1024bits
		uint32_t	birthday;
	};
} sha512_ctx;

//opencl data types end
#define MAX_MOMENTUM_NONCE  (1<<26)
#define SEARCH_SPACE_BITS 50
#define BIRTHDAYS_PER_HASH 8

#define GETCHAR(buf, index) ((buf)[(index)])
#define PUTCHAR(buf, index, val) (buf)[(index)] = val

inline void ctx_update(sha512_ctx *ctx, const uint64_t *string) {

	const uint32_t num = 0;//get_global_id(0);
	ctx->birthday = num*BIRTHDAYS_PER_HASH;
    ctx->total = 4*8+4;
	ctx->buflen = 4*8+4;

	uint64_t* const p = (uint64_t*)&ctx->buffer->mem_32[1];
	for(int i = 0; i < 4; i++)
		p[i] = string[i];
}

inline void ctx_append_1(sha512_ctx *ctx) {

    uint32_t length = ctx->buflen;
    PUTCHAR(ctx->buffer->mem_08, length, 0x80);

    while (++length & 3)
        PUTCHAR(ctx->buffer->mem_08, length, 0);

    if (length & 7) {
        uint32_t * l = (uint32_t*) (ctx->buffer->mem_08 + length);
        *l = 0;
        length += 4;
    }
    uint64_t *l = (uint64_t*) (ctx->buffer->mem_08 + length);

    while (length < 128) {
        *l++ = 0;
        length += 8;
    }
}

inline void ctx_add_length(sha512_ctx *ctx) {

    ctx->buffer->mem_64[15] = BSWAP64((uint64_t) (ctx->total * 8));
}

void finish_ctx(sha512_ctx *ctx) {
    ctx_append_1(ctx);
    ctx_add_length(ctx);
    ctx->buflen = 0;
}

//ctx functions end

using namespace std;

#ifndef CL_VERSION_1_2
#define enqueueBarrierWithWaitList enqueueBarrier
#endif

struct OpenCLSha512 {
	struct KernelArguments {
		int inputCtx, bitmap, bitmap2, collisionList, domain, rotateN;
		string name;
		KernelArguments() : bitmap2(-1) {}
	};

	struct KernelsArguments {
		KernelArguments zeroBitmap, phase1, phase2, phase3, phase4, phase5, phase6;
	};


	KernelsArguments kernelsArguments;
	cl_int	status;
	unique_ptr<cl::Context> context;
	unique_ptr<cl::CommandQueue> commandQueue;
	unique_ptr<cl::Program> program;
	unique_ptr<cl::Kernel> phase1kernel, zeroBitmapKernel, phase2kernel, phase3kernel, phase4kernel, phase5kernel, phase6kernel;

	unique_ptr<cl::Buffer> bitmapBuffer, bitmapBuffer2, inputBuffer, collisionListBuffer, collisionList2Buffer;

	static const unsigned int workSize = 1<<23;  //1<<23; //MAX_MOMENTUM_NONCE/BIRTHDAYS_PER_HASH;

	//static const unsigned int localWorkers = 128; //workSize must be divisibe by this
	unsigned int bitmapBufSize; // = BITMAP_SIZE; //1<<29; //2^32 bits, 512 megabits //outputBufCount*sizeof(unsigned long long);

	static const unsigned int inputBufferCount = 16;
	static const unsigned int inputBufferSize = 16 * sizeof(uint64_t);
	std::unique_ptr<uint64_t[]> in;

	/* on average, we can expect ~2^19 collisions
		n - how many items we hash
		k - how many locations in hash table
		def col(n, k):
		    return n - k + k*((1-1.0/k)**n)

	   the first half of the buffer is the random access part, zeroed at the beginning.
	   the second is the ordered list part - first dword is the current last dword. requires atomic_inc

	   the algorithm for storage is:
	   	 index = atomic_inc(buf[size/2])
		 buf[size/2+index] = nonce*/
	
	/*index = random (some bits from sha512)
	   old = atomic_max(&buf[index], 1)
	   if(old == 0) //field was empty, now there's 1
	     buf[index] = nonce+2;
	   else //no luck
	     index = atomic_inc(buf[size/2])
		 buf[size/2+index] = nonce
	*/
	static const unsigned int collisionListCount = 1<<22; //2^22
	static const unsigned int collisionListSize; //16 MB

	std::unique_ptr<unsigned int[]> output; //list of suspicious nonces, first dword is a count
	static const unsigned int outputCount = collisionListCount;
	static const unsigned int outputSize = outputCount*sizeof(unsigned int);

	bool firstRun;

	OpenCLSha512(cl_int &err, unsigned int deviceId, unsigned platformId, unsigned int memoryToUse) : bitmapBufSize((memoryToUse-32)*1024*1024) {
		kernelsArguments = oneBufKernelsArguments(); //twoBufKernelsArguments();
		firstRun = true;
		if(bitmapBufSize != 256*1024*1024 && bitmapBufSize != 512*1024*1024 && bitmapBufSize != 1024*1024*1024) {
			cerr << "Invalid memory buf size" << endl;
			err = -1;
		}
		else {
			try {
				startup(deviceId, platformId, memoryToUse);
				initializeKernels();
			}
			catch (GpuException &e) {
				e.print();
				err = -1;
			}
		}
	}

private:

	static KernelsArguments oneBufKernelsArguments() {
		KernelsArguments kernels1Buf;
		kernels1Buf.phase1.inputCtx = 0;
		kernels1Buf.phase1.bitmap = 1;
		kernels1Buf.phase1.collisionList = 2;
		kernels1Buf.phase1.name = "birthdayPhase1";

		kernels1Buf.phase2.inputCtx = 0;
		kernels1Buf.phase2.bitmap = 1;
		kernels1Buf.phase2.collisionList = 2;
		kernels1Buf.phase2.name = "birthdayPhase2";

		kernels1Buf.phase3.inputCtx = 0;
		kernels1Buf.phase3.bitmap = 1;
		kernels1Buf.phase3.collisionList = 2;
		kernels1Buf.phase3.name = "birthdayPhase3";

		kernels1Buf.phase4.inputCtx = 0;
		kernels1Buf.phase4.bitmap = 1;
		kernels1Buf.phase4.domain = 2;
		kernels1Buf.phase4.collisionList = 3;
		kernels1Buf.phase4.rotateN = 4;
		kernels1Buf.phase4.name = "birthdayPhase4";

		kernels1Buf.phase5.inputCtx = 0;
		kernels1Buf.phase5.bitmap = 1;
		kernels1Buf.phase5.collisionList = 2;
		kernels1Buf.phase5.rotateN = 3;
		kernels1Buf.phase5.name = "birthdayPhase5";

		kernels1Buf.phase6.inputCtx = 0;
		kernels1Buf.phase6.bitmap = 1;
		kernels1Buf.phase6.domain = 2;
		kernels1Buf.phase6.collisionList = 3;
		kernels1Buf.phase6.rotateN = 4;
		kernels1Buf.phase6.name = "birthdayPhase6";

		kernels1Buf.zeroBitmap.bitmap = 0;
		kernels1Buf.zeroBitmap.name = "zeroBitmap";

		return kernels1Buf;
	}

	static KernelsArguments twoBufKernelsArguments() {
		KernelsArguments kernels2Buf;
		kernels2Buf.phase1.inputCtx = 0;
		kernels2Buf.phase1.bitmap = 1;
		kernels2Buf.phase1.bitmap2 = 2;
		kernels2Buf.phase1.collisionList = 3;
		kernels2Buf.phase1.name = "_2buf_birthdayPhase1";

		kernels2Buf.phase2.inputCtx = 0;
		kernels2Buf.phase2.bitmap = 1;
		kernels2Buf.phase2.bitmap2 = 2;
		kernels2Buf.phase2.collisionList = 3;
		kernels2Buf.phase2.name = "_2buf_birthdayPhase2";

		kernels2Buf.phase3.inputCtx = 0;
		kernels2Buf.phase3.bitmap = 1;
		kernels2Buf.phase3.bitmap2 = 2;
		kernels2Buf.phase3.collisionList = 3; 
		kernels2Buf.phase3.name = "_2buf_birthdayPhase3";

		kernels2Buf.phase4.inputCtx = 0;
		kernels2Buf.phase4.bitmap = 1;
		kernels2Buf.phase4.bitmap2 = 2;
		kernels2Buf.phase4.domain = 3;
		kernels2Buf.phase4.collisionList = 4;
		kernels2Buf.phase4.rotateN = 5;
		kernels2Buf.phase4.name = "_2buf_birthdayPhase4";

		kernels2Buf.phase5.inputCtx = 0;
		kernels2Buf.phase5.bitmap = 1;
		kernels2Buf.phase5.bitmap2 = 2;
		kernels2Buf.phase5.collisionList = 3;
		kernels2Buf.phase5.rotateN = 4;
		kernels2Buf.phase5.name = "_2buf_birthdayPhase5";

		kernels2Buf.phase6.inputCtx = 0;
		kernels2Buf.phase6.bitmap = 1;
		kernels2Buf.phase6.bitmap2 = 2;
		kernels2Buf.phase6.domain = 3;
		kernels2Buf.phase6.collisionList = 4;
		kernels2Buf.phase6.rotateN = 5;
		kernels2Buf.phase6.name = "_2buf_birthdayPhase6";

		kernels2Buf.zeroBitmap.bitmap = 0;
		kernels2Buf.zeroBitmap.bitmap2 = 1;
		kernels2Buf.zeroBitmap.name = "_2buf_zeroBitmap";

		return kernels2Buf;
	}

	class GpuException : public std::runtime_error {
	public:
		cl_int status;
		string statusDescription;
		GpuException(const string &msg, cl_int _status) : std::runtime_error(msg), status(_status) {}
		GpuException(const string &msg, cl_int _status, const string &_statusDescription) : std::runtime_error(msg), status(_status), statusDescription(_statusDescription) {}

		void print() {
			cerr << "ERROR: " << this->what();
			if(statusDescription != "")
				cerr << ", STATUS: " << statusDescription;
			cerr << endl;
		}
	};

	void errCheck(cl_int status, string errorMsg, std::function<string(cl_int)> stringifier) const {
		if(status != CL_SUCCESS)
			throw GpuException(errorMsg, status, stringifier(status));
	}

	void errCheck(cl_int status, string errorMsg) const {
		if(status != CL_SUCCESS)
			throw GpuException(errorMsg, status);
	}

	void errTrueCheck(bool cond, string errorMsg) const {
		if(!cond)
			throw GpuException(errorMsg, status);
	}

	void startup(unsigned int deviceId, unsigned int platformId, unsigned int memoryToUse) {
		in.reset(new uint64_t[inputBufferCount]);
		output.reset(new unsigned int[outputCount]);
		//Step1: Getting platforms and choose an available one.
		std::vector< cl::Platform > platformList;
		errCheck(cl::Platform::get(&platformList), "getting platforms");
		errTrueCheck(platformList.size() != 0, "got no opencl platforms");
		errTrueCheck(platformId < platformList.size(), "Too high platform id");
		cl::Platform &platform = platformList[platformId];
		//Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.
		std::vector<cl::Device> devices;
		errCheck(platform.getDevices(CL_DEVICE_TYPE_GPU, &devices), "getting gpu devices");
		errTrueCheck(devices.size() != 0, "No GPU device available.");
		errTrueCheck(deviceId < devices.size(), "Too high deviceId requested");
		//Step 3: Create context.
		context.reset(new cl::Context(devices[deviceId]));
		
		//Step 4: Creating command queue associate with the context.
		commandQueue.reset(new cl::CommandQueue(*context, devices[deviceId]));

		//Step 5: Create program object
		vector<cl::Device> usedDevices;
		usedDevices.push_back(devices[deviceId]);
#ifdef OPENCL_KERNEL_SOURCE
		auto kernelFileName = "momentum_miner.cl";
		std::ifstream kernelFile(kernelFileName);
		errTrueCheck(kernelFile.is_open(), "Error opening kernel file " + string(kernelFileName));

		//read .cl file
		std::string clCode(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
		std::ostringstream header;
		header << "#define BITMAP_SIZE " << bitmapBufSize << "lu" << std::endl;
		header << "#define BITMAP_INDEX_TYPE ";
		header << ( ((uint64_t)bitmapBufSize)*8 > (uint64_t)UINT_MAX ? "uint64_t" : "uint32_t") << std::endl;

		std::string prog = header.str() + clCode;

		cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));
		program.reset(new cl::Program(*context, source));
#else
		//binary kernel
		ostringstream kernelFileName;
		kernelFileName << "miner" << memoryToUse << ".bin";
		cout << "Loading kernel " << kernelFileName.str() << endl;
		std::ifstream kernelFile(kernelFileName.str(), std::ios::in|std::ios::binary);
		errTrueCheck(kernelFile.is_open(), "Error opening kernel file " + kernelFileName.str());
		std::vector<char> binary((istreambuf_iterator<char>(kernelFile)),
											(istreambuf_iterator<char>()));
		cl::Program::Binaries binaries;
		binaries.push_back(make_pair<const void*, size_t>((const void*)binary.data(), binary.size()));
		vector<cl_int> binaryStatus(1);
		program.reset(new cl::Program(*context, usedDevices, binaries, &binaryStatus, &status));
		errCheck(binaryStatus[0], "Error creating kernels from binary");
#endif
		//Step 6: Build program.
		errCheck(program->build(usedDevices), "building program", stringifyBuildProgramError);
	}

	OpenCLSha512(const OpenCLSha512&);                 //no copy-construction
	OpenCLSha512& operator=(const OpenCLSha512&);      //no assignment

	void initializeZeroBitmapKernel() {
		cl_int 	err = CL_SUCCESS;
		zeroBitmapKernel.reset(new cl::Kernel(*program, kernelsArguments.zeroBitmap.name.c_str(), &err));
		errCheck(err, "creating zeroBitmap kernel");
		errCheck(zeroBitmapKernel->setArg(kernelsArguments.zeroBitmap.bitmap, *bitmapBuffer), "zeroBitmapKernel->setArg 0"); 
		if(kernelsArguments.zeroBitmap.bitmap2 != -1) 
			errCheck(zeroBitmapKernel->setArg(kernelsArguments.zeroBitmap.bitmap2, *bitmapBuffer2), "zeroBitmapKernel->setArg 1");
	}

	void initializePhaseOneCollisionListArgs(cl::Kernel *kernel) {
		errCheck(kernel->setArg(kernelsArguments.phase1.inputCtx, *inputBuffer), "kernel.setArg 0"); 
		errCheck(kernel->setArg(kernelsArguments.phase1.bitmap, *bitmapBuffer), "kernel.setArg 1");
		if(kernelsArguments.phase1.bitmap2 != -1) 
			errCheck(kernel->setArg(kernelsArguments.phase1.bitmap2, *bitmapBuffer2), "kernel.setArg 2");
		errCheck(kernel->setArg(kernelsArguments.phase1.collisionList, *collisionListBuffer), "kernel.setArg 3");
	}

	cl::Kernel *createOneCollisionKernel(const string kernelName, cl_int &err) {
		auto kernel = new cl::Kernel(*program, kernelName.c_str(), &err);
		errCheck(err, "creating " + kernelName);
		initializePhaseOneCollisionListArgs(kernel);
		return kernel;
	}

	void initializeBuffers() {
		cl_int err;
		inputBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_ONLY, 16*sizeof(uint64_t), nullptr, &err));
		errCheck(err, "creating inputBuffer");

		if(kernelsArguments.phase1.bitmap2 != -1) {
			bitmapBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, bitmapBufSize/2, nullptr, &err));
			errCheck(err, "creating bitmapBuffer");

			bitmapBuffer2.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, bitmapBufSize/2, nullptr, &err));
			errCheck(err, "creating bitmapBuffer2");
		}
		else {
			bitmapBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, bitmapBufSize, nullptr, &err));
			errCheck(err, "creating bitmapBuffer");
		}

		collisionListBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, collisionListSize, nullptr, &err));
		errCheck(err, "creating collisionListBuffer");

		collisionList2Buffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, collisionListSize, nullptr, &err));
		errCheck(err, "creating collisionList2Buffer");
	}

	void setPhases4_5_6_RotateAmount(unsigned int rotateAmount) {
		errCheck(phase4kernel->setArg(kernelsArguments.phase4.rotateN, rotateAmount), "phase4Kernel rot");
		errCheck(phase5kernel->setArg(kernelsArguments.phase5.rotateN, rotateAmount), "phase5Kernel rot");
		errCheck(phase6kernel->setArg(kernelsArguments.phase6.rotateN, rotateAmount), "phase6Kernel rot"); 
	}

	void initializeKernels() {
		initializeBuffers();
		initializeZeroBitmapKernel();

		cl_int 	err = CL_SUCCESS;
		phase1kernel.reset(createOneCollisionKernel(kernelsArguments.phase1.name, err));
		errCheck(err, "creating birthdayPhase1");
		phase2kernel.reset(createOneCollisionKernel(kernelsArguments.phase2.name, err));
		errCheck(err, "creating birthdayPhase2");
		phase3kernel.reset(createOneCollisionKernel(kernelsArguments.phase3.name, err));
		errCheck(err, "creating birthdayPhase3");
		phase4kernel.reset(createOneCollisionKernel(kernelsArguments.phase4.name, err));
		errCheck(err, "creating birthdayPhase4");
		errCheck(phase4kernel->setArg(kernelsArguments.phase4.collisionList, *collisionList2Buffer), "phase4Kernel setArg 4");
		phase5kernel.reset(new cl::Kernel(*program, kernelsArguments.phase5.name.c_str(), &err));
		errCheck(err, "creating birthdayPhase5");
		errCheck(phase5kernel->setArg(kernelsArguments.phase5.inputCtx, *inputBuffer), "birthdayPhase5 setArg 0");
		errCheck(phase5kernel->setArg(kernelsArguments.phase5.bitmap, *bitmapBuffer), "birthdayPhase5 setArg 1");
		if(kernelsArguments.phase5.bitmap2 != -1)
			errCheck(phase5kernel->setArg(kernelsArguments.phase5.bitmap2, *bitmapBuffer2), "birthdayPhase5 setArg 2");
		errCheck(phase5kernel->setArg(kernelsArguments.phase5.collisionList, *collisionList2Buffer), "birthdayPhase5 setArg 3");

		phase6kernel.reset(createOneCollisionKernel(kernelsArguments.phase6.name, err));
		errCheck(err, "creating birthdayPhase6");
		errCheck(phase6kernel->setArg(kernelsArguments.phase6.collisionList, *collisionList2Buffer), "birthdayPhase6 setArg 4");

		setPhases4_5_6_RotateAmount(13);
	}

	public:

	bool setArgument(const unsigned int *midHash) {
		sha512_ctx ctx = {};
		ctx_update(&ctx, (const uint64_t*)midHash);
		finish_ctx(&ctx);

		for(int i = 0; i < 16; i++)
			in[i] = BSWAP64(ctx.buffer[0].mem_64[i]);

		/*for(int i = 0; i < 32; i++)
			printf("%x,", ((uint32_t*)in)[i]);
		puts("\n");*/

		cl_int	status;
		status = commandQueue->enqueueWriteBuffer(*inputBuffer, CL_FALSE, 0, inputBufferSize, in.get());
		if(status != CL_SUCCESS) {
			cout << "Error enqueueWriteBuffer" << endl;
			return false;
		}
		return true;
	}

	static const uint32_t zero;
	cl::Event lastEventForRead;

	void zeroBitmap(cl::Event *e = nullptr) {
		unsigned int size;
		if(kernelsArguments.zeroBitmap.bitmap2 != -1)
			size = bitmapBufSize/sizeof(cl_float8)/2;
		else
			size = bitmapBufSize/sizeof(cl_float8);
		errCheck(commandQueue->enqueueNDRangeKernel(*zeroBitmapKernel, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, e), "zeroing memory");
	}

	void setPhase3_4_6Arguments(cl::Buffer &domain, cl::Buffer &collisions, uint32_t rotateAmount) {
		errCheck(phase4kernel->setArg(kernelsArguments.phase4.domain, domain), "setPhase3_4_6Arguments:phase4Kernel->setArg 3");
		errCheck(phase4kernel->setArg(kernelsArguments.phase4.collisionList, collisions), "setPhase3_4_6Arguments:phase4Kernel->setArg 4");
		errCheck(phase5kernel->setArg(kernelsArguments.phase5.collisionList, collisions), "setPhase3_4_6Arguments:phase5Kernel->setArg 3");
		errCheck(phase6kernel->setArg(kernelsArguments.phase6.domain, domain), "setPhase3_4_6Arguments:phase6Kernel->setArg 3");
		errCheck(phase6kernel->setArg(kernelsArguments.phase6.collisionList, collisions), "setPhase3_4_6Arguments:phase6Kernel->setArg 4");

		setPhases4_5_6_RotateAmount(rotateAmount);
	}

	bool run() {
		try {
			cl::Event ev[3];
			std::vector<cl::Event> waitingList;
			volatile uint32_t collisionsFound = 0;

			//we duplicate this in a nonblocking way at the end of the queue - so that the work can be done while cpu does something else
			if(firstRun) {
				zeroBitmap();
				firstRun = false;
			}
			errCheck(commandQueue->enqueueWriteBuffer(*collisionListBuffer, CL_FALSE, 0, sizeof(zero), (void*)&zero, nullptr), "enqueueWriteBuffer(*collisionListBuffer", stringifyEnqueueReadBufferError);
			commandQueue->enqueueBarrierWithWaitList();
			errCheck(commandQueue->enqueueNDRangeKernel(*phase1kernel, cl::NullRange, cl::NDRange(workSize), cl::NullRange), "enqueueNDRangeKernel(*phase1kernel", stringifyenqueueNDRangeKernelError);
			commandQueue->enqueueBarrierWithWaitList();
			errCheck(commandQueue->enqueueReadBuffer(*collisionListBuffer, CL_FALSE, 0, sizeof(collisionsFound), (void*)&collisionsFound, nullptr, &ev[0]), "enqueueReadBuffer(*collisionListBuffer", stringifyEnqueueReadBufferError);
			zeroBitmap();

			ev[0].wait();
		
	#ifdef DEBUG_PRINT
			printf("phase 1 collisions: %d\n", collisionsFound);
	#endif
			commandQueue->enqueueBarrierWithWaitList();
			errCheck(commandQueue->enqueueNDRangeKernel(*phase2kernel, cl::NullRange, cl::NDRange(collisionsFound), cl::NullRange), "enqueueNDRangeKernel(*phase2kernel", stringifyenqueueNDRangeKernelError);
			errCheck(commandQueue->enqueueWriteBuffer(*collisionListBuffer, CL_FALSE , 0, sizeof(zero), &zero), "enqueueWriteBuffer(*collisionListBuffer", stringifyEnqueueReadBufferError);
			commandQueue->enqueueBarrierWithWaitList();
			errCheck(commandQueue->enqueueNDRangeKernel(*phase3kernel, cl::NullRange, cl::NDRange(workSize), cl::NullRange), "enqueueNDRangeKernel(*phase3kernel", stringifyenqueueNDRangeKernelError);
			commandQueue->enqueueBarrierWithWaitList();

			errCheck(commandQueue->enqueueReadBuffer(*collisionListBuffer, CL_FALSE, 0, sizeof(collisionsFound), (void*)&collisionsFound, nullptr, &ev[0]), "enqueueReadBuffer(*collisionListBuffer", stringifyEnqueueReadBufferError);
			zeroBitmap();

			ev[0].wait(); //we have to wait in order to set worksize
			setPhase3_4_6Arguments(*collisionListBuffer, *collisionList2Buffer, 13);
	#ifdef DEBUG_PRINT
			printf("phase 3 collisions: %d\n", collisionsFound);
	#endif

			int c = runPhases4_5_6(collisionsFound, *collisionList2Buffer);
			collisionsFound = (unsigned int)c; //works for <2^31 collisions
	#ifdef DEBUG_PRINT
			printf("phase 6 collisions: %d\n", c);
	#endif
			//druga iteracja

			/*
			1. phase4 (0 inputBuffer, 1 bitmapBuffer, 2 collisionListBuffer, 3 collisionList2Buffer, 4 rotateN)
			w cl (input, bitmap, in domain, out collisions)
			2. phase5 (0 inputBuffer, 1 bitmapBuffer, 2 collisionList2Buffer, 4 rotateN)
			w cl (input, out bitmap, in kolizje z phase 4)
			3. phase6 (0 inputBuffer, 1 bitmapBuffer, 2 collisionListBuffer, 3 collisionList2Buffer, 4 rotateN)
			w cl (input, in bitmap, in domain, out collisions)
			i teraz trzeba zawolac phase4 z collisionList2Buffer jako domain i innym hashem
			*/
		
			setPhase3_4_6Arguments(*collisionList2Buffer, *collisionListBuffer, 21);

			//phase4 oczekuje pustej
			zeroBitmap();
			c = runPhases4_5_6(collisionsFound, *collisionListBuffer);
			collisionsFound = (unsigned int)c; //works for <2^31 collisions
	#ifdef DEBUG_PRINT
			printf("phase 6B collisions: %d\n", c);
	#endif

			//przywroc poczatkowa kolejnosc
			setPhase3_4_6Arguments(*collisionListBuffer, *collisionList2Buffer, 13);
		
			commandQueue->enqueueBarrierWithWaitList();
			zeroBitmap();
			return true;
		}
		catch (GpuException &e) {
			e.print();
			return false;
		}
	}

	int runPhases4_5_6(uint32_t collisionsFound, cl::Buffer &collisions) {
		errCheck(commandQueue->enqueueWriteBuffer(collisions, CL_FALSE, 0, sizeof(zero), &zero), "runPhases4_5_6:enqueueWriteBuffer(collisions", stringifyEnqueueReadBufferError);

		commandQueue->enqueueBarrierWithWaitList();
		/*
		to ma wypelnic bitmape nowym hashem dla wszystkich elementow znalezionych przez fazy 1-3, oraz wypelnic liste collisionList2Buffer nowymi kolizjami
		powtorka fazy 1, tylko dla elementow znalezionych przez faze 3
		*/
		errCheck(commandQueue->enqueueNDRangeKernel(*phase4kernel, cl::NullRange, cl::NDRange(collisionsFound), cl::NullRange), "runPhases4_5_6:enqueueNDRangeKernel(*phase4kernel", stringifyenqueueNDRangeKernelError);

		const uint32_t domainSize = collisionsFound;
		commandQueue->enqueueBarrierWithWaitList();
		zeroBitmap(nullptr);

		cl::Event ev;
		errCheck(commandQueue->enqueueReadBuffer(collisions, CL_FALSE, 0, sizeof(collisionsFound), (void*)&collisionsFound, nullptr, &ev), "runPhases4_5_6:enqueueReadBuffer(collisions", stringifyEnqueueReadBufferError);
		
		/*
		to wypelnia pusta bitmape wszystkimi kolizjami z poprzedniego kroku
		trzeci argument to czwarty argument phase4kernel
		*/
		commandQueue->enqueueBarrierWithWaitList(); //for zerobitmap

		ev.wait();
		errCheck(commandQueue->enqueueNDRangeKernel(*phase5kernel, cl::NullRange, cl::NDRange(collisionsFound), cl::NullRange), "runPhases4_5_6:enqueueNDRangeKernel(*phase5kernel", stringifyenqueueNDRangeKernelError);
		errCheck(commandQueue->enqueueWriteBuffer(collisions, CL_FALSE, 0, sizeof(zero), &zero), "runPhases4_5_6:enqueueWriteBuffer(collisions", stringifyEnqueueReadBufferError);
		commandQueue->enqueueBarrierWithWaitList();

		errCheck(commandQueue->enqueueNDRangeKernel(*phase6kernel, cl::NullRange, cl::NDRange(domainSize), cl::NullRange, nullptr, &lastEventForRead), "runPhases4_5_6:enqueueNDRangeKernel(*phase6kernel", stringifyenqueueNDRangeKernelError); 
		commandQueue->enqueueBarrierWithWaitList();
		errCheck(commandQueue->enqueueReadBuffer(collisions, CL_TRUE, 0, sizeof(collisionsFound), (void*)&collisionsFound), "runPhases4_5_6:enqueueReadBuffer(collisions", stringifyEnqueueReadBufferError); 
		return collisionsFound;
	}

	bool readOutput(unsigned char *outBuffer, unsigned int outBufferSize) {
		cl_int	status;

		std::vector<cl::Event> waitList;
		waitList.push_back(lastEventForRead);
		status = commandQueue->enqueueReadBuffer(*collisionListBuffer, CL_TRUE, 0, min(outBufferSize,collisionListSize), outBuffer, &waitList);
		if(status != CL_SUCCESS) {
			cout << "Error enqueueReadBuffer in readOutput" << endl;
			cout << stringifyEnqueueReadBufferError(status) << endl;
			return false;
		}
		return true;
	}

#define ERRSTRCASE(e) case e: return #e

	static std::string stringifyenqueueNDRangeKernelError(cl_int status) {
		switch(status) {
			ERRSTRCASE(CL_INVALID_PROGRAM_EXECUTABLE);
			ERRSTRCASE(CL_INVALID_COMMAND_QUEUE);
			ERRSTRCASE(CL_INVALID_KERNEL);
			ERRSTRCASE(CL_INVALID_CONTEXT);
			ERRSTRCASE(CL_INVALID_KERNEL_ARGS);
			ERRSTRCASE(CL_INVALID_WORK_DIMENSION);
			ERRSTRCASE(CL_INVALID_GLOBAL_WORK_SIZE);
			ERRSTRCASE(CL_INVALID_GLOBAL_OFFSET);
			ERRSTRCASE(CL_INVALID_WORK_GROUP_SIZE);
			ERRSTRCASE(CL_INVALID_WORK_ITEM_SIZE);
			ERRSTRCASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
			ERRSTRCASE(CL_INVALID_IMAGE_SIZE);
			ERRSTRCASE(CL_OUT_OF_RESOURCES);
			ERRSTRCASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
			ERRSTRCASE(CL_INVALID_EVENT_WAIT_LIST);
			ERRSTRCASE(CL_OUT_OF_HOST_MEMORY);
			default: return "unknown";
		}
	}

	static std::string stringifyBuildProgramError(cl_int status) {
		switch(status) {
			ERRSTRCASE(CL_SUCCESS);
			ERRSTRCASE(CL_INVALID_PROGRAM);
			ERRSTRCASE(CL_INVALID_VALUE);
			ERRSTRCASE(CL_INVALID_DEVICE);
			ERRSTRCASE(CL_INVALID_BINARY);
			ERRSTRCASE(CL_INVALID_BUILD_OPTIONS);
			ERRSTRCASE(CL_INVALID_OPERATION);
			ERRSTRCASE(CL_COMPILER_NOT_AVAILABLE);
			ERRSTRCASE(CL_BUILD_PROGRAM_FAILURE);
			ERRSTRCASE(CL_OUT_OF_RESOURCES);
			ERRSTRCASE(CL_OUT_OF_HOST_MEMORY);
			default: return "unknown";
		}
	}

	static std::string stringifyEnqueueReadBufferError(cl_int status) {
		switch(status) {
		case CL_SUCCESS:
			return "CL_SUCCESS";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		default:
			return "unknown status!";
		}
	}
};

const unsigned int OpenCLSha512::collisionListSize = OpenCLSha512::collisionListCount*sizeof(unsigned int); //16 MB
const uint32_t OpenCLSha512::zero = 0;

};