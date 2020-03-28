#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <cstdio>
#include <cstdint>
#include <string>
#include <filesystem>
#include <random>

#include "CL/cl.h"
#include "cl2.hpp"

#include "utils.h"
#include "OpenclHelpers.h"
#include "ShuffleOperations.h"
#include "SimpleOperations.h"

// TODO: Error handling
int main(int argc, char* argv[])
{
  if(argc != 3)
  {
    printf("Usage: OpenclRunner.exe filename mode");
    return 0;
  }

  const std::string filename(argv[1]);
  const auto openclFileFullPath= std::filesystem::current_path().append(filename);
  const OpenclFileType openclFileType = openclFileFullPath.extension() == ".cl" ? OpenclFileType::Code : OpenclFileType::Binary;

  const int modeIndex = std::stoi(std::string(argv[2]));

  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint64_t> distr64;
  std::uniform_int_distribution<uint32_t> distr32;

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    const cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::Program program;
    if(openclFileType == OpenclFileType::Code)
    {
      program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    }
    else
    {
      program = CreateOpenclProgramFromBinary(openclFileFullPath, context, device);
    }
    const cl::Kernel kernel = CreateOpenclKernel(program, KERNEL_NAME);


    switch(modeIndex)
    {
      case 1:
      case 2:
      {
        std::array<cl_ulong, WORKSIZE> a{}, b{}, c{};

        for(size_t i = 0; i < WORKSIZE; i++)
        {
          //a[i] = highBytesBase + 1024 + i;
          //b[i] = highBytesBase + 128 + i;
          a[i] = distr64(eng);
          b[i] = distr64(eng);
        }

        cl_int status = CL_SUCCESS;

        const cl::Buffer aBuffer(context, CL_MEM_READ_ONLY, sizeof(a), nullptr, &status);
        CheckOpenclCall(status, "clCreateBuffer aBuffer");

        const cl::Buffer bBuffer(context, CL_MEM_READ_ONLY, sizeof(b), nullptr, &status);
        CheckOpenclCall(status, "clCreateBuffer bBuffer");

        const cl::Buffer cBuffer(context, CL_MEM_WRITE_ONLY, sizeof(c), nullptr, &status);
        CheckOpenclCall(status, "clCreateBuffer cBuffer");

        ExecuteSimpleOperationsKernel(commandQueue, kernel, aBuffer, bBuffer, cBuffer, a, b, c, modeIndex);

        break;
      }
      case 3:
      {
        std::array<cl_uint, WORKSIZE> a{}, b{};

        for (size_t i = 0; i < WORKSIZE; i++)
        {
          //a[i] = highBytesBase + 1024 + i;
          //b[i] = highBytesBase + 128 + i;
          a[i] = distr32(eng);
          b[i] = distr32(eng);
        }

        cl_int status = CL_SUCCESS;

        const cl::Buffer aBuffer(context, CL_MEM_READ_ONLY, sizeof(a), nullptr, &status);
        CheckOpenclCall(status, "clCreateBuffer aBuffer");

        const cl::Buffer bBuffer(context, CL_MEM_WRITE_ONLY, sizeof(b), nullptr, &status);
        CheckOpenclCall(status, "clCreateBuffer bBuffer");

        ExecuteShuffleOperationsKernel(commandQueue, kernel, aBuffer, bBuffer, a, b);

        break;
      }
      default:
      {
        printf("mode index %d is unknown.", modeIndex);
        break;
      }
    }
  }
  catch(std::exception &e)
  {
    printf("Error during execution: %s\n", e.what());
  }
  return 0;
}
