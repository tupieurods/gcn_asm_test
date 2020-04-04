#include "ShuffleOperations.h"
#include "fmt/core.h"

void ExecuteShuffleOperationsKernel(
  cl::CommandQueue commandQueue,
  cl::Kernel kernel,
  cl::Buffer aBuffer,
  cl::Buffer bBuffer,
  std::array<cl_uint, WORKSIZE> a,
  std::array<cl_uint, WORKSIZE> b
)
{
  CheckOpenclCall(cl::copy(commandQueue, a.begin(), a.end(), aBuffer), "copy to aBuffer from host");

  cl_int status = CL_SUCCESS;
  cl::KernelFunctor<cl::Buffer, cl::Buffer> kernelFunctor(kernel);
  kernelFunctor(
    cl::EnqueueArgs(
      commandQueue,
      cl::NDRange(WORKSIZE, 1, 1),
      cl::NDRange(WORKSIZE, 1, 1)
    ),
    aBuffer,
    bBuffer,
    status
  );
  CheckOpenclCall(status, "Kernel execution");
  CheckOpenclCall(cl::copy(commandQueue, bBuffer, b.begin(), b.end()), "copy from bBuffer to host");

  std::array<cl_uint, WORKSIZE> bTest;
  for(size_t i = 0; i < WORKSIZE; i++)
  {
    bTest[i] = a[ShuffleIndexes1[i]];
  }

  for(size_t i = 0; i < WORKSIZE; i++)
  {
    const std::string message = fmt::format(
      "Index: {0}. Expected value: {1:#018x}. Actual value: {2:#018x}. {3}\n",
      i,
      bTest[i],
      b[i],
      b[i] == bTest[i] ? "PASS" : "FAIL!!!!!!!!!!!!!!!!!!!!!"
    );
    printf(message.c_str());
  }
}
