#include "ShuffleOperations2.h"
#include "fmt/core.h"

void ExecuteShuffleOperations2Kernel(
  cl::CommandQueue commandQueue,
  const cl::Kernel kernel,
  cl::Buffer aBuffer,
  std::array<cl_ulong, 8 * 128> a
)
{
  CheckOpenclCall(cl::copy(commandQueue, a.begin(), a.end(), aBuffer), "copy to aBuffer from host");
  std::array<cl_ulong, 8 * 128> b{}, bTest{};

  cl_int status = CL_SUCCESS;
  cl::KernelFunctor<cl::Buffer> kernelFunctor(kernel);
  kernelFunctor(
    cl::EnqueueArgs(
      commandQueue,
      cl::NDRange(32, 1, 8),
      cl::NDRange(32, 1, 8)
    ),
    aBuffer,
    status
  );
  CheckOpenclCall(status, "Kernel execution");
  CheckOpenclCall(cl::copy(commandQueue, aBuffer, b.begin(), b.end()), "copy from aBuffer to host");

  for(size_t gid = 0; gid < 8; gid++)
  {
    const size_t startIdx = gid * 128;
    for(size_t k = 0; k < 4; k++)
    {
      const size_t startIdx2 = startIdx + k * 32;
      for(size_t i = 0; i < 32; i++)
      {
        bTest[startIdx2 + i] = a[startIdx2 + lindexes0[i][k]];
      }
    }
  }

  size_t numberOfErrors = 0;
  for(size_t i = 0; i < 8 * 128; i++)
  {
    const std::string message = fmt::format(
      "Index: {0}. Expected value: {1:#018x}. Actual value: {2:#018x}. {3}\n",
      i,
      bTest[i],
      b[i],
      b[i] == bTest[i] ? "PASS" : "FAIL!!!!!!!!!!!!!!!!!!!!!"
    );
    numberOfErrors += b[i] != bTest[i];
    printf(message.c_str());
  }

  printf(fmt::format("Number of invalid values: {0}\n", numberOfErrors).c_str());
}
