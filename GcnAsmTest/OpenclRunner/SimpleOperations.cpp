#include "SimpleOperations.h"
#include "fmt/core.h"

void ExecuteSimpleOperationsKernel(
  cl::CommandQueue commandQueue,
  cl::Kernel kernel,
  cl::Buffer aBuffer,
  cl::Buffer bBuffer,
  cl::Buffer cBuffer,
  std::array<cl_ulong, WORKSIZE> a,
  std::array<cl_ulong, WORKSIZE> b,
  std::array<cl_ulong, WORKSIZE> c
)
{
  CheckOpenclCall(cl::copy(commandQueue, a.begin(), a.end(), aBuffer), "copy to aBuffer from host");
  CheckOpenclCall(cl::copy(commandQueue, b.begin(), b.end(), bBuffer), "copy to bBuffer from host");

  /*kernel.setArg(0, aBuffer);
  kernel.setArg(1, bBuffer);
  kernel.setArg(2, cBuffer);*/
  cl_int status = CL_SUCCESS;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> kernelFunctor(kernel);
  kernelFunctor(
    cl::EnqueueArgs(
      commandQueue,
      cl::NDRange(WORKSIZE, 1, 1),
      cl::NDRange(WORKSIZE, 1, 1)
    ),
    aBuffer,
    bBuffer,
    cBuffer,
    status
  );
  CheckOpenclCall(status, "Kernel execution");
  CheckOpenclCall(cl::copy(commandQueue, cBuffer, c.begin(), c.end()), "copy from cBuffer to host");

  std::array<cl_ulong, WORKSIZE> cTest;
  for(size_t i = 0; i < WORKSIZE; i++)
  {
    cTest[i] = a[i] + b[i];
  }

  for(size_t i = 0; i < WORKSIZE; i++)
  {
    const std::string message = fmt::format(
      "Index: {0}. Expected value: {1:#018x}. Actual value: {2:#018x}. {3}\n",
      i,
      cTest[i],
      c[i],
      c[i] == cTest[i] ? "PASS" : "FAIL!!!!!!!!!!!!!!!!!!!!!"
    );
    printf(message.c_str());
  }
}