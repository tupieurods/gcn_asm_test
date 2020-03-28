#pragma once
#include "OpenclHelpers.h"
#include "utils.h"

void ExecuteSimpleOperationsKernel(
  cl::CommandQueue commandQueue,
  cl::Kernel kernel,
  cl::Buffer aBuffer,
  cl::Buffer bBuffer,
  cl::Buffer cBuffer,
  std::array<cl_ulong, WORKSIZE> a,
  std::array<cl_ulong, WORKSIZE> b,
  std::array<cl_ulong, WORKSIZE> c,
  const int modeIndex
);
