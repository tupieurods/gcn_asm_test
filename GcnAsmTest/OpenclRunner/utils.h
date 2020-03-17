#pragma once

static const cl_uint PLATFORM_ID = 1;
static const cl_uint DEVICE_ID = 0;
static const cl_uint WORKSIZE = 64;
static const std::string KERNEL_NAME = "testKernel";

enum class OpenclFileType
{
  Code,
  Binary
};