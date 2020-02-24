__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void testKernel(__global ulong *A, __global ulong *B, __global ulong *C)
{
  //Get the work-item's unique ID
  int idx = get_global_id(0);
  //Calc result
  C[idx] = A[idx] + B[idx];
}