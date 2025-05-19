extern "C" __global__ void normalize01(
    float* samples,
    long length,
    float amplitude)
{
    // Phase 1: Finde den maximalen absoluten Wert (Reduktion)
    __shared__ float s_max[256]; // Shared Memory für Block-Reduktion
    float thread_max = 0.0f;

    // Jeder Thread sucht das Maximum in seinem Bereich
    for (long i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < length; 
         i += blockDim.x * gridDim.x)
    {
        float val = fabsf(samples[i]);
        if (val > thread_max) thread_max = val;
    }

    // Block-weise Reduktion (finde Maximum im Block)
    s_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            if (s_max[threadIdx.x + stride] > s_max[threadIdx.x])
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Globales Maximum ermitteln (atomare Operation)
    if (threadIdx.x == 0)
    {
        atomicMax((int*)&s_max[0], __float_as_int(thread_max));
    }
    __syncthreads();

    float global_max = s_max[0];

    // Phase 2: Normalisierung anwenden
    if (global_max == 0.0f) return; // Vermeide Division durch Null

    float scale = amplitude / global_max;

    for (long i = threadIdx.x + blockIdx.x * blockDim.x;
         i < length;
         i += blockDim.x * gridDim.x)
    {
        samples[i] *= scale;
    }
}