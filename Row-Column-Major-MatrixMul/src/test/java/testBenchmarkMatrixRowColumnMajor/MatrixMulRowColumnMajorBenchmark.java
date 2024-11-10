package testBenchmarkMatrixGPU;

import org.jocl.*;

import static sun.jvm.hotspot.debugger.win32.coff.DebugVC50X86RegisterEnums.CL;

public class MatrixMulGPUBenchmark {

    // OpenCL kernel for matrix multiplication
    private static final String programSource =
            "__kernel void matrixMul(__global const float *A, __global const float *B, __global float *C, int N) {" +
                    "    int row = get_global_id(1);" +
                    "    int col = get_global_id(0);" +
                    "    float sum = 0;" +
                    "    for (int k = 0; k < N; k++) {" +
                    "        sum += A[row * N + k] * B[k * N + col];" +
                    "    }" +
                    "    C[row * N + col] = sum;" +
                    "}";

    public static void main(String[] args) {
        int N = 4; // Matrix size (N x N)

        // Initialize matrices A, B, and C
        float[] A = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };

        float[] B = {
                16, 15, 14, 13,
                12, 11, 10, 9,
                8, 7, 6, 5,
                4, 3, 2, 1
        };

        float[] C = new float[N * N]; // Result matrix

        // Perform matrix multiplication on the GPU
        matrixMultiplyGPU(A, B, C, N);

        // Print the result matrix C
        System.out.println("Matrix C (Result):");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(C[i * N + j] + " ");
            }
            System.out.println();
        }
    }

    // Method to perform matrix multiplication on the GPU using OpenCL
    public static void matrixMultiplyGPU(float[] A, float[] B, float[] C, int N) {
        // Initialize the OpenCL platform
        CL.setExceptionsEnabled(true);
        int platformIndex = 0;
        int deviceType = CL.CL_DEVICE_TYPE_ALL;
        int deviceIndex = 0;

        // Obtain platform and device information
        CLPlatform[] platforms = CLPlatform.listCLPlatforms();
        CLPlatform platform = platforms[platformIndex];
        CLDevice device = platform.listCLDevices(deviceType)[deviceIndex];

        // Create a context and a command queue
        CLContext context = CLContext.create(platform, device);
        CLCommandQueue queue = CLCommandQueue.create(context, device);

        // Allocate memory for matrices A, B, and C on the device
        CLMem memA = CLMem.createReadOnly(context, Sizeof.cl_float * N * N);
        CLMem memB = CLMem.createReadOnly(context, Sizeof.cl_float * N * N);
        CLMem memC = CLMem.createWriteOnly(context, Sizeof.cl_float * N * N);

        // Write the data for matrices A and B to the device
        clEnqueueWriteBuffer(queue, memA, CL.CL_TRUE, 0, Sizeof.cl_float * N * N, Pointer.to(A), 0, null, null);
        clEnqueueWriteBuffer(queue, memB, CL.CL_TRUE, 0, Sizeof.cl_float * N * N, Pointer.to(B), 0, null, null);

        // Create the OpenCL program and kernel
        CLProgram program = CLProgram.create(context, programSource);
        CLKernel kernel = CLKernel.create(program, "matrixMul");

        // Set kernel arguments
        kernel.setArg(0, memA);
        kernel.setArg(1, memB);
        kernel.setArg(2, memC);
        kernel.setArg(3, N);

        // Define the global and local work size
        long[] globalWorkSize = new long[]{N, N};

        // Execute the kernel
        clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, null, 0, null, null);

        // Read the result back into matrix C
        clEnqueueReadBuffer(queue, memC, CL.CL_TRUE, 0, Sizeof.cl_float * N * N, Pointer.to(C), 0, null, null);

        // Release OpenCL resources
        clReleaseMemObject(memA);
        clReleaseMemObject(memB);
        clReleaseMemObject(memC);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
}
