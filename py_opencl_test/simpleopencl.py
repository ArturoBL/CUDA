import pyopencl as cl
import numpy as np

# Crear contexto (elige GPU si hay, si no CPU)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Datos de entrada
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
c = np.empty_like(a)

# Kernel OpenCL (C-like)
kernel_code = """
__kernel void suma(
    __global const float *a,
    __global const float *b,
    __global float *c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# Compilar kernel
program = cl.Program(ctx, kernel_code).build()

# Buffers en GPU
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

# Ejecutar kernel
program.suma(
    queue,
    a.shape,      # número de hilos
    None,
    a_buf,
    b_buf,
    c_buf
)

# Copiar resultado a CPU
cl.enqueue_copy(queue, c, c_buf)

print("Resultado:", c)

