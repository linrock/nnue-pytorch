# https://github.com/Sopel97/nnue-pytorch/commit/d5d794be84a83a03f30f7fff8c916c829defc568

import torch
import cupy as cp

# the kernels need to be compiled lazily because otherwise it takes up
# space on the default cuda device for some reason
quantmoid_kernels = dict()
def make_quantmoid4_forward_kernel():
    key = 'quantmoid4_forward_kernel'
    if not key in quantmoid_kernels:
        # an approximation of sigmoid(x*4)
        quantmoid4_forward_kernel = cp.RawKernel(r'''
        /*
            This function is an approximation of sigmoid(x*4)
            https://www.desmos.com/calculator/ae6tvvlgmu
            https://godbolt.org/z/Y85Pn9h9Y
        */
        extern "C" __global__
        void quantmoid4_forward(
            const float*   const input,
                  float*   const output,
            const int            total
        ) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= total)
               return;
            const float x = input[i];
            const float v = min(floor(abs(x * 127.0f)), 127.0f) - 127.0f;
            const float vv = floor(v * v / 256.0f);
            const float vvv = x > 0.0f ? 126.0f - vv : vv;
            output[i] = vvv / 127.0f;
        }
        ''',
            'quantmoid4_forward'
        )
        quantmoid4_forward_kernel.compile()
        quantmoid_kernels[key] = quantmoid4_forward_kernel
    return quantmoid_kernels[key]

def _quantmoid4(x):
    assert x.is_contiguous()
    assert x.is_cuda
    assert x.dtype == torch.float32

    kernel = make_quantmoid4_forward_kernel()
    device = x.device
    count = x.numel()
    output = torch.empty(*x.shape, dtype=torch.float32, device=device, requires_grad=False)

    kernel(
        grid=((count + 1023) // 1024,),
        block=(1024,),
        args=(
            x.data_ptr(),
            output.data_ptr(),
            count
        )
    )
    return output

class Quantmoid4(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(torch.sigmoid(input * 4.0))
    return _quantmoid4(input)

  @staticmethod
  def backward(ctx, grad_output):
    sigmoid_output, = ctx.saved_tensors
    return grad_output * (1.0 - sigmoid_output) * 4.0 * sigmoid_output

quantmoid4 = Quantmoid4.apply

if __name__ == '__main__':
    for i in range(-255, 255):
        x = i / 127.0
        print(x, quantmoid4(torch.tensor([x]).cuda()), quantmoid4(torch.tensor([x]).cuda())-torch.sigmoid(torch.tensor([x]).cuda()*4.0))
