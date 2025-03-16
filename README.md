# Learn Triton

I am currently learning Triton to build an optimised BERT model [(cuBERT)](https://github.com/divusree/cuBERT). The machine I'm working on is RTX 3060 TI.


## Kernels added so far

- `addition/addition.py`: Has 1D and 2D matrix addition `add_1D` and `add_2D` respectively. 
- `addition/sumvec.py`: adds all the elements in a 1D vector with `sum1D`.
- `multiplication/multiplication.py`: performs matrix multiplication between 2 2D matrices with `matmul`. The kernel `group_matmul_kernel` is for matmul with super-grouped block matmul (and col major ordering). works in both fp16 and fp32. The kernel `matmul_kernel_naive` does ungrouped block matmul (naive).
- `multiplication/batch_matmul.py`: implements (B, M, K) x (B, K, N) tensor multiplication. basically extending the group_matmul_kernel for 3D matrices. Major learning was multi dimensional offsetting is easy (additive), but the program ids begin at 1 for rows and 2 for columns. program id 0 is not used at all.  
- `softmax/row_max.py`:  for a (M, N) matrix, the kernel `rmax_kernel` gets the maximum value in each row block by block.
- `softmax/softmax.py`:  for a (M, N) matrix, the kernel `softmax_kernel` calculates row wise online safe softmax as detailed by [Flash Attention 2](https://arxiv.org/abs/2307.08691). 


## Kernels to add

- Online softmax
- Layer Norm

Misc - need to create a notebook to benchmark the functions against python's native functions.  
