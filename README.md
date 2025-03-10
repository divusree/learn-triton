# Learn Triton

I am currently learning Triton to build an optimised BERT model [(cuBERT)](https://github.com/divusree/cuBERT). The machine I'm working on is RTX 3060 TI.


## Kernels added so far

- `addition/addition.py`: Has 1D and 2D matrix addition `add_1D` and `add_2D` respectively. 
- `addition/sumvec.py`: adds all the elements in a 1D vector with `sum1D`.
- `multiplication/multiplication.py`: performs matrix multiplication between 2 2D matrices with `matmul`. 

## Kernels to add

- Extend Matrix multiplication to handle batches of matrices, ref: [(pytorch doc)](https://pytorch.org/docs/stable/generated/torch.bmm.html) 
- Matrix transpose
- Online softmax
- Layer Norm

Misc - need to create a notebook to benchmark the functions against python's native functions.  
