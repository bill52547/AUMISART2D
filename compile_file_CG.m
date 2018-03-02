clc
mexcuda -Iinclude src/mex_CG.cu src/CG_A_operation.cu src/cu_add.cu src/cu_deform.cu ...
    src/cu_initial.cu src/cu_projection.cu src/cu_backprojection.cu src/forwardTV.cu ...
    src/invertTV.cu src/cu_vector_multiply.cu -outdir bin 
% cd(currentpath)


% mexcuda -Iinclude src/mex_backprojection.cu src/cu_backprojection.cu -outdir bin