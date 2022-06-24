export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

file=$1
# jemalloc_lib=/home/user/yanbing/jemalloc-bin/lib/libjemalloc.so
jemalloc_lib=/home/user/anaconda3/envs/pyg/lib/libjemalloc.so
iomp_lib=/home/user/anaconda3/envs/pyg/lib/libiomp5.so
export LD_PRELOAD="$jemalloc_lib":"$iomp_lib"
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
python -m intel_extension_for_pytorch.cpu.launch \
     --use_default_allocator \
     --node_id=0 \
     --log_path=/home/user/yanbing/pyg/pytorch_geometric/benchmark/kernel \
     --log_file_prefix="./throughput_log" \
     $1
     #ogbn_products_sage.py
     #main_performance.py
