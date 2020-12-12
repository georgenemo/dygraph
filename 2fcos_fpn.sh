export FLAGS_allocator_strategy=auto_growth

job_name=fcos_r50_fpn_1x_coco_debug
config=configs/${job_name}.yml
log_dir=log_dir/${job_name}
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${work_dir}
# 1. training
# python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpu 0,1 tools/train.py -c ${config} --eval 2>&1 | tee ${log_dir}/log_train_${now}.txt
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpu 0,1 tools/train.py -c ${config} --eval #&> ${job_name}.log & 
CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} -o use_gpu=true --eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=${log_dir}/best_model --output_eval ${log_dir}
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=fcos_dygraph.pdparams --output_eval ${log_dir}

# 3. infer
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o use_gpu=true weights=fcos_dygraph.pdparams --infer_img=demo/000000014439.jpg
