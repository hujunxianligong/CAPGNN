for i in $(seq 1 10)
do
  python -u main_capgnn.py pubmed \
    --model_name $1 \
    --gpu_ids $2 \
    --lr 2e-1 \
    --l2_coef 2e-3 \
    --cl_coef 1.0 \
    --input_drop_rate 0.1 \
    --dense_drop_rate 0.15 \
    --edge_drop_rate 0.1 \
    --coef_att_drop_rate 0.3 \
    --num_iters 10 \
    --alpha 0.2 \
    --num_views 8 \
    --temp 1.0 \
    --bn
done