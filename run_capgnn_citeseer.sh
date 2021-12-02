for i in $(seq 1 10)
do
  python -u main_capgnn.py citeseer \
    --model_name $1 \
    --gpu_ids $2 \
    --lr 1e-2 \
    --l2_coef 1e-3 \
    --cl_coef 1.0 \
    --input_drop_rate 0.5 \
    --dense_drop_rate 0.1 \
    --edge_drop_rate 0.0 \
    --coef_att_drop_rate 0.3 \
    --num_iters 10 \
    --alpha 0.1 \
    --num_views 8 \
    --temp 0.4
done