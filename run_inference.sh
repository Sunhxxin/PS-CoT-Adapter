
# rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type clip \
    --bs 4 --eval_bs 1 --eval_acc 10 --output_len 512 \
    --final_eval --prompt_format QCM-LE \
    --evaluate_dir /data/mm-cot-main/experiments/rationale_allenai-unifiedqa-t5-base_clip_QCM-LE_lr5e-05_bs16_op512_ep55\
    --mode inference --use_generate




CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4  --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments
    --evaluate_dir models/mm-cot-large-rationale

# answer inference
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type clip \
    --bs 1 --eval_bs 1 --eval_acc 5 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --eval_le /data/mm-cot-main/models/frozen_semantic_model_rational02/predictions_ans_eval.json \
    --test_le /data/mm-cot-main/models/frozen_semantic_model_rational02/predictions_ans_test.json \
    --evaluate_dir /data/mm-cot-main/models/frozen_semantic_model_rational02_answer\
    --mode inference

