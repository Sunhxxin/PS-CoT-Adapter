# rationale generation 使用rational预训练
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-large \
    --user_msg rationale --img_type clip \
    --bs 2 --eval_bs 1 --eval_acc 10 --output_len 512 \
    --epoch 20\
    --final_eval --prompt_format QCM-LE\
    --mode train --use_caption

# rationale generation 接预训练checkpoint

CUDA_VISIBLE_DEVICES=0 python main.py \
    --load_checkpoint /data/mm-cot-main/models/frozen_semantic_model_large_06\
    --model allenai/unifiedqa-t5-large \
    --user_msg rationale --img_type clip \
    --bs 2 --eval_bs 1 --eval_acc 10 --output_len 512 \
    --final_eval --epoch 20\
    --prompt_format QCM-LE\
    --mode train --use_caption

# rationale generation 使用新数据集预训练
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type detr \
    --bs  2--eval_bs 1 --eval_acc 10 --output_len 512 \
    --epoch 1\
    --final_eval --prompt_format Q-E\
    --mode train

# answer inference 接预训练checkpoint
CUDA_VISIBLE_DEVICES=0 python main.py \
    --load_checkpoint /data/mm-cot-main/models/frozen_semantic_model_large_06\
    --model allenai/unifiedqa-t5-large \
    --user_msg answer --img_type clip \
    --bs 4 --eval_bs 1 --eval_acc 10 --output_len 64 \
    --epoch 20\
    --final_eval --prompt_format QCMG-A \
    --mode train --use_caption \
    --eval_le /data/mm-cot-main/experiments/rationale_allenai-unifiedqa-t5-large_clip_QCM-LE_lr5e-05_bs2_op512_ep20_rational06_large/predictions_ans_eval.json \
    --test_le /data/mm-cot-main/experiments/rationale_allenai-unifiedqa-t5-large_clip_QCM-LE_lr5e-05_bs2_op512_ep20_rational06_large/predictions_ans_test.json


CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type clip \
    --bs 4 --eval_bs 1 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --mode train\
    --eval_le /data/mm-cot-main/models/frozen_semantic_model_rational02/predictions_ans_eval.json \
    --test_le /data/mm-cot-main/models/frozen_semantic_model_rational02/predictions_ans_test.json
