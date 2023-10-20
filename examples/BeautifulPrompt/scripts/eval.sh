python eval.py \
    --method raw \
    --num_inference_steps 20 \
    --data_path data/test.json \
    --result_path outputs/eval-results-raw.json \
    --imgs_path data/imgs/raw/ \
    --disable_tqdm

python eval.py \
    --method magic-prompt \
    --num_inference_steps 20 \
    --data_path data/test.json \
    --result_path outputs/eval-results-magic-prompt.json \
    --imgs_path data/imgs/magic-prompt/ \
    --disable_tqdm

python eval.py \
    --method chatgpt \
    --num_inference_steps 20 \
    --data_path data/test.json \
    --result_path outputs/eval-results-chatgpt.json \
    --imgs_path data/imgs/chatgpt/ \
    --disable_tqdm

python eval.py \
    --method beautiful-prompt \
    --num_inference_steps 20 \
    --data_path data/test.json \
    --result_path outputs/eval-results-beautiful-prompt-sft.json \
    --imgs_path data/imgs/beautiful-prompt-sft/ \
    --model_path outputs/sft \
    --disable_tqdm

python eval.py \
    --method beautiful-prompt \
    --num_inference_steps 20 \
    --data_path data/test.json \
    --result_path outputs/eval-results-beautiful-prompt.json \
    --imgs_path data/imgs/beautiful-prompt/ \
    --model_path outputs/ppo/checkpoint_5000 \
    --disable_tqdm
