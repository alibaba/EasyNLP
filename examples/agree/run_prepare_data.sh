if [ ! -f ./tmp/datasets/flickr30k_images.tgz ]; then
    wget -P ./tmp 
    tar zxvf ./tmp/datasets/flickr30k_images.tgz -C ./tmp
fi

if [ ! -f ./tmp/VG.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion_kb.tgz
    tar zxvf ./tmp/VG.tgz -C ./tmp
fi

if [ ! -f ./tmp/datasets/flickr30k_images/flickr30k_visual_grounding.224.npz ]; then
    wget -P ./tmp/datasets/flickr30k_images/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion_kb.tgz
fi

if [ ! -f ./tmp/pretrained_models/ViT-L-14.pt ]; then
    wget -P ./tmp/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/agree/openai/ViT-B-32.pt
fi
