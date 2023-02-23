if [ ! -f ./tmp/datasets/flickr30k_images.tgz ]; then
    wget -P ./tmp/datasets https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/flickr30k_images.tgz
    tar zxvf ./tmp/datasets/flickr30k_images.tgz -C ./tmp/datasets
fi

if [ ! -f ./tmp/VG.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/VG.tgz
    tar zxvf ./tmp/VG.tgz -C ./tmp
fi

if [ ! -f ./tmp/datasets/flickr30k_images/flickr30k_visual_grounding.224.npz ]; then
    wget -P ./tmp/datasets/flickr30k_images/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/flickr30k_visual_grounding.224.npz
fi

if [ ! -f ./tmp/pretrained_models/ViT-L-14.pt ]; then
    wget -P ./tmp/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/openai/ViT-L-14.pt
fi
