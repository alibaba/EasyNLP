If you need to use Megatron models such as GLM, particular packages are required.
```bash
$ cd EasyNLP
$ pip install -r requirements_glm.txt -i http://mirrors.aliyun.com/pypi/simple/
$ pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ git clone https://github.com/NVIDIA/apex
$ cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
$ rm -rf apex
```