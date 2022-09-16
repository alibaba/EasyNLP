# Preprocess
Before use this shell, you should make sure your graphics card works and confire that your enviroment contains the following packages: transformer, ltp and torch.
```
pip install -U ltp ltp-core ltp-extension -i https://pypi.org/simple
bash run_local_preprocess.sh
```
This shell `run_local_preprocess.sh` will download little original data automatically. Then, it will preporcess these data with ltp and a specific mask strategy.