base_dir=$PWD
wget -P $base_dir https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/parasum/Xsum.zip
unzip Xsum.zip
rm Xsum.zip

nohup python finetune_for_CNewSum/train_matching.py --restore_from ~/model/matchsum_paraphrase_qqp/cnewsum_1w/2022-12-30-14-50-56/epoch-37_step-11500_f1_score-0.796566.pt --save_path /home/moming/model/matchsum_efl_paraphrase_bce/cnewsum_7500qqp --data_path cnewsum/ --gpus 1 >> cenwsum.log 2>&1 &

