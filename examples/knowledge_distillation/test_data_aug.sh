# data augmentation

mkdir tmp
wget 'http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sentence_classification/data_aug/train_head.tsv'
mv train_head.tsv tmp/

easnlp --app_name=data_augmentation
 --worker_count=1
 --worker_gpu=1
 --mode=predict
 --tables=tmp/train_head.tsv
 --input_schema=index:str:1,sent:str:1,label:str:1
 --first_sequence=sent
 --label_name=label
 --outputs=tmp/train_aug.tsv
 --output_schema=augmented_data
 --checkpoint_dir=_
 --sequence_length=128
 --micro_batch_size=8
 --user_defined_parameters="'pretrain_model_name_or_path': 'bert-small-uncased', 'type': 'mlm_da', 'expansion_rate': 2, 'mask_proportion': 0.1, 'remove_blanks': True, 'append_original': True"
