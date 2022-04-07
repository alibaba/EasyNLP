set -e
set -x

cur_path="/workspace/project_all/EasyNLP/examples/knowledge_distillation"
cd $cur_path

sh ./run_meta_preprocess.sh
sh ./meta_teacher_train.sh
sh ./meta_student_distill.sh