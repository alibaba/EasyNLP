from easynlp.core import Trainer
from easynlp.appzoo import ClassificationDataset, SequenceClassification
from easynlp.utils import initialize_easynlp

args = initialize_easynlp()

train_dataset = ClassificationDataset(
    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    data_file=args.tables,
    max_seq_length=args.sequence_length,
    input_schema=args.input_schema,
    first_sequence=args.first_sequence,
    label_name=args.label_name,
    label_enumerate_values=args.label_enumerate_values,
    is_training=True)

model = SequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path)
Trainer(model=model,  train_dataset=train_dataset).train()
