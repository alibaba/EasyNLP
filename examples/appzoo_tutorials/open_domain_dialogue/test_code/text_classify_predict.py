from easynlp.appzoo import get_application_predictor, SequenceClassificationPredictor, SequenceClassification
from easynlp.core import PredictorManager

app_name = 'text_classify'
checkpoint_dir = './classification_model/'
first_sequence = 'sent1'
second_sequence = 'sent2'
sequence_length = 128
user_defined_parameters = {'app_parameters': {}}
tables = 'dev.tsv'
input_schema = 'label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1'
outputs = 'dev.pred.tsv'
output_schema = 'predictions,probabilities,logits,output'
append_cols = 'label'
micro_batch_size = 2

# predictor = get_application_predictor(
#             app_name=app_name, model_dir=checkpoint_dir,
#             first_sequence=first_sequence,
#             second_sequence=second_sequence,
#             sequence_length=sequence_length,
#             user_defined_parameters=user_defined_parameters)
predictor = SequenceClassificationPredictor(
            model_dir=checkpoint_dir,
            model_cls=SequenceClassification,
            user_defined_parameters=user_defined_parameters)

predictor_manager = PredictorManager(
    predictor=predictor,
    input_file=tables.split(",")[-1],
    input_schema=input_schema,
    output_file=outputs,
    output_schema=output_schema,
    append_cols=append_cols,
    batch_size=micro_batch_size
)
predictor_manager.run()
exit()