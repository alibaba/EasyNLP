from modeling_MTA import T5ForConditionalGenerationMTA
from transformers import T5Tokenizer , SwitchTransformersForConditionalGeneration 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch,os
import argparse
import json ,tqdm

def preprocess(text):
    return text.replace("\n", "_")
def postprocess(text):
    return text.replace("_", "\n")

def answer_fn(model_trained,tokenizer,input_string,type,use_type=False):
    '''sample：是否抽样。生成任务，可以设置为True;
         top_p：0-1之间，生成的内容越多样、
    '''
    text = preprocess(input_string)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(torch.device('cuda')) 

    if not use_type: # 不进行采样
        out = model_trained.generate(**encoding,return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else: # 采样（生成）
        out = model_trained.generate(**encoding,type_label=type,return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)


    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])    


def predict_on_test(model_path,source_file,model_trained,tokenizer,select_top,use_type):
    lines=open(source_file,'r').readlines()
    if select_top!=-1: # select_top==-1 -->全量预测；其他值，则选取top值进行预测
        lines=lines[0:select_top] 
        
    print("length of lines:",len(lines))
    result_path = model_path + "/predict_result.json"
    print(result_path)
    target_object=open(result_path,'w')
    for i,line in enumerate(tqdm.tqdm(lines)):
        # print(i,line)
        json_string_right=json.loads(line)
        input_string=json_string_right["input"]
        type=json_string_right["type"]

        predict_answer=answer_fn(model_trained,tokenizer,input_string,[type],use_type)
        json_string_predict={"target":predict_answer.strip(),"type":type}
        json_string_predict=json.dumps(json_string_predict,ensure_ascii=False)
        target_object.write(json_string_predict+"\n")

def predict(model_path,source_file):

    model_trained = T5ForConditionalGenerationMTA.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    device = torch.device('cuda') # cuda
    model_trained.to(device)
    select_top=-1 # TODO 改变select_top的值，使得用一个大的数量，或全量数据

    predict_on_test(model_path=model_path,
                    source_file=source_file,
                    model_trained=model_trained,
                    tokenizer=tokenizer,
                    select_top=select_top,
                    use_type=True)
    print(model_path)
