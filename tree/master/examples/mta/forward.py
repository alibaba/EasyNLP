from modeling_MTA import T5ForConditionalGenerationMTA
from transformers import T5Tokenizer  
from transformers import T5Tokenizer
import torch,os
import argparse
import json ,tqdm

def preprocess(text):
    return text.replace("\n", "_")
def postprocess(text):
    return text.replace("_", "\n")

def answer_fn(text,type,use_type=False, top_p=0.2):
    '''sample：是否抽样。生成任务，可以设置为True;
         top_p：0-1之间，生成的内容越多样、
    '''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 

    if not use_type: # 不进行采样
        out = model_trained.generate(**encoding,return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else: # 采样（生成）
        out = model_trained.generate(**encoding,type_label=type,return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)


    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])    


def predict_on_test(source_file,target_file,select_top,use_type):
    lines=open(source_file,'r').readlines()
    if select_top!=-1: # select_top==-1 -->全量预测；其他值，则选取top值进行预测
        lines=lines[0:select_top] 
    print("length of lines:",len(lines))
    target_object=open(target_file,'w')
    for i,line in enumerate(tqdm.tqdm(lines)):
        # print(i,line)
        json_string_right=json.loads(line)
        input_string=json_string_right["input"]
        target_answer=json_string_right["target"]
        type=json_string_right["type"]

        predict_answer=answer_fn(input_string,[type],use_type)
        json_string_predict={"target":predict_answer.strip(),"type":type}
        json_string_predict=json.dumps(json_string_predict,ensure_ascii=False)
        target_object.write(json_string_predict+"\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/apsarapangu/disk1/xieyukang.xyk/project/models/result_model")
    args = parser.parse_args()

    model_path =args.model_path

    model_trained = T5ForConditionalGenerationMTA.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)


    device = torch.device('cuda') # cuda
    model_trained.to(device)
    select_top=-1 # TODO 改变select_top的值，使得用一个大的数量，或全量数据

    source_file = "/apsarapangu/disk1/xieyukang.xyk/project/data/clean_data_ori/english_val_new_shuffle.json"
    target_file= os.path.join(model_path,"nosample_result_trainded.json")
    predict_on_test(source_file,target_file,select_top,use_type=True)
    print(model_path)
