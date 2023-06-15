import torch,os
import json ,tqdm
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
def preprocess(text):
    return text.replace("\n", "_")
def postprocess(text):
    return text.replace("_", "\n")

def answer_fn(text,type,tokenizer,model,sample=False, top_p=0.6):
    '''sample：是否抽样。生成任务，可以设置为True;
         top_p：0-1之间，生成的内容越多样、
    '''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 

    if not sample: # 不进行采样
        out = model.generate(**encoding,type_label=type,return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else: # 采样（生成）
        out = model.generate(**encoding,type_label=type,return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)

    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])    

def forward_predict(source_file,target_file,tokenizer,model,select_top=100):
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

        predict_answer=answer_fn(input_string,[type],tokenizer,model) # use label
        json_string_predict={"target":predict_answer.strip(),"type":type}
        json_string_predict=json.dumps(json_string_predict,ensure_ascii=False)
        target_object.write(json_string_predict+"\n")