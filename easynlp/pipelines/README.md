# Pipeline [中文介绍](https://github.com/alibaba/EasyNLP/blob/master/easynlp/pipelines/README.cn.md)
You can use `pipelines` to quickly deploy models trained based on EasyNLP.

## Instruction
The `pipeline` function return a instantiated `Pipeline` object to process input data. Most of the time, you need to specify the `app_name` and `model_path`. 
```python
classifictor = pipeline("text_classify", local_model_path)
```
You can invoke `get_supported_tasks` to get all supported `app_name`.
```python
support_app_list = get_supported_tasks()
```
When you only specify `app_name`, `pipeline` function will loads the default model to build pipeline.
```python
classifictor = pipeline("text_classify")
```
We provide some trained models, call `get_app_model_list` to get information about these models. You do not need to specify `app_name` when using these models, the name is determined automatically by `pipeline`.
```python
app_model_list = get_app_model_list()
classifictor = pipeline("bert-base-sst2")
```
Now, you can use the pipeline to get the prediction about your data.
```python
data = ["Yucaipa owned Dominick's before selling the chain to Safeway \
        in 1998 for $2.5 billion.",
        "Around 0335 GMT, Tab shares were up 19 cents, or 4.4%, at A$4.56, \
        having earlier set a record high of A$4.57."]
print(classifictor(data))
```
