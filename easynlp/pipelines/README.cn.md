# Pipeline [English](https://github.com/alibaba/EasyNLP/tree/master/README.md)
使用`pipelines` 快速测试或部署您的EasyNLP模型。

## 介绍
`pipeline` 函数返回一个实例化的 `Pipeline` 对象处理输入数据. 通常情况下，函数传入的参数为 `app_name` 和 `model_path`。
```python
classifictor = pipeline("text_classify", local_model_path)
```
调用 `get_supported_tasks` 获取所有支持的 `app_name`。
```python
support_app_list = get_supported_tasks()
```
你可以仅指定 `app_name`, `pipeline` 会根据你指定的`app_name` 加载默认模型.
```python
classifictor = pipeline("text_classify")
```
我们提供了一些训练好的模型, 调用函数 `get_app_model_list` 可以获得这些模型的基本信息。当你使用这些模型时，无需输入参数 `app_name` ， `pipeline` 会根据模型的训练设置自动推理`app_name`.
```python
app_model_list = get_app_model_list()
classifictor = pipeline("bert-base-sst2")
```
上面的步骤已经帮我们构建好了一个分类器，现在可以使用这个分类器预测数据了。
```python
data = ["Yucaipa owned Dominick's before selling the chain to Safeway \
        in 1998 for $2.5 billion.",
        "Around 0335 GMT, Tab shares were up 19 cents, or 4.4%, at A$4.56, \
        having earlier set a record high of A$4.57."]
print(classifictor(data))
```
