# Pipeline [English](https://github.com/alibaba/EasyNLP/blob/master/easynlp/pipelines/README.md)
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

文图生成的例子

```python
from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO

# 构建pipeline。模型为每个文本生成${max_generated_num}张图片(该值默认为1)
generator = pipeline('text2image_generation', max_generated_num = 4)

# base64转换为图像
def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image

# 输入数据
data = ['远处的雪山，表面覆盖着厚厚的积雪']

# 模型生成
results = generator(data)

# 保存以文本命名的图像
for text, result in zip(data, results):
    imgbase64_str_list = result['gen_imgbase64']
    imgpath_list = []
    for base64_idx in range(len(imgbase64_str_list)):
        imgbase64_str = imgbase64_str_list[base64_idx]
        image = base64_to_image(imgbase64_str)
        imgpath = '{}_{}.png'.format(text, base64_idx)
        image.save(imgpath)
        imgpath_list.append(imgpath)
    print ('text: {}, save generated image: {}'.format(text, imgpath_list))
```

图文生成的例子
```python
from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO

# 构建pipeline。模型为每张图片生成${max_generated_num}个标题(该值默认为1)
generator = pipeline('image2text_generation', max_generated_num = 4)

# 图像转换为base64
def image_to_base64(img_path):
    img = Image.open(img_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = str(base64.b64encode(byte_data), 'utf-8')
 
    return base64_str

# 输入数据
data = ['./example.png']

# 模型生成
data_imgbase64 = [image_to_base64(imgpath) for imgpath in data]
results = generator(data_imgbase64)

# 显示生成的标题
for imgpath, result in zip(data, results):
    text_list = result['gen_text']
    print ('imgpath: {}, generated text: {}'.format(imgpath, text_list))
```

