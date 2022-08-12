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

There is an example of text-image generation.

```python
from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO

# init pipeline. The model generates ${max_generated_num} (setting it to 1 by default) images for each text.
generator = pipeline('text2image_generation', max_generated_num = 4)

# convert base64 to image
def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image

# input data
data = ['远处的雪山，表面覆盖着厚厚的积雪']

# model generation
generator = pipeline('text2image_generation')

# save images named after text
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

There is an example of image caption.

```python
from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO

# init pipeline.  The model generates ${max_generated_num} (setting it to 1 by default) captions for each image.
generator = pipeline('image2text_generation', max_generated_num = 4)

# convert image to base64
def image_to_base64(img_path):
    img = Image.open(img_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = str(base64.b64encode(byte_data), 'utf-8')
 
    return base64_str

# input data
data = ['./example.png']

# model generation
data_imgbase64 = [image_to_base64(imgpath) for imgpath in data]
results = generator(data_imgbase64)

# display the predicted captions
for imgpath, result in zip(data, results):
    text_list = result['gen_text']
    print ('imgpath: {}, generated text: {}'.format(imgpath, text_list))
```

