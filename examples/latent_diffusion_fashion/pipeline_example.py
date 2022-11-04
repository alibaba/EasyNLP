from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from einops import rearrange
import os

# convert image to base64
def image_to_base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer,format='png')
    byte_data = img_buffer.getvalue()
    base64_str = str(base64.b64encode(byte_data), 'utf-8')
    return base64_str

# init pipeline. 
generator = pipeline('latent_diffusion',pipeline_params={"n_samples":2})

# input data
data = ['远处的雪山，表面覆盖着厚厚的积雪','湖边有一片森林']
result=generator(data)

for one_prompt in result:
    # print(one_prompt)
    for idx,one_image_tensor_raw in enumerate(one_prompt['image_tensor']):
        one_image_tensor = 255. * rearrange(one_image_tensor_raw.cpu().numpy(), 'c h w -> h w c')
        pil_image=Image.fromarray(one_image_tensor.astype(np.uint8))
        # 保存图片
        # pil_image.save(os.path.join('./',one_prompt['text']+f"_{idx:04}.png"))
        # 生成base64
        # b64_image=image_to_base64(pil_image)
        # print(b64_image)