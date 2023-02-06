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

# # general
generator = pipeline('chinese-ldm-general',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
data = ['绿色外套搭配蓝色牛仔裤']

# # fashion
# generator = pipeline('chinese-ldm-fashion',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['黄色连帽卫衣']

# # art
# generator = pipeline('chinese-ldm-art',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['有群牛羊在吃草']

# # poem
# generator = pipeline('chinese-ldm-poem',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['远上寒山石径斜，白云生处有人家']

# # anime
# generator = pipeline('chinese-ldm-anime',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['粉色头发，穿裙子的少女']

# # pet
# generator = pipeline('chinese-ldm-pet',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['一只黄色的猫']

# # food
# generator = pipeline('chinese-ldm-food',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['小炒黄牛肉']

# sdm
# generator = pipeline('stable-diffusion-general',pipeline_params={"n_samples":1,"do_sr":True,'not_use_gradient_checkpoint':True})
# data = ['an astronaut, high quality']

##修改采样图片数量及采样步长
# generator.reset(n_samples=2,sample_steps=50)
# 生成结果
result=generator(data)
for one_prompt in result:
    # print(one_prompt)
    for idx,one_image_tensor_raw in enumerate(one_prompt['image_tensor']):
        one_image_tensor = 255. * rearrange(one_image_tensor_raw.cpu().numpy(), 'c h w -> h w c')
        pil_image=Image.fromarray(one_image_tensor.astype(np.uint8))
        # 保存图片
        pil_image.save(os.path.join('./',one_prompt['text']+f"_{idx:04}.png"))
        #生成base64
        # b64_image=image_to_base64(pil_image)
        # print(b64_image)
