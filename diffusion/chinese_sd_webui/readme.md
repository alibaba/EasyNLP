## stable-diffusion-webui-Chinese-diffusion
An extension for [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that lets you generate image from chinese lanuage.

## download model

Download or place the pai diffusion models in `path_to_your/stable-diffusion-webui/models/ChineseDiffusion`. 
For example,
```
cd path_to_your/stable-diffusion-webui/models/ChineseDiffusion/Chinese_diffusion 
git clone https://huggingface.co/alibaba-pai/pai-diffusion-artist-large-zh 
cd path_to_your/stable-diffusion-webui/models/ChineseDiffusion/Chinese_Controlnet 
git clone https://huggingface.co/alibaba-pai/pai-diffusion-artist-large-zh-controlnet-canny
cd path_to_your/stable-diffusion-webui/models/ChineseDiffusion/Chinese_Lora 
git clone https://huggingface.co/alibaba-pai/pai-diffusion-artist-large-zh-lora-poem
```

## Directories tree
```

 📁 webui root directory
┗━━ 📁 stable-diffusion-webui
    ┗━━ 📁 models                            
        ┗━━ 📁 ChineseDiffusion    
            ┗━━ 📁 Chinese_diffusion                              
                ┗━━ 📁 your chinese diffusion model    <----- any name can be used, such as 'pai-diffusion-artist-large-zh‘
                    ┣━━ 📁feature_tractor                
                    ┣━━ 📁safety_checker                     
                    ┣━━ 📁scheduler
                    ┣━━ 📁text_encoder 
                    ┗━━ 📁tokenizer 
                    ┣━━ 📁unet
                    ┗━━ 📁vae
                    model_index.josn
                    README.md
            ┗━━ 📁 Chinese_Lora
                ┗━━ 📁 your lora model    <----- any name can be used, such as 'pai-diffusion-artist-large-zh-lora-poem‘
                    ┣━━ 📁...               

            ┗━━ 📁 Chinese_Controlnet                              
                ┗━━ 📁 your controlnet model   <----- any name can be used, such as 'pai-diffusion-artist-large-zh-controlnet-canny‘
                    ┣━━ 📁...   


```

