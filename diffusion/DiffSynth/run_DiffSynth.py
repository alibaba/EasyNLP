import os, sys, cv2, torch, json
from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel
from DiffSynth.controlnet_processors import (
    ControlnetImageProcesserHED,
    ControlnetImageProcesserShuffle,
    ControlnetImageProcesserDepth,
    ControlnetImageProcesserTile,
    VideoControlnetImageProcesserOpenpose
)
from DiffSynth.pipeline import VideoStylizingPipeline
from DiffSynth.smoother import PySynthSmoother
from DiffSynth.utils import save_images, save_video, read_video_from_video, read_video_from_images


cv2.setNumThreads(16)
torch.set_num_threads(16)


with open(sys.argv[1]) as f:
    config = json.load(f)

controlnet = []
for model_id in config["controlnet_model_id"]:
    controlnet.append(ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16))

local = False
controlnet_processor = []
for processor_name in config["controlnet_processor"]:
    if processor_name=="ControlnetImageProcesserHED":
        processor = ControlnetImageProcesserHED("../models/Annotators" if local else "lllyasviel/Annotators")
    elif processor_name=="ControlnetImageProcesserShuffle":
        processor = ControlnetImageProcesserShuffle()
    elif processor_name=="ControlnetImageProcesserDepth":
        processor = ControlnetImageProcesserDepth("../models/Intel/dpt-large" if local else "Intel/dpt-large")
    elif processor_name=="ControlnetImageProcesserTile":
        processor = ControlnetImageProcesserTile()
    elif processor_name=="VideoControlnetImageProcesserOpenpose":
        processor = VideoControlnetImageProcesserOpenpose.from_pretrained("../models/Annotators" if local else "lllyasviel/Annotators")
    else:
        raise ValueError("unkonwn controlnet_processor")
    controlnet_processor.append(processor)


if len(controlnet)!=len(controlnet_processor):
    raise ValueError(f"controlnets ({len(controlnet)}) and processors ({len(controlnet_processor)}) error")


smoother = None
if config["smoother"] == "PySynthSmoother":
    smoother = PySynthSmoother(**config["smoother_config"])

pipe = VideoStylizingPipeline.from_pretrained(
    config["model_id"],
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

frames = read_video_from_video(config["input_video"], height=config["frame_height"], width=config["frame_width"])
frames = frames[::config["frame_interval"]][:config["frame_maximum_count"]]
print(f"{len(frames)} frames will be rendered")

if "style_image" in config:
    style_image = Image.open(config["style_image"]).resize((config["frame_width"], config["frame_height"]))
else:
    style_image = None

prompt = config["prompt"]
negative_prompt = config["negative_prompt"]

frames_reference = []
for frame_reference in config["frames_reference"]:
    frames_reference.append(Image.open(frame_reference).resize((config["frame_width"], config["frame_height"])))

controlnet_frames = []
for processor_name in config["controlnet_processor"]:
    if processor_name == "ControlnetImageProcesserShuffle":
        controlnet_frames.append([style_image] * len(frames))
    else:
        controlnet_frames.append(frames)

controlnet_frames_reference = []
for processor_name in config["controlnet_processor"]:
    if processor_name == "ControlnetImageProcesserShuffle":
        controlnet_frames_reference.append([style_image] * len(frames_reference))
    else:
        controlnet_frames_reference.append(frames_reference)

results = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    frames=frames,
    frames_reference=frames_reference,
    controlnet_processers=controlnet_processor,
    controlnet_frames=controlnet_frames,
    controlnet_frames_reference=controlnet_frames_reference,
    num_inference_steps=config["num_inference_steps"],
    seed_of_noise=config["seed_of_noise"],
    combine_pattern=config["combine_pattern"],
    controlnet_scale=config["controlnet_scale"],
    img2img_strength=config["img2img_strength"],
    flow_frames=frames,
    smoother=smoother,
    ignore_smoother_steps=config["ignore_smoother_steps"],
    smoother_interval=config["smoother_interval"]
)
rendered_frames = results["images"]

if "post_smoother" in config:
    if config["post_smoother"] == "PySynthSmoother":
        post_smoother = PySynthSmoother(**config["post_smoother_config"])
    post_smoother.prepare(frames)
    rendered_frames = post_smoother.smooth(rendered_frames)

os.makedirs(config["output_path"], exist_ok=True)
save_images(rendered_frames, os.path.join(config["output_path"], "frames"))
save_video(rendered_frames, os.path.join(config["output_path"], "video_output.mp4"))
with open(os.path.join(config["output_path"], "config.json"), "w") as file:
    json.dump(config, file, indent = 4)
