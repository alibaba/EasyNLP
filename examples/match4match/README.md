# Match4Match: Enhancing Text-Video Retrieval by Maximum Flow with Minimum Cost

Match4Match is a text-video retrieval method based on [CLIP](https://github.com/openai/CLIP) and graph optimization theories.

## Install

requirements.txt:
```
tqdm
ftfy
torch
torchvision
joblib
pillow
regex
pandas
matplotlib
faiss-cpu
ortools
opencv-python-headless
argparse
git+https://github.com/openai/CLIP.git@main#egg=clip
```

```
pip install -r requirements.txt
```

## Data Format

To improve the training and inference efficiency, you have to process the data in the following format:

* `data_table.csv`: a table including all text and the corressponding video id.

```
text,video_id
a person is connecting something to system,video9770
a little girl does gymnastics,video9771
a woman creating a fondant baby and flower,video7020
...
```

* `video_frames`: a folder including the sampled frames from each video. 

```
video_frames
├──video9770.pkl
├──video9771.pkl
├──video7020.pkl
└──...
```

You can use the following code to process a video:

```python
from Match4Match import DataConverter

c = DataConverter()
c.process_video("video9770.mp4", "video9770.pkl")
```

## Training

```
python train.py \
  --model_save_path "asset/models/coarse_grained_model"\
  --model_category "coarse-grained"\
  --data_table_path "asset/data/data_table.csv"\
  --video_frame_path "asset/data/video_frames"
```

## Inference

```
python inference.py \
  --coarse_model_path "asset/models/coarse_grained_model/model.pth"\
  --fine_model_path "asset/models/fine_grained_model/model.pth"\
  --data_table_path "asset/data/data_table.csv"\
  --video_frame_path "asset/data/video_frames"\
  --inference_mode 3
  --K 30
```

The K value influences performance and efficiency. You can obtain better retrieval performance by increasing K, but meanwhile, may increase the time consumption.

## Performance

The performance of Match4Match on MSR-VTT dataset (text-to-video retrieval).

| K    | Inference Mode              | R@1  | R@5  | R@10          | MdR | MnR              |
|------|-----------------------------|------|------|---------------|-----|------------------|
| 5    | Fast Vector Retrieval Mode  | 45.1 | 69.1 | [69.1, 100.0] | 2   | [3.017, 309.169] |
| 5    | Fine-grained Alignment Mode | 49.6 | 69.1 | [69.1, 100.0] | 2   | [2.915, 309.067] |
| 5    | Flow-style Matching Mode    | 50.3 | 69.1 | [69.1, 100.0] | 2   | [2.903, 309.055] |
| 30   | Fast Vector Retrieval Mode  | 45.1 | 69.1 | 81.5          | 2   | [6.7, 90.0]      |
| 30   | Fine-grained Alignment Mode | 50.0 | 73.4 | 83.3          | 2   | [6.0, 89.3]      |
| 30   | Flow-style Matching Mode    | 53.6 | 74.4 | 83.4          | 1   | [5.9, 89.2]      |
| 180  | Fast Vector Retrieval Mode  | 45.1 | 69.1 | 81.5          | 2   | [13.19, 37.76]   |
| 180  | Fine-grained Alignment Mode | 49.9 | 73.5 | 82.7          | 2   | [12.157, 36.727] |
| 180  | Flow-style Matching Mode    | 53.9 | 74.2 | 83.0          | 1   | [12.035, 36.605] |
| 1000*| Flow-Style Matching Mode    | 55.5 | 77.8 | 86.6          | 1   | 9.8              |

*Note that Match4Match can achieve the best performance when K=1000 but we recommend setting K=30 because of efficiency.

For more experimental results, please refer to our paper.
