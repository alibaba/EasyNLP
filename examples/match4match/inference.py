import Match4Match as m4m
import torch, argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coarse_model_path",
        type=str,
        default="",
        nargs="?",
        help="coarse model path",
    )
    parser.add_argument(
        "--fine_model_path",
        type=str,
        default="",
        nargs="?",
        help="fine model path",
    )
    parser.add_argument(
        "--data_table_path",
        type=str,
        default="",
        nargs="?",
        help="data table path",
    )
    parser.add_argument(
        "--video_frame_path",
        type=str,
        default="",
        nargs="?",
        help="data table path",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=12,
        help="number of frames per video",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=30,
        help="the number of videos obtained by coarse-grain model",
    )
    parser.add_argument(
        "--inference_mode",
        type=int,
        default=3,
        help="inference mode, 1: Fast Vector Retrieval Mode, 2: Fine-grained Alignment Mode, 3: Flow-style Matching Mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        nargs="?",
        help="training device",
    )
    opt = parser.parse_args()

    COARSE_MODEL_PATH = opt.coarse_model_path
    FINE_MODEL_PATH = opt.fine_model_path
    DATA_TABLE_PATH = opt.data_table_path
    VIDEO_FRAMES_PATH = opt.video_frame_path
    MAX_FRAMES = opt.max_frames
    BATCH_SIZE = opt.batch_size
    K = opt.K
    INFERENCE_MODE = opt.inference_mode
    device = torch.device(opt.device)

    # load data
    data_table = pd.read_csv(DATA_TABLE_PATH)
    text_list, video_id_list = data_table["text"].tolist(), data_table["video_id"].tolist()
    # load model
    coarse_model = m4m.CoarseMatchModel(data_folder=VIDEO_FRAMES_PATH, max_frames=MAX_FRAMES, device=device)
    coarse_model.load_state_dict(torch.load(COARSE_MODEL_PATH, map_location=device))
    fine_model = m4m.MatchModel(data_folder=VIDEO_FRAMES_PATH, max_frames=MAX_FRAMES, device=device)
    fine_model.load_state_dict(torch.load(FINE_MODEL_PATH, map_location=device))
    # retrieval
    tvr = m4m.TextVideoRetrievaler(device=device)
    tvr.prepare(coarse_model, fine_model, video_id_list, batch_size=BATCH_SIZE)
    retrieval_result = tvr.query(text_list, topk=K, batch_size=BATCH_SIZE, inference_mode=INFERENCE_MODE)
    tvr.analyze_result(retrieval_result, text_list, video_id_list)
