import Match4Match as m4m
import torch, os, tqdm, argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="",
        nargs="?",
        help="model save path",
    )
    parser.add_argument(
        "--model_category",
        type=str,
        default="fine-grained",
        nargs="?",
        help='model category, "fine-grained" or "coarse-grained"',
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
        "--B",
        type=int,
        default=0,
        help="a amall batch size. If you do not have enough GPU memory, use a small B (e.g. B=8), otherwise please set B=0.",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=100,
        help="frequency of saving the model",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=1000,
        help="total training steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        nargs="?",
        help="training device",
    )
    opt = parser.parse_args()

    MODEL_SAVE_PATH = opt.model_save_path
    MODEL_CATEGORY = opt.model_category
    DATA_TABLE_PATH = opt.data_table_path
    VIDEO_FRAMES_PATH = opt.video_frame_path
    MAX_FRAMES = opt.max_frames
    BATCH_SIZE = opt.batch_size
    B = opt.B
    SAVE_FREQUENCY = opt.save_frequency
    TRAIN_STEPS = opt.train_steps
    device = torch.device(opt.device)

    # load data
    data_table = pd.read_csv(DATA_TABLE_PATH)
    text_list, video_id_list = data_table["text"].tolist(), data_table["video_id"].tolist()
    # train
    if MODEL_CATEGORY == "coarse-grained":
        model = m4m.CoarseMatchModel(data_folder=VIDEO_FRAMES_PATH, max_frames=MAX_FRAMES, device=device)
    elif MODEL_CATEGORY == "fine-grained":
        model = m4m.MatchModel(data_folder=VIDEO_FRAMES_PATH, max_frames=MAX_FRAMES, device=device)
    else:
        raise Exception("invalid MODEL_CATEGORY")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-3)
    for steps in tqdm.trange(1, TRAIN_STEPS+1):
        index_batch = torch.randint(0, len(text_list), (BATCH_SIZE,))
        text_batch = [text_list[i] for i in index_batch]
        video_id_batch = [video_id_list[i] for i in index_batch]
        if B==0:
            m4m.train_model_small_batch(model, optimizer, text_batch, video_id_batch)
        else:
            if MODEL_CATEGORY == "coarse-grained":
                raise Exception("Now coarse-grained model do not support B!=0")
            m4m.train_model_huge_batch(model, optimizer, text_batch, video_id_batch, batch_size=B, device=device)
        if steps%SAVE_FREQUENCY==0:
            save_path = os.path.join(MODEL_SAVE_PATH, f"model_step_{steps}.pth")
            torch.save(model.state_dict(), save_path)
