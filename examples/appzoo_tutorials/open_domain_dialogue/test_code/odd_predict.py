import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.appzoo import OpenDomainDialoguePredictor, OpenDomainDialogue

if __name__ == '__main__':
    initialize_easynlp()
    args = get_args()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)

    if args.mode == "predict":
        predictor = OpenDomainDialoguePredictor(
            model_dir=args.checkpoint_dir,
            model_cls=OpenDomainDialogue,
            data_dir=args.tables,
            user_defined_parameters=user_defined_parameters
        )
        predictor.run()
        exit()