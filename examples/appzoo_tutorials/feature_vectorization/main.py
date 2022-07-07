from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.core import PredictorManager
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)

    if args.mode == "predict":
        predictor = get_application_predictor(
            app_name=args.app_name, model_dir=args.checkpoint_dir,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            sequence_length=args.sequence_length,
            user_defined_parameters=user_defined_parameters)
        predictor_manager = PredictorManager(
            predictor=predictor,
            input_file=args.tables.split(",")[-1],
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size
        )
        predictor_manager.run()
