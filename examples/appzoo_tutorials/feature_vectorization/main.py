from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.core import PredictorManager
from easynlp.utils import initialize_easynlp, get_args

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()

    if args.mode == "predict":
        predictor = get_application_predictor(
            app_name=args.app_name, model_dir=args.checkpoint_dir,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            sequence_length=args.sequence_length)
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
