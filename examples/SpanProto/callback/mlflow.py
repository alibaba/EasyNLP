# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 5:06 pm.
# @Author  : JianingWang
# @File    : MlflowCallback
import os
import logging
import numpy as np
from tool.retrying import retry
from transformers import TrainerCallback
from transformers.integrations import is_mlflow_available
from config import ModelArguments, DataTrainingArguments, TrainingArguments
from collections import OrderedDict
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = logging.getLogger(__name__)


class MLflowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `MLflow <https://www.mlflow.org/>`__.
    """

    def __init__(self, model_args, data_args, training_args):
        import mlflow
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        experiment_name, run_name, mlflow_location, tracking_uri = data_args.task_name, data_args.exp_name, data_args.mlflow_location, data_args.tracking_uri
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        self.experiment_name = experiment_name
        self.mlflow_location = mlflow_location
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._log_artifacts = False
        self._ml_flow = mlflow
        self.best_score = None

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in :class:`~transformers.TrainingArguments`'s ``output_dir`` to the local or remote
                artifact storage. Using it without a remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            if self.mlflow_location:
                tracking_location = os.path.join(self.mlflow_location, 'runs')
                if not os.path.exists(tracking_location):
                    os.makedirs(tracking_location)
                self._ml_flow.set_tracking_uri(tracking_location)
            if self.tracking_uri:
                self._ml_flow.set_tracking_uri(self.tracking_uri)
            experiment = self._ml_flow.get_experiment_by_name(self.experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                if self.mlflow_location:
                    artifact_location = os.path.join(self.mlflow_location, 'artifact')
                    if not os.path.exists(artifact_location):
                        os.makedirs(artifact_location)
                    experiment_id = self._ml_flow.create_experiment(self.experiment_name, artifact_location=artifact_location)
                else:
                    experiment_id = self._ml_flow.create_experiment(self.experiment_name)

            self._ml_flow.start_run(experiment_id=experiment_id, run_name=self.run_name)

            combined_dict = OrderedDict()
            for arg, ori in [[self.data_args, DataTrainingArguments], [self.model_args, ModelArguments], [self.training_args, TrainingArguments]]:
                for k, v in arg.to_dict().items():
                    if (hasattr(ori, k) and getattr(ori, k) == v) or k in ['tracking_uri', 'log_level', 'log_level_replica', 'hub_token', 'push_to_hub_token'] :
                        continue
                    else:
                        combined_dict[k] = v

            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._log_params(dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                else:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. '
                        f"MLflow's log_metric() only accepts float and "
                        f"int types so we dropped this attribute."
                    )
                if args.metric_for_best_model and 'loss' not in args.metric_for_best_model:
                    metric_to_check = args.metric_for_best_model
                    if not metric_to_check.startswith("eval_"):
                        metric_to_check = f"eval_{metric_to_check}"
                    if metric_to_check in logs:
                        operator = np.greater if args.greater_is_better else np.less
                        if not self.best_score:
                            self.best_score = logs[metric_to_check]
                        elif operator(logs[metric_to_check], self.best_score):
                            self.best_score = logs[metric_to_check]
                        metrics['best_score'] = self.best_score

            try:
                self._log_metrics(metrics=metrics, step=state.global_step)
            except Exception as e:
                logger.error('mlflow log metrics error' + str(e))

    @retry(stop_max_attempt_number=5, wait_fixed=1000, after_attempts=lambda x: print(f'log metrics failed, retry {x}'), skip_raise=True)
    def _log_metrics(self, metrics, step):
        self._ml_flow.log_metrics(metrics, step)

    @retry(stop_max_attempt_number=5, wait_fixed=1000, after_attempts=lambda x: print(f'log params failed, retry {x}'), skip_raise=True)
    def _log_params(self, params):
        self._ml_flow.log_params(params)

    @retry(stop_max_attempt_number=5, wait_fixed=1000, after_attempts=lambda x: print(f'log run failed, retry {x}'), skip_raise=True)
    def _end_run(self):
        self._ml_flow.end_run()

    # def on_train_end(self, args, state, control, **kwargs):
    #     if self._initialized and state.is_world_process_zero:
    #         if self._log_artifacts:
    #             logger.info("Logging artifacts. This may take time.")
    #             self._ml_flow.log_artifacts(args.output_dir)

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if self._ml_flow.active_run is not None:
            self._end_run()
