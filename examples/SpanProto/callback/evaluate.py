# -*- coding: utf-8 -*-
# @Time    : 2022/2/1 12:35 am.
# @Author  : JianingWang
# @File    : evaluate.py
import os
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)


class DoPredictDuringTraining(TrainerCallback):

    def __init__(self, test_dataset, processor):
        super(DoPredictDuringTraining, self).__init__()
        self.test_dataset = test_dataset.remove_columns("label")
        self.processor = processor
        self.best_score = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        if args.metric_for_best_model:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            operator = np.greater if args.greater_is_better else np.less
            if not self.best_score or operator(metrics[metric_to_check], self.best_score):
                self.best_score = metrics[metric_to_check]
                self.do_predict(args.output_dir)

    def do_predict(self, output_dir):
        logits = self.trainer.predict(self.test_dataset, metric_key_prefix="predict").predictions
        if hasattr(self.processor, 'save_result'):
            if self.trainer.is_world_process_zero():
                self.processor.save_result(logits)
        else:
            predictions = np.argmax(logits, axis=1)
            output_predict_file = os.path.join(output_dir, f"predict_results.txt")
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = self.processor.labels[item]
                        writer.write(f"{index}\t{item}\n")
