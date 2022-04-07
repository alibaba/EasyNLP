import time
import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score
from easynlp.utils import losses
from easynlp.utils.logger import logger
from easynlp.core.evaluator import Evaluator

class SequenceClassificationEvaluator(Evaluator):

    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)
        self.metrics = ["accuracy", "f1"]

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        logits_list = list()
        y_trues = list()

        total_spent_time = 0.0
        for _step, batch in enumerate(self.valid_loader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop("label_ids")
                outputs = model(batch["input_ids"],batch["attention_mask"],batch["token_type_ids"])
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in outputs
            logits = outputs["logits"]

            y_trues.extend(label_ids.tolist())
            logits_list.extend(logits.tolist())
            hit_num += torch.sum(torch.argmax(logits, dim=-1) == label_ids).item()
            total_num += label_ids.shape[0]

            if len(logits.shape) == 1 or logits.shape[-1] == 1:
                tmp_loss = losses.mse_loss(logits, label_ids)
            elif len(logits.shape) == 2:
                tmp_loss = losses.cross_entropy(logits, label_ids)
            else:
                raise RuntimeError

            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += self.valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" %
                    (_step + 1, len(self.valid_loader.dataset) // self.valid_loader.batch_size))

        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info("Eval loss: {}".format(eval_loss))

        logits_list = np.array(logits_list)
        eval_outputs = list()
        for metric in self.metrics:
            if metric.endswith("accuracy"):
                acc = hit_num / total_num
                logger.info("Accuracy: {}".format(acc))
                eval_outputs.append(("accuracy", acc))
            elif metric == "f1":
                if model.config.num_labels == 2:
                    f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1))
                    logger.info("F1: {}".format(f1))
                    eval_outputs.append(("f1", f1))
                else:
                    f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1), average="macro")
                    logger.info("Macro F1: {}".format(f1))
                    eval_outputs.append(("macro-f1", f1))
                    f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1), average="micro")
                    logger.info("Micro F1: {}".format(f1))
                    eval_outputs.append(("micro-f1", f1))
            elif metric == "auc":
                auc = roc_auc_score(y_trues, np.argmax(logits_list, axis=-1))
                logger.info("AUC: {}".format(auc))
                eval_outputs.append(("auc", auc))
            elif metric == "matthews_corrcoef":
                mcc = matthews_corrcoef(y_trues, np.argmax(logits_list, axis=-1))
                logger.info("Matthews Corrcoef: {}".format(mcc))
                eval_outputs.append(("matthews_corrcoef", mcc))
            elif metric == "pearson_and_spearman":
                preds = logits_list[:, 0]
                pearson_corr = pearsonr(preds, y_trues)[0]
                spearman_corr = spearmanr(preds, y_trues)[0]
                logger.info("Peasrson: {}".format(pearson_corr))
                logger.info("Spearmanr: {}".format(spearman_corr))
                corr = (pearson_corr + spearman_corr) / 2.0
                logger.info("Peasrson_and_spearmanr: {}".format(corr))
                eval_outputs.append(("pearson_and_spearman", corr))
            elif metric == "classification_report":
                logger.info("\n{}".format(
                    classification_report(y_trues, np.argmax(logits_list, axis=-1), digits=4)))
            elif "last_layer_mse" in self.metrics:
                logger.info("Last layer MSE: {}".format(eval_loss))
                eval_outputs.append(("last_layer_mse", -eval_loss))
            else:
                raise NotImplementedError("Metric %s not implemented" % metric)

        return eval_outputs