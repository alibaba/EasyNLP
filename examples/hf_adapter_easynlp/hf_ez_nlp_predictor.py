from easynlp.appzoo import ClassificationDataset
import os, json, math
from easynlp.utils import io, parse_row_by_schema
from threading import Lock
from tqdm import tqdm
import numpy as np
import uuid
from torch.utils.data import DataLoader
import torch
from transformers import BertForSequenceClassification
from easynlp.modelzoo import AutoTokenizer
from examples.hf_adapter_easynlp.hf_ez_nlp_user_defined import forward_repre
from types import MethodType

class Predictor(object):
    def __init__(self, model_dir, user_defined_parameters, **kwargs):
        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.MUTEX = Lock()

        self.model_predictor = PyModelPredictor(model_cls=BertForSequenceClassification,
                            saved_model_path=model_dir,
                            input_keys=[("input_ids", torch.LongTensor), ("attention_mask", torch.LongTensor),
                                        ("token_type_ids", torch.LongTensor)],
                            output_keys=["predictions", "probabilities", "logits"])

        self.label_path = os.path.join(model_dir, "label_mapping.json")
        with io.open(self.label_path) as f:
            self.label_mapping = json.load(f)
        
        self.label_id_to_name = {idx: name for name, idx in self.label_mapping.items()}
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.second_sequence = kwargs.pop("second_sequence", "second_sequence")
        self.sequence_length = kwargs.pop("sequence_length", 128)
        self.input_file = kwargs.pop("input_file")
        self.output_file = kwargs.pop("output_file")
        self.input_schema = kwargs.pop("input_schema")
        self.output_schema = kwargs.pop("output_schema")
        self.append_cols = kwargs.pop("append_cols")
        self.batch_size = kwargs.pop("batch_size", 32)
        self.skip_first_line = kwargs.pop("skip_first_line", False)
        self.args = kwargs.pop("args")

        self.fout = io.open(self.output_file, "w")
        with io.open(self.input_file) as f:
            if self.skip_first_line:
                f.readline()
            self.data_lines = f.readlines()

    def preprocess(self):

        eval_dataset = ClassificationDataset(
        pretrained_model_name_or_path=self.args.pretrained_model_name_or_path,
        data_file=self.args.tables.split(",")[-1],
        max_seq_length=self.args.sequence_length,
        input_schema=self.args.input_schema,
        first_sequence=self.args.first_sequence,
        second_sequence=self.args.second_sequence,
        label_name=self.args.label_name,
        label_enumerate_values=self.args.label_enumerate_values,
        is_training=False)

        self.eval_dataloader = DataLoader(eval_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       collate_fn=eval_dataset.batch_fn)
        return self.eval_dataloader

    def postprocess(self, result_list):
        return_list = []
        for result in result_list:
            probs = result["probabilities"]
            logits = result["logits"]
            predictions = np.argsort(-probs, axis=-1)

            new_results = list()
            for b, preds in enumerate(predictions):
                new_result = list()
                for pred in preds:
                    new_result.append({
                        "pred": self.label_id_to_name[pred],
                        "prob": float(probs[b][pred]),
                        "logit": float(logits[b][pred])
                    })
                new_results.append({
                    "id": result["id"][b] if "id" in result else str(uuid.uuid4()),
                    "output": new_result,
                    "predictions": new_result[0]["pred"],
                    "probabilities": ",".join([str(t) for t in result["probabilities"][b]]),
                    "logits": ",".join([str(t) for t in result["logits"][b]])
                })
            if len(new_results) == 1:
                new_results = new_results[0]
            return_list.append(new_results)
        return return_list

    def run(self):
        return_output_dict_list = self.postprocess(self.model_predictor.predict(self.args, self.preprocess()))
        for output_dict_list in return_output_dict_list:
            for output_dict in output_dict_list:
                out_record = []
                for colname in self.output_schema.split(","):
                    out_record.append(str(output_dict[colname]))
                self.fout.write("\t".join(out_record) + "\n")

                # if self.append_cols:
                #     for colname in self.append_cols.split(","):
                #         out_record.append(str(input_dict[colname]))
                self.fout.write("\t".join(out_record) + "\n")
        self.fout.close()

class PyModelPredictor(object):

    def __init__(self, model_cls, saved_model_path, input_keys, output_keys):
        
        self.model = model_cls.from_pretrained(pretrained_model_name_or_path=saved_model_path)
        self.model.forward_repre = MethodType(forward_repre, self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.input_keys = input_keys
        self.output_keys = output_keys

    def predict(self, args, eval_dataloder):
        return_list = []
        for _step, batch in tqdm(enumerate(eval_dataloder), desc='processing test data...'):
            predictions, _, in_tensor = self.model.forward_repre(self.model, args, batch)
            predictions = {
                'logits': predictions['logits'],
                'predictions': torch.argmax(predictions['logits'], dim=-1),
                'probabilities': torch.softmax(predictions['logits'], dim=-1)
            }
            ret = {}
            for key, val in in_tensor.items():
                ret[key] = val.data.cpu().numpy()

            for key in self.output_keys:
                ret[key] = predictions[key].data.cpu().numpy()
            return_list.append(ret)
        return return_list
