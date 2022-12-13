import torch
import numpy as np
from ...utils.logger import logger
from ...core.evaluator import Evaluator

def fush_multi_answer(has_answer, new_answer):
    # 对于某个id测试集，出现多个example时（例如同一个测试样本使用了多个模板而生成了多个example），此时将预测的topk结果进行合并
    # has为已经合并的结果，new为当前新产生的结果，
    # has格式为 {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
    # new {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
    # print('has_answer=', has_answer)
    for ans, value in new_answer.items():
        if ans not in has_answer.keys():
            has_answer[ans] = value
        else:
            has_answer[ans]['prob'] += value['prob']
            has_answer[ans]['pos'].extend(value['pos'])
    return has_answer

def get_predict_result(batchs, probs, indices, max_seq_length):
    probs = probs.squeeze(1)  # topk结果的概率
    indices = indices.squeeze(1)  # topk结果的索引

    predictions = {}
    topk_predictions = {}

    for _id, instruction, offset_mapping, prob, index in zip(batchs["id"], batchs["instruction"], batchs["offset_mapping"], probs, indices):

        index_ids = torch.Tensor([i for i in range(len(index))]).long()
        answer = []
        topk_answer_dict = dict()
        # TODO 1. 调节阈值 2. 处理输出实体重叠问题
        entity_index = index[prob > 0.6]
        index_ids = index_ids[prob > 0.6]

        for ei, entity in enumerate(entity_index):

            entity = entity.cpu().numpy()
            start_end = np.unravel_index(entity, (max_seq_length, max_seq_length))

            s = offset_mapping[start_end[0]][0]
            e = offset_mapping[start_end[1]][1]
            ans = instruction[s: e]

            if ans not in answer:
                answer.append(ans)
                # topk_answer.append({'answer': ans, 'prob': float(prob[index_ids[ei]]), 'pos': (s, e)})
                topk_answer_dict[ans] = {'prob': float(prob[index_ids[ei]]), 'pos': [(s, e)]}
        predictions[_id] = answer

        if _id not in topk_predictions.keys():
            topk_predictions[_id] = topk_answer_dict
        else:
            topk_predictions[_id] = fush_multi_answer(topk_predictions[_id], topk_answer_dict)

    for id_, values in topk_predictions.items():

        answer_list = list()
        for ans, value in values.items():
            answer_list.append({'answer': ans, 'prob': value['prob'], 'pos': value['pos']})
        topk_predictions[id_] = answer_list

    return predictions, topk_predictions

class InformationExtractionEvaluator(Evaluator):
    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)

        self.max_seq_length = kwargs["few_shot_anchor_args"].sequence_length
        #easynlp.appzoo.api.py中的get_application_evaluator()函数缺乏sequence_length的引入。为了一致性，information_extraction的evaluator通过few_shot_anchor_args引入sequence_length
    
    def _compute(self, label, pred, hit):
        if label == 0:
            recall = 1 if pred == 0 else 0
            precision = 1 if pred == 0 else (hit / pred)
        else:
            recall = hit / label
            precision = 0 if pred == 0 else (hit / pred)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def calc_metric(self, golden, predictions):
        f1 = 0.
        acc = 0.
        for k in golden.keys():
            hit_entities = [e for e in predictions[k] if e in golden[k]]
            _recall, _precision, _f1 = self._compute(
                len(golden[k]),
                len(predictions[k]),
                len(hit_entities)
            )
            f1 += _f1
            acc += _precision
        return {
            'acc': acc/len(golden.keys()),
            'f1': f1/len(golden.keys())
        }
    
    def evaluate(self, model):

        model.eval()
        dataname_map = {}
        golden = {}
        predictions = {}
        for _step, batch in enumerate(self.valid_loader):

            try:
                batch = {
                            key: val.cuda() if isinstance(val, torch.Tensor) else val
                            for key, val in batch.items()
                        }
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}
            
            with torch.no_grad():
                outputs = model(batch)
            topk_probs = outputs["topk_probs"]
            topk_indices = outputs["topk_indices"]

            prediction, _ = get_predict_result(batch, topk_probs, topk_indices, self.max_seq_length)
            predictions.update(prediction) #更新字典的操作
                
            for i in range(len(batch["id"])):
                id_ = batch["id"][i]
                dataname = "-".join(id_.split("-")[:-2])
                if dataname in dataname_map:
                    dataname_map[dataname].append(id_)
                else:
                    dataname_map[dataname] = [id_]

                golden[id_] = batch["target"][i].split('|')
        
        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
        }
        
        for dataname, data_ids in dataname_map.items():
            gold = {k: v for k, v in golden.items() if k in data_ids}
            pred = {k: v for k, v in predictions.items() if k in data_ids}
            score = self.calc_metric(golden=gold, predictions=pred)
            acc, f1 = score['acc'], score['f1']

            all_metrics["macro_f1"] += f1
            all_metrics["micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics[dataname] = round(acc, 4)
        all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        
        eval_outputs = list()
       
        for key, value in all_metrics.items():
            eval_outputs.append((key, value))
            logger.info("{}: {}".format(key, value))

        return eval_outputs
