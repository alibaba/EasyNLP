# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import time

import torch
from tqdm import tqdm

from ..utils import io, parse_row_by_schema, parse_tf_config
from ..utils.logger import logger

"""
try:
    import tensorflow as tf
    from tensorflow.python.saved_model.signature_constants import (
        DEFAULT_SERVING_SIGNATURE_DEF_KEY, )
except:
    pass

try:
    import common_io
except:
    pass

try:
    import easy_predict
    from easy_predict import (
        PredictorProcess,
        ProcessExecutor,
        TableReadProcess,
        TableReadProcessV1,
        TableWriteProcess,
        TableWriteProcessV1,
    )
    from easy_predict.core.engine import get_queue

    from ..utils.parallel_processes import (
        SelfDefinedFileReaderProcess,
        SelfDefinedFileWriterProcess,
        SelfDefinedPredictorProcess,
        SelfDefineTableFormatProcess,
    )

    USE_EASY_PREDICT = True
except Exception as err:
    print(err)
    USE_EASY_PREDICT = False
"""


class Predictor(object):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, in_data):
        return self.postprocess(self.predict(self.preprocess(in_data)))

    def preprocess(self, in_data):
        raise NotImplementedError

    def predict(self, in_data):
        raise NotImplementedError

    def postprocess(self, result):
        raise NotImplementedError


"""
class TFModelPredictor(object):
    def __init__(self, saved_model_path, input_keys, output_keys):
        # self.sess = tf.Session(graph=tf.Graph())
        self._graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_config = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False,
        )
        self.sess = tf.Session(config=session_config, graph=self._graph)
        meta_graph_def = tf.saved_model.loader.load(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            saved_model_path)
        self.signature = meta_graph_def.signature_def
        self.signature_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.graph = self.sess.graph

        self.input_keys = input_keys
        self.output_keys = output_keys
        for key in input_keys:
            tensor_name = self.signature[self.signature_key].inputs[key].name
            setattr(self, key + '_tensor',
                    self.graph.get_tensor_by_name(tensor_name))
        self.predictions = dict()
        for key in output_keys:
            tensor_name = self.signature[self.signature_key].outputs[key].name
            self.predictions[key] = self.graph.get_tensor_by_name(tensor_name)

    def predict(self, in_data):
        predictions = self.sess.run(self.predictions,
                                    feed_dict={
                                        getattr(self, key + '_tensor'):
                                        in_data[key]
                                        for key in self.input_keys
                                    })
        ret = {}
        for key, val in in_data.items():
            ret[key] = val
        for key, val in predictions.items():
            ret[key] = val
        return ret
"""


class PyModelPredictor(object):
    def __init__(self, model_cls, saved_model_path, input_keys, output_keys):
        self.model = model_cls.from_pretrained(
            pretrained_model_name_or_path=saved_model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.input_keys = input_keys
        self.output_keys = output_keys

    def predict(self, in_data):
        in_tensor = dict()
        for key, tensor_type in self.input_keys:
            in_tensor[key] = tensor_type(in_data[key])
            if torch.cuda.is_available():
                in_tensor[key] = in_tensor[key].cuda()
        with torch.no_grad():
            predictions = self.model.forward(in_tensor)
        ret = {}
        for key, val in in_data.items():
            ret[key] = val

        for key in self.output_keys:
            ret[key] = predictions[key].data.cpu().numpy()
        return ret


def get_model_predictor(model_dir,
                        input_keys,
                        output_keys,
                        model_cls=None,
                        use_tf=None):
    """
    if use_tf is None:
        use_tf = io.exists(os.path.join(model_dir, 'saved_model.pb'))
    if use_tf:
        logger.info('Using TF saved model to predict...')
        return TFModelPredictor(saved_model_path=model_dir,
                                input_keys=[t[0] for t in input_keys],
                                output_keys=output_keys)
    else:
        logger.info('Using PyTorch .bin model to predict...')
        return PyModelPredictor(model_cls=model_cls,
                                saved_model_path=model_dir,
                                input_keys=input_keys,
                                output_keys=output_keys)
    """
    return PyModelPredictor(model_cls=model_cls,
                                saved_model_path=model_dir,
                                input_keys=input_keys,
                                output_keys=output_keys)


class SimplePredictorManager(object):
    def __init__(self,
                 predictor,
                 input_file,
                 input_schema,
                 output_file,
                 output_schema,
                 append_cols,
                 skip_first_line=False,
                 batch_size=32):
        self.predictor = predictor
        self.input_file = input_file
        self.output_file = output_file
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.append_cols = append_cols
        self.batch_size = batch_size
        with io.open(input_file) as f:
            if skip_first_line:
                f.readline()
            self.data_lines = f.readlines()

        self.fout = io.open(output_file, 'w')

    @staticmethod
    def get_batches(lst, k):
        total_batches = math.ceil(len(lst) / k)
        for i in range(total_batches):
            yield lst[i * k:(i + 1) * k]

    def run(self):
        for batch in tqdm(self.get_batches(self.data_lines, self.batch_size)):
            input_dict_list = list()
            for row in batch:
                input_dict = parse_row_by_schema(row, self.input_schema)
                input_dict_list.append(input_dict)

            output_dict_list = self.predictor.run(input_dict_list)

            for input_dict, output_dict in zip(input_dict_list,
                                               output_dict_list):
                out_record = []
                for colname in self.output_schema.split(','):
                    out_record.append(str(output_dict[colname]))
                if self.append_cols:
                    for colname in self.append_cols.split(','):
                        out_record.append(str(input_dict[colname]))
                self.fout.write('\t'.join(out_record) + '\n')
        self.fout.close()


"""
class EasyPredictorManager(object):
    def __init__(self,
                 predictor,
                 input_file,
                 output_file,
                 output_schema,
                 append_cols,
                 input_schema=None,
                 batch_size=32,
                 queue_size=1024,
                 slice_size=4096,
                 thread_num=1,
                 table_read_thread_num=16):
        self.predictor = predictor
        self.input_file = input_file
        self.input_schema = input_schema
        self.output_file = output_file
        self.output_schema = output_schema
        self.append_cols = append_cols
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.slice_size = slice_size
        self.thread_num = thread_num
        self.table_read_thread_num = table_read_thread_num
        self.restore_works_dir = os.environ.get(
            'TF_FAILOVER_RESTORE_WORKS_DIR', '')
        print('=' * 5 + ' Predict Manager ' + '=' * 5)
        print('QUEUE SIZE: ', self.queue_size)
        print('SLICE SIZE: ', self.slice_size)
        print('THREAD NUM: ', self.thread_num)
        print('TABLE READ THREAD NUM: ', self.table_read_thread_num)
        print('RESTORE Works Dir: ', self.restore_works_dir)

    def run(self):
        # batch size should be less than the total number of data in input table
        table_read_batch_size = 1
        table_read_thread_num = self.table_read_thread_num
        table_read_slice_size = self.slice_size
        if self.input_file.startswith('odps://'):
            reader = common_io.table.TableReader(self.input_file,
                                                 selected_cols='',
                                                 excluded_cols='',
                                                 slice_id=0,
                                                 slice_count=1,
                                                 num_threads=1,
                                                 capacity=1)
            schemas = reader.get_schema()
            selected_cols = list(set([col_name for col_name, _, _ in schemas]))
            cluster, job_name, task_index = parse_tf_config()
            distributed_elastic_inference = not (cluster is None
                                                 or job_name is None
                                                 or task_index is None)
        else:
            selected_cols = None
            distributed_elastic_inference = False
        output_cols = self.output_schema.split(',') + (
            self.append_cols.split(',') if self.append_cols else [])

        # create proc executor
        proc_exec = ProcessExecutor(self.queue_size)
        if not distributed_elastic_inference:
            # distribute info
            worker_id = 0
            num_worker = 1
            thread_num = self.thread_num
            # Local mode
            if self.input_file.startswith('odps://'):
                data_reader_process = TableReadProcess(
                    self.input_file,
                    selected_cols=selected_cols,
                    slice_id=worker_id,
                    slice_count=num_worker,
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=table_read_batch_size,
                    num_threads=table_read_thread_num)
            else:
                data_reader_process = SelfDefinedFileReaderProcess(
                    self.input_file,
                    input_schema=self.input_schema,
                    job_name='file_reader',
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=table_read_batch_size)

            proc_exec.add(data_reader_process)
            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='preprocess',
                    job_name='predictor_preprocess',
                    thread_num=thread_num * 2,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=get_queue(
                        queue_size=max(1, self.queue_size // self.batch_size)),
                    batch_size=self.batch_size))
            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='predict',
                    job_name='predictor_predict',
                    thread_num=thread_num,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=1))
            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='postprocess',
                    job_name='predictor_postprocess',
                    thread_num=thread_num * 2,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=1))
            proc_exec.add(
                SelfDefineTableFormatProcess(
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    reserved_col_names=self.append_cols.split(','),
                    output_col_names=self.output_schema.split(',')))
            if self.input_file.startswith('odps://'):
                output_writer_process = TableWriteProcess(
                    self.output_file,
                    output_col_names=output_cols,
                    slice_id=worker_id,
                    input_queue=proc_exec.get_input_queue())
            else:
                output_writer_process = SelfDefinedFileWriterProcess(
                    self.output_file,
                    output_col_names=output_cols,
                    input_queue=proc_exec.get_input_queue(),
                    job_name='file_writer')
            proc_exec.add(output_writer_process)
        else:
            assert 'ps' in cluster, 'elastic inference need 1 ps.'
            ps_hosts = cluster['ps']
            worker_hosts = cluster.get('worker', [])
            chief_hosts = cluster.get('chief', [])
            is_chief = job_name == 'chief'

            # distribute info
            worker_id = task_index
            num_worker = len(worker_hosts) + len(chief_hosts)
            if job_name == 'worker' and 'chief' in cluster:
                worker_id += 1

            thread_num = self.thread_num

            if job_name == 'ps':
                restore_works_dir = self.restore_works_dir
                if not io.exists(restore_works_dir):
                    io.makedirs(restore_works_dir)
                if io.isdir(restore_works_dir):
                    work_files = io.listdir(restore_works_dir)
                    if work_files:
                        raise RuntimeError(
                            '{} should be empty, not: {}'.format(
                                restore_works_dir, work_files))
                else:
                    raise RuntimeError(
                        '{} not a directory.'.format(restore_works_dir))

            if is_chief is None:
                is_chief = (job_name == 'worker' and task_index == 0)
                cluster = tf.train.ClusterSpec({
                    'ps': ps_hosts,
                    'worker': worker_hosts
                })
            else:
                if job_name == 'worker':
                    chief_hosts = ['@']
                cluster = tf.train.ClusterSpec({
                    'ps': ps_hosts,
                    'worker': worker_hosts,
                    'chief': chief_hosts
                })
            server = tf.train.Server(cluster,
                                     job_name=job_name,
                                     task_index=task_index)
            print('=' * 5 + ' Print Server ' + '=' * 5)
            print('PS HOSTS: ', ps_hosts)
            print('Worker HOSTS: ', worker_hosts)
            print('Chief HOSTS: ', chief_hosts)
            print('Cluster: ', cluster)
            print('Job name: ', job_name)
            print('Task Index: ', task_index)
            print('=' * 10)

            if job_name == 'ps':
                server.join()
                return

            # worker wait for ps to make restore_works_dir
            while not io.isdir(self.restore_works_dir):
                tf.logging.info(
                    'waiting for ps to create restore_works_dir %s' %
                    self.restore_works_dir)
                time.sleep(5)

            proc_exec.add(
                TableReadProcessV1(cluster=cluster,
                                   server=server,
                                   job_name=job_name,
                                   task_index=task_index,
                                   is_chief=is_chief,
                                   input_table=self.input_file,
                                   selected_cols=selected_cols,
                                   slice_size=table_read_slice_size,
                                   restore_works_dir=self.restore_works_dir,
                                   output_queue=proc_exec.get_output_queue(),
                                   batch_size=table_read_batch_size))

            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='preprocess',
                    job_name='predictor_preprocess',
                    thread_num=thread_num * 2,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=get_queue(
                        queue_size=max(1, self.queue_size // self.batch_size)),
                    batch_size=self.batch_size))
            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='predict',
                    job_name='predictor_predict',
                    thread_num=thread_num,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=1))
            proc_exec.add(
                SelfDefinedPredictorProcess(
                    predictor=self.predictor,
                    mode='postprocess',
                    job_name='predictor_postprocess',
                    thread_num=thread_num * 2,
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    batch_size=1))
            proc_exec.add(
                SelfDefineTableFormatProcess(
                    input_queue=proc_exec.get_input_queue(),
                    output_queue=proc_exec.get_output_queue(),
                    reserved_col_names=self.append_cols.split(','),
                    output_col_names=self.output_schema.split(',')))
            proc_exec.add(
                TableWriteProcessV1(output_table=self.output_file,
                                    output_col_names=output_cols,
                                    restore_work_dir=self.restore_works_dir,
                                    job_name=job_name,
                                    task_index=task_index,
                                    input_queue=proc_exec.get_input_queue()))

        proc_exec.run()
        proc_exec.wait()
        if not distributed_elastic_inference and not self.input_file.startswith(
                'odps://'):
            output_writer_process.close()
"""


class PredictorManager(object):
    def __init__(self,
                 predictor,
                 input_file,
                 output_file,
                 output_schema,
                 append_cols,
                 input_schema=None,
                 skip_first_line=False,
                 batch_size=32,
                 queue_size=1024,
                 slice_size=4096,
                 thread_num=1,
                 table_read_thread_num=16):
        """
        if input_file.startswith('odps://'):
            assert USE_EASY_PREDICT, 'Predict ODPS need to `pip install easypredict` first'
        if USE_EASY_PREDICT:
            logger.info('Using EasyPredict to predict...')
            self.predictor_manager = EasyPredictorManager(
                predictor,
                input_file,
                output_file,
                output_schema,
                append_cols,
                input_schema=input_schema,
                batch_size=batch_size,
                queue_size=queue_size,
                slice_size=slice_size,
                thread_num=thread_num,
                table_read_thread_num=table_read_thread_num)
        else:
            logger.info('Using SimplePredict to predict...')
            self.predictor_manager = SimplePredictorManager(
                predictor, input_file, input_schema, output_file,
                output_schema, append_cols, skip_first_line, batch_size)
        """
        self.predictor_manager = SimplePredictorManager(
                predictor, input_file, input_schema, output_file,
                output_schema, append_cols, skip_first_line, batch_size)


    def run(self):
        self.predictor_manager.run()
