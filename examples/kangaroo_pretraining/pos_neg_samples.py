import pandas as pd
import numpy as np
from tqdm import tqdm

"""
生成positive 和negative采样
"""

saving_file = "../../tmp/finData/finSamples4Level.npy"

# 实体表csv文件路径
ENTITY_DATA_PATH = "../../tmp/finData/finEntity.csv"
# 关系表csv文件路径
RELATION_DATA_PATH = "../../tmp/finData/finRelation.csv"
# [CLS]符号
CLS_CHAR = "$"
# [SEP]符号
SEP_CHAR = "#"
# '[PATH]'符号
PATH_CHAR = "@"
# 是否使用双重路径
USE_DOUBLE_PATH = True
# 选择目标节点时是否以更高概率选择同标签的实体
USE_WEIGHTED_SAMPLE = True
# 选择目标节点时，选择同标签的实体的概率/选择不同标签的实体的概率
SAME_LABEL_WEIGHT = 10
# 序列长度限制
SEQUENCE_LENGTH = 128
# 随机种子
RANDOM_SEED = 666
# 最大的邻域半径
MAX_LEVEL = 4
# 每个实体每个邻域半径生成的数据量
DATA_NUM = 2
# 反向关系映射表
REVERSE_RELATION = {
    "位于": "有公司",
    "属于行业": "行业包含",
    "法人代表": "法人代表",
    "产业链包含产品": "产品属于产业链",
    "公司生产产品": "由公司生产",
    "上下游产品": "上下游产品",
    "行业包含": "属于行业",
    "产品属于行业": "行业包含产品",
}


np.random.seed(RANDOM_SEED)

def get_entity_label(entity_data, relation_data):
    entity_label = {}
    for label, name_list in zip(entity_data["label"], entity_data["name_list"]):
        name_list = [i for i in name_list.split("|") if len(i) > 0]
        for name in name_list:
            entity_label[name] = label
    return entity_label


def get_relation_list(entity_data, relation_data):
    entity_id_to_name_list = {}
    for index, name_list in zip(entity_data["index"], entity_data["name_list"]):
        name_list = [i for i in name_list.split("|") if len(i) > 0]
        entity_id_to_name_list[index] = name_list
    relation_list = []
    for head_id, tail_id, label in zip(
            relation_data["head"], relation_data["tail"], relation_data["label"]):
        for head in entity_id_to_name_list[head_id]:
            for tail in entity_id_to_name_list[tail_id]:
                relation_list.append((head, tail, label))
    relation_list = sorted(list(set(relation_list)))
    return relation_list


def create_full_name_map(entity_df):
    """
    构造name-->id映射，包含全称、简称
    :param entity_df:
    :return:
    """
    full_name_to_id = {}
    for i in range(len(entity_df)):
        full_name = entity_df.iloc[i]['main_name']
        name_list = entity_df.iloc[i]['name_list'].split('|')
        if pd.isna(full_name):
            full_name = entity_df.iloc[i]["name_list"].split('|')[0]
            name_list = entity_df.iloc[i]['name_list'].split('|')
        id = int(entity_df.iloc[i]['index'])
        for name in name_list:
            full_name_to_id[name] = id
    return full_name_to_id

class KnowledgeGraphManager:
    def __init__(self, relation_list, entity_label):
        self.node = []
        self.neighbor = []
        self.relation = {}
        self.entity_name_to_id = {}
        self.entity_label = entity_label
        self.valid_node = []
        self.build_graph(relation_list)
        self.mark = [-1]*len(self.node)
        self.previous_node = [-1]*len(self.node)
        self.node_cache = []

    def add_node(self, name):
        if name not in self.entity_name_to_id:
            self.node.append(name)
            self.neighbor.append([])
            self.entity_name_to_id[name] = len(self.entity_name_to_id)

    def build_graph(self, relation_list):
        for head, tail, relation in relation_list:
            self.add_node(head)
            self.add_node(tail)
            if head != tail:
                head = self.entity_name_to_id[head]
                tail = self.entity_name_to_id[tail]
                self.neighbor[head].append(tail)
                self.neighbor[tail].append(head)
                self.relation[(head, tail)] = relation
                self.relation[(tail, head)] = REVERSE_RELATION.get(
                    relation, relation)
        for i in range(len(self.node)):
            if len(self.neighbor[i]) > 0:
                self.valid_node.append(self.node[i])
        np.random.shuffle(self.valid_node)

    def __iter__(self):
        return self.valid_node.__iter__()

    def __next__(self):
        return self.valid_node.__next__()

    def __len__(self):
        return len(self.valid_node)

    def clear_cache(self):
        for u in self.node_cache:
            self.mark[u] = -1
            self.previous_node[u] = -1
        self.node_cache.clear()

    def bfs(self, start_node, max_depth):
        queue = [start_node]
        self.mark[start_node] = 0
        self.node_cache.append(start_node)
        while len(queue) > 0:
            u = queue.pop(0)
            for v in self.neighbor[u]:
                if self.mark[v] == -1:
                    self.mark[v] = self.mark[u]+1
                    self.previous_node[v] = u
                    self.node_cache.append(v)
                    if self.mark[v] < max_depth:
                        queue.append(v)

    def generate_shortest_path(self, start_node, end_node):
        shortest_path = [end_node]
        while shortest_path[0] != start_node:
            u = shortest_path[0]
            shortest_path.insert(0, self.previous_node[u])
        return shortest_path

    def generate_another_path(self, start_node, end_node, shortest_path):
        unvisible_node = set([i for i in shortest_path if i != end_node])
        visited_node = set()
        previous_node = {}
        queue = [start_node]
        visited_node.add(start_node)
        while len(queue) > 0:
            u = queue.pop(0)
            for v in self.neighbor[u]:
                if self.mark[v] != -1 and (v not in unvisible_node)\
                        and (v not in visited_node):
                    visited_node.add(v)
                    previous_node[v] = u
                    queue.append(v)
                    if v == end_node:
                        path = [v]
                        while path[0] != start_node:
                            path.insert(0, previous_node[path[0]])
                        return path
        return None

    def select_end_node(self, start_node, end_node_list):
        if USE_WEIGHTED_SAMPLE:
            p = [SAME_LABEL_WEIGHT if self.entity_label[start_node] == self.entity_label[end_node] else 1
                 for end_node in end_node_list]
            sum_p = sum(p)
            p = [i/sum_p for i in p]
            return np.random.choice(end_node_list, p=p)
        else:
            return np.random.choice(end_node_list)

    def path_to_str(self, path):
        result = self.node[path[0]] + SEP_CHAR
        for u, v in zip(path[:-1], path[1:]):
            result += self.relation[(u, v)] + SEP_CHAR
            result += self.node[v] + SEP_CHAR
        return result

    def __call__(self, start_node, max_depth):
        start_node = self.entity_name_to_id[start_node]
        self.bfs(start_node, max_depth)
        end_node_list = [self.node[i]
                         for i in self.node_cache if self.mark[i] == max_depth]
        if len(end_node_list) == 0:
            self.clear_cache()
            return None, None
        data = CLS_CHAR
        position = [0]
        while True:
            end_node = self.select_end_node(
                self.node[start_node], end_node_list)
            end_node = self.entity_name_to_id[end_node]
            shortest_path = self.generate_shortest_path(start_node, end_node)
            path = self.path_to_str(shortest_path)
            if USE_DOUBLE_PATH:
                another_path = self.generate_another_path(
                    start_node, end_node, shortest_path)
                if another_path is not None:
                    path += self.path_to_str(another_path)
            if len(data)+len(path) <= SEQUENCE_LENGTH:
                data += path
                position += [position[-1] + 1] * len(path)
                # print(len(data), len(position))
            else:
                self.clear_cache()
                return data, position

# 读取csv文件
entity_data = pd.read_csv(ENTITY_DATA_PATH)
relation_data = pd.read_csv(RELATION_DATA_PATH)
# 获取实体名到实体标签的映射
entity_label = get_entity_label(entity_data, relation_data)
# 获取关系列表（已将实体按照名称克隆）
relation_list = get_relation_list(entity_data, relation_data)
# 输出KG信息
print("%d entities, %d relations" % (len(entity_label), len(relation_list)))
# 构造name2id字典
name2id = create_full_name_map(entity_data)



# 初始化KG管理器
manager = KnowledgeGraphManager(relation_list, entity_label)

# 搞起来！
data_dict = {}
for start_entity in tqdm(manager):
    start_entity_id = name2id[start_entity]
    data_dict[start_entity_id] = {}
    for level in range(1, MAX_LEVEL+1):
        data_dict[start_entity_id]["level_%d" % level] = []
        for i in range(DATA_NUM):
            sequence, position = manager(start_entity, level)
            # 需要注意，如果半径为level的邻域边界上没有节点，则返回None，这仅在邻域半径过大时发生
            if sequence is not None:

                # 转换为token_id，并且控制sequence_len
                # tokens = ["[CLS]" if x == "$" else ("[SEP]" if x == "#" else ("[PATH]" if x=="@" else x)) for x in sequence]
                tokens = ["[CLS]" if x == "$" else ("[SEP]" if x == "#" else x) for x in sequence]
                data_dict[start_entity_id]["level_%d" % level].append(
                    {'name': start_entity, 'sequence': sequence, 'tokens': tokens, 'position_id': position}
                )


# print(data_dict["浦发银行"])
np.save(saving_file, data_dict)
print("successful save ", saving_file)