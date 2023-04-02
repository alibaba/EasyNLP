import os, torch, time, joblib, faiss, cv2
import numpy as np
from ortools.graph.python import min_cost_flow
from PIL import Image
import clip


class BiSequencialSimilarity(torch.nn.Module):
    def __init__(self):
        super(BiSequencialSimilarity, self).__init__()

    def CosSimilarity(self, u, v):
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-5)
        similarity = torch.matmul(u, v.T)
        return similarity

    def masked_mean_dim2(self, x, length):
        B1, B2, L2 = x.shape
        mask = torch.arange(L2).view(1, L2).repeat(B2, 1).to(length.device)
        mask = mask < (length.view(B2, 1).repeat(1, L2))
        mask = mask.view(1, B2, L2).repeat(B1, 1, 1)
        x = (x * mask).sum(dim=2) / length.view(1, B2).repeat(B1, 1).to(x.dtype)
        return x

    def masked_mean_dim1(self, x, length):
        V, F, D = x.shape
        mask = (torch.arange(F).repeat(V,1)).to(length.device)>=(length.repeat(F,1).T)
        x[mask] = 0
        x = x.sum(axis=1)/length.repeat(D,1).T.to(torch.float16)
        return x

    def forward(self, s1, s2, len_s1=None, len_s2=None):
        B1, L1, D = s1.shape
        B2, L2, D = s2.shape
        s1 = s1.view(B1*L1, D)
        s2 = s2.view(B2*L2, D)
        s = self.CosSimilarity(s1, s2)
        s = s.view(B1, L1, B2, L2)
        s1 = s.max(dim=1)[0]
        s1 = self.masked_mean_dim2(s1, len_s2)
        s2 = s.max(dim=3)[0]
        s2 = self.masked_mean_dim2(s2.permute(2, 0, 1), len_s1).T
        s = s1 + s2
        return s

    
class BiSequencialSimilarityPairwise(torch.nn.Module):
    def __init__(self):
        super(BiSequencialSimilarityPairwise, self).__init__()

    def CosSimilarity_cross(self, u, v):
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-5)
        similarity = torch.matmul(u, v.permute(0, 2, 1))
        return similarity

    def length2mask(self, length, max_length):
        mask = (torch.arange(max_length).repeat(length.shape[0],1).to(length.device))<(length.repeat(max_length,1).T)
        return mask

    def masked_mean(self, x, length, max_length):
        B, L = x.shape
        mask = self.length2mask(length, max_length)
        x = (x * mask).sum(dim=1) / length.to(x.dtype)
        return x

    def process_s_mean_max(self, s, len_s1, len_s2):
        B, L1, L2 = s.shape
        s1 = s.max(dim=1)[0]
        s1 = self.masked_mean(s1, len_s2, L2)
        s2 = s.max(dim=2)[0]
        s2 = self.masked_mean(s2, len_s1, L1)
        return s1 + s2

    def forward(self, s1, s2, len_s1=None, len_s2=None):
        s = self.CosSimilarity_cross(s1, s2)
        s = self.process_s_mean_max(s, len_s1, len_s2)
        return s


class MatchModel(torch.nn.Module):
    def __init__(self, data_folder, max_frames=12, max_tokens=77, device=torch.device("cuda")):
        super(MatchModel, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        self.similarity_model = BiSequencialSimilarity()
        self.similarity_model_pairwise = BiSequencialSimilarityPairwise()
        self.data_folder = data_folder
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.device = device

    def encode_text(self, text):
        text = clip.tokenize(text, context_length=1000)[:,:77].tolist()
        for i in range(len(text)):
            if text[i][-1]!=0 and text[i][-1]!=49407:
                text[i][-1] = 49407
        text = torch.LongTensor(text).to(self.device)
        # The following code is from CLIP
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        # The above code is from CLIP
        B, L, D = x.shape
        x = (x.view(B*L, D) @ self.clip_model.text_projection).view(B, L, D)
        num_tokens = text.argmax(dim=-1) + 1
        return x, num_tokens

    def encode_frames(self, video_id):
        # frames
        frames = []
        for i in video_id:
            f = joblib.load(os.path.join(self.data_folder, "%s.pkl" % i))
            if f is None:
                f = torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float16)
            frames.append(f)
        frames = [torch.concat([f, torch.zeros((self.max_frames-f.shape[0], 3, 224, 224), dtype=torch.float16)], dim=0) for f in frames]
        frames = torch.stack(frames).to(torch.float16).to(self.device)
        # num_frames
        num_frames = [self.max_frames for i in video_id]
        num_frames = torch.LongTensor(num_frames).to(self.device)
        # frames_features
        V, F, C, H, W = frames.shape
        frames = frames.view((V*F, C, H, W))
        frames_feature = self.clip_model.encode_image(frames)
        D = frames_feature.shape[-1]
        frames_feature = frames_feature.view((V, F, D))
        return frames_feature, num_frames

    def calculate_similarity(self, text_feature, num_tokens, frames_feature, num_frames):
        similarity = self.similarity_model(text_feature, frames_feature, num_tokens, num_frames)
        return similarity

    def calculate_similarity_pairwise(self, text_feature, num_tokens, frames_feature, num_frames):
        similarity = self.similarity_model_pairwise(text_feature, frames_feature, num_tokens, num_frames)
        return similarity

    def forward(self, text, video_id, return_feature=False):
        text_feature, num_tokens = self.encode_text(text)
        frames_feature, num_frames = self.encode_frames(video_id)
        if return_feature:
            return text_feature, num_tokens, frames_feature, num_frames
        similarity = self.calculate_similarity(text_feature, num_tokens, frames_feature, num_frames)
        return similarity


class CoarseMatchModel(torch.nn.Module):
    def __init__(self, data_folder, max_frames=12, max_tokens=77, device=torch.device("cuda")):
        super(CoarseMatchModel, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        self.data_folder = data_folder
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.device = device

    def masked_mean_dim1(self, x, length):
        V, F, D = x.shape
        mask = (torch.arange(F).repeat(V,1)).to(self.device)>=(length.repeat(F,1).T)
        x[mask] = 0
        x = x.sum(axis=1)/length.repeat(D,1).T.to(torch.float16)
        return x

    def encode_text(self, text):
        text = clip.tokenize(text, context_length=1000)[:,:77].tolist()
        for i in range(len(text)):
            if text[i][-1]!=0 and text[i][-1]!=49407:
                text[i][-1] = 49407
        text = torch.LongTensor(text).to(self.device)
        # The following code is from CLIP
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        # The above code is from CLIP
        B, L, D = x.shape
        x = (x.view(B*L, D) @ self.clip_model.text_projection).view(B, L, D)
        # x = self.masked_mean_dim1(x, text.argmax(dim=-1)+1)
        x = x[np.arange(B), text.argmax(dim=-1)]
        return x

    def encode_frames(self, video_id):
        # frames
        frames = []
        for i in video_id:
            f = joblib.load(os.path.join(self.data_folder, "%s.pkl" % i))
            if f is None:
                f = torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float16)
            frames.append(f)
        frames = [torch.concat([f, torch.zeros((self.max_frames-f.shape[0], 3, 224, 224), dtype=torch.float16)], dim=0) for f in frames]
        frames = torch.stack(frames).to(torch.float16).to(self.device)
        # num_frames
        num_frames = [self.max_frames for i in video_id]
        num_frames = torch.LongTensor(num_frames).to(self.device)
        # frames_features
        V, F, C, H, W = frames.shape
        frames = frames.view((V*F, C, H, W))
        frames_feature = self.clip_model.encode_image(frames)
        D = frames_feature.shape[-1]
        frames_feature = frames_feature.view((V, F, D))
        return frames_feature.mean(dim=1)

    def calculate_similarity(self, text_feature, frames_feature):
        frames_feature = frames_feature / frames_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return torch.matmul(text_feature, frames_feature.T)

    def forward(self, text, video_id):
        text_feature = self.encode_text(text)
        frames_feature = self.encode_frames(video_id)
        similarity = self.calculate_similarity(text_feature, frames_feature)
        return similarity


def cross_entropy_loss(s):
    s = s * 100.0
    loss_v2t = torch.nn.functional.log_softmax(s, dim=0)
    loss_v2t = -torch.diag(loss_v2t).mean()
    loss_t2v = torch.nn.functional.log_softmax(s, dim=1)
    loss_t2v = -torch.diag(loss_t2v).mean()
    loss = loss_t2v + loss_v2t
    return loss


class VectorRetrievalEngine:
    def __init__(self, data, nprobe=10):
        data = data.astype('float32')
        faiss.normalize_L2(data)
        d = data.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, data.shape[0]//50, faiss.METRIC_INNER_PRODUCT)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = nprobe

    def query(self, q, topk):
        q = q.astype('float32')
        faiss.normalize_L2(q)
        D, I = self.index.search(q, topk)
        return I


class NetworkFlowEngine:
    def __init__(self):
        self.node_dict = {}
        self.node_list = []
        self.edge_list = []

    def get_min_cost(self, start_nodes, end_nodes, capacities, unit_costs, supplies):
        smcf = min_cost_flow.SimpleMinCostFlow()
        smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)
        for count, supply in enumerate(supplies):
            smcf.set_node_supply(count, supply)
        status = smcf.solve_max_flow_with_min_cost()
        if status != smcf.OPTIMAL:
            print('There was an issue with the min cost flow input.')
            print(f'Status: {status}')
        minc = smcf.optimal_cost()
        edge_list = []
        for i in range(smcf.num_arcs()):
            if smcf.flow(i)>0:
                edge_list.append((smcf.tail(i), smcf.head(i)))
        return minc, edge_list

    def add_edge(self, u, v, f, c):
        if u not in self.node_dict:
            self.node_dict[u] = len(self.node_dict)
            self.node_list.append(u)
        if v not in self.node_dict:
            self.node_dict[v] = len(self.node_dict)
            self.node_list.append(v)
        u = self.node_dict[u]
        v = self.node_dict[v]
        self.edge_list.append((u, v, f, c))

    def match(self, retrieval_result_coarse, similarity):
        scale = 10**10
        similarity = (similarity.to(torch.float64) * scale).to(torch.int64)
        n, m = len(retrieval_result_coarse), retrieval_result_coarse.max() + 1
        for i in range(n):
            self.add_edge(("source", 0), ("row", i), 1, 0)
        for j in range(m):
            self.add_edge(("column", j), ("target", 0), (n+m-1)//m, 0)
        for i, (topk_video_id, similarity_row) in enumerate(zip(retrieval_result_coarse, similarity)):
            for j, s in zip(topk_video_id, similarity_row):
                self.add_edge(("row", i), ("column", j), 1, -int(s))
        start_nodes = [i[0] for i in self.edge_list]
        end_nodes = [i[1] for i in self.edge_list]
        capacities = [i[2] for i in self.edge_list]
        unit_costs = [i[3] for i in self.edge_list]
        maxf = n
        supplies = [0]*len(self.node_list)
        supplies[self.node_dict[("source", 0)]] = maxf
        supplies[self.node_dict[("target", 0)]] = -maxf
        minc, edge_list = self.get_min_cost(start_nodes, end_nodes, capacities, unit_costs, supplies)
        result = [[] for i in range(n)]
        for u,v in edge_list:
            u = self.node_list[u]
            v = self.node_list[v]
            if u[0]=="row" and v[0]=="column":
                result[u[1]].append(v[1])
        return result


class Timer:
    def __init__(self):
        self.t = 0
    def reset(self):
        self.t = 0
    def start(self):
        self.start_time = time.time()
    def end(self):
        self.t += time.time() - self.start_time
    def show(self, message=""):
        print("%s: %.2f" % (message, self.t*1000))


class TextVideoRetrievaler:
    def __init__(self, device=torch.device("cuda")):
        self.device = device

    def prepare_coarse_retrieval(self, coarse_model, video_id_list, progress_bar, batch_size):
        coarse_model.eval()
        timer = Timer()
        timer.start()
        v_list = []
        with torch.no_grad():
            for k in progress_bar(range(0, len(video_id_list), batch_size)):
                l=min(k+batch_size, len(video_id_list))
                video_fearure = coarse_model.encode_frames(video_id_list[k:l])
                v_list.append(video_fearure.cpu())
        v_list = torch.concat(v_list, dim=0).numpy()
        timer.end()
        timer.show("Coarse-grained: video encoding")
        timer = Timer()
        timer.start()
        self.vre = VectorRetrievalEngine(v_list)
        timer.end()
        timer.show("Vector retrieval engine: training")

    def prepare_fine_retrieval(self, fine_model, video_id_list, progress_bar, batch_size):
        fine_model.eval()
        timer = Timer()
        timer.start()
        v_list = []
        with torch.no_grad():
            for k in progress_bar(range(0, len(video_id_list), batch_size)):
                l=min(k+batch_size, len(video_id_list))
                frames_feature, num_frames = fine_model.encode_frames(video_id_list[k:l])
                v_list.append((frames_feature.cpu(), num_frames.cpu()))
        frames_feature = torch.concat([i[0] for i in v_list], dim=0)
        num_frames = torch.concat([i[1] for i in v_list], dim=0)
        self.frames_feature = frames_feature.to(self.device)
        self.num_frames = num_frames.to(self.device)
        timer.end()
        timer.show("Fine-grained: video encoding")

    def prepare(self, coarse_model, fine_model, video_id_list, progress_bar = lambda x:x, batch_size = 64):
        print("==================== Offline preparation ====================")
        self.fine_model = fine_model
        self.prepare_fine_retrieval(fine_model, video_id_list, progress_bar, batch_size)
        self.coarse_model = coarse_model
        self.prepare_coarse_retrieval(coarse_model, video_id_list, progress_bar, batch_size)

    def process_text_coarse(self, coarse_model, text_list, progress_bar, batch_size):
        coarse_model.eval()
        t_list = []
        with torch.no_grad():
            for i in progress_bar(range(0, len(text_list), batch_size)):
                j=min(i+batch_size, len(text_list))
                text_feature = coarse_model.encode_text(text_list[i:j])
                t_list.append(text_feature.cpu())
        text_feature_coarse = torch.concat(t_list, dim=0)
        return text_feature_coarse

    def process_text_fine(self, fine_model, text_list, progress_bar, batch_size):
        fine_model.eval()
        t_list = []
        with torch.no_grad():
            for i in progress_bar(range(0, len(text_list), batch_size)):
                j=min(i+batch_size, len(text_list))
                text_feature, num_tokens = fine_model.encode_text(text_list[i:j])
                t_list.append((text_feature.cpu(), num_tokens.cpu()))
        text_feature = torch.concat([i[0] for i in t_list], dim=0)
        num_tokens = torch.concat([i[1] for i in t_list], dim=0)
        return text_feature, num_tokens

    def index_map(self, index_list):
        index_dict = {}
        for i in index_list:
            if i not in index_dict:
                index_dict[i] = len(index_dict)
        index_list_a = [i for i in index_dict]
        index_list_b = [index_dict[i] for i in index_list]
        return index_list_a, index_list_b

    def process_similarity_fine(self, retrieval_result_coarse, fine_model, text_feature, num_tokens, frames_feature, num_frames, batch_size = 1024):
        with torch.no_grad():
            s_list = []
            for k in range(0, len(retrieval_result_coarse), batch_size):
                l = min(k+batch_size, len(retrieval_result_coarse))
                i_list, j_list = [], []
                for i, topk_video_id in zip(range(k,l), retrieval_result_coarse):
                    for j in topk_video_id:
                        i_list.append(i)
                        j_list.append(j)
                i_list_a, i_list_b = self.index_map(i_list)
                j_list_a, j_list_b = self.index_map(j_list)
                tf = text_feature[i_list_a].to(self.device)[i_list_b]
                nt = num_tokens[i_list_a].to(self.device)[i_list_b]
                ff = frames_feature[j_list_a].to(self.device)[j_list_b]
                nf = num_frames[j_list_a].to(self.device)[j_list_b]
                similarity = fine_model.calculate_similarity_pairwise(tf, nt, ff, nf).cpu()
                similarity = similarity.view((l-k, retrieval_result_coarse.shape[1]))
                s_list.append(similarity)
            s_list = torch.concat(s_list, dim=0)
        return similarity

    def sparse_dual_softmax(self, retrieval_result_coarse, similarity, alpha=100.0):
        similarity_dsl = torch.ones(similarity.size()).tolist()
        similarity = similarity.tolist()
        column_result = {}
        for i, (topk_video_id, similarity_row) in enumerate(zip(retrieval_result_coarse, similarity)):
            for j, s in zip(topk_video_id, similarity_row):
                if j not in column_result:
                    column_result[j] = []
                column_result[j].append(i)
        for j in column_result:
            column = torch.softmax(torch.tensor([similarity[i][j] for i in column_result[j]])*alpha).tolist()
            for i, value in zip(column_result[j], column):
                similarity_dsl[i][retrieval_result_coarse.index(j)] *= value
        similarity_dsl = torch.tensor(similarity_dsl) * torch.softmax(torch.tensor(similarity)*alpha, dim=1)
        return similarity_dsl

    def query_coarse(self, q, topk):
        return self.vre.query(q, topk)

    def resort(self, retrieval_result, similarity):
        retrieval_result_resort = []
        for topk_video_id, similarity_row in zip(retrieval_result, similarity.tolist()):
            ls = sorted([(s, i) for i, s in zip(topk_video_id, similarity_row)], reverse=True)
            retrieval_result_resort.append([i[1] for i in ls])
        return np.array(retrieval_result_resort)

    def merge_retrieval_result_flow(self, similarity, retrieval_result, retrieval_result_flow, beta=1.0):
        retrieval_result = retrieval_result.tolist()
        for i in range(len(retrieval_result_flow)):
            for j in retrieval_result_flow[i]:
                similarity[i,retrieval_result[i].index(j)] += beta
        return similarity

    def query(self, text_list, topk=30, progress_bar=lambda x:x, batch_size=64, inference_mode=3):
        print("==================== Online inference ====================")
        timer = Timer()
        timer.start()
        text_feature_coarse = self.process_text_coarse(self.coarse_model, text_list, progress_bar, batch_size).numpy()
        timer.end()
        timer.show("Coarse-grained: text encoding")

        timer = Timer()
        timer.start()
        retrieval_result_coarse = self.query_coarse(text_feature_coarse, topk)
        timer.end()
        timer.show("Vector retrieval engine: inference")

        if inference_mode==1:
            return retrieval_result_coarse

        timer = Timer()
        timer.start()
        text_feature, num_tokens = self.process_text_fine(self.fine_model, text_list, progress_bar, batch_size)
        timer.end()
        timer.show("Fine-grained: text encoding")

        timer = Timer()
        timer.start()
        similarity = self.process_similarity_fine(retrieval_result_coarse, self.fine_model, text_feature, num_tokens, self.frames_feature, self.num_frames)
        timer.end()
        timer.show("Fine-grained: similarity calculation")

        if inference_mode==3:
            timer = Timer()
            timer.start()
            retrieval_result_flow = NetworkFlowEngine().match(retrieval_result_coarse, similarity)
            timer.end()
            timer.show("Others: flow-style matching")

            timer = Timer()
            timer.start()
            similarity = self.merge_retrieval_result_flow(similarity, retrieval_result_coarse, retrieval_result_flow)
            timer.end()
            timer.show("Others: Sparse Dual Softmax")

        timer = Timer()
        timer.start()
        retrieval_result = self.resort(retrieval_result_coarse, similarity)
        timer.end()
        timer.show("Others: Reranking")
        return retrieval_result

    def test_retrieval_result(self, retrieval_result, text_list, video_id_list, task="t2v", out_range_ranking="worst"):
        if task=="v2t":
            text_list, video_id_list = video_id_list, text_list
        goal_set = {}
        for i, text in enumerate(text_list):
            if text not in goal_set:
                goal_set[text] = set()
            goal_set[text].add(i)
        rank_list = []
        for text, topk_video_id in zip(text_list, retrieval_result):
            rank = len(video_id_list) if out_range_ranking=="worst" else (len(topk_video_id) + 1)
            for i, video_id in enumerate(topk_video_id):
                if video_id in goal_set[text]:
                    rank = i + 1
                    break
            rank_list.append(rank)
        metric = {
            "R@1": sum([r<=1 for r in rank_list])/len(rank_list)*100,
            "R@5": sum([r<=5 for r in rank_list])/len(rank_list)*100,
            "R@10": sum([r<=10 for r in rank_list])/len(rank_list)*100,
            "MdR": sorted(rank_list)[len(rank_list)//2],
            "MnR": sum(rank_list)/len(rank_list)
        }
        return metric

    def analyze_result(self, retrieval_result, text_list, video_id_list, task="t2v"):
        metric_worst = self.test_retrieval_result(retrieval_result, text_list, video_id_list, task, out_range_ranking="worst")
        metric_best  = self.test_retrieval_result(retrieval_result, text_list, video_id_list, task, out_range_ranking="best")
        for i in metric_worst:
            print(f"{i}: [{min(metric_worst[i], metric_best[i])}, {max(metric_worst[i], metric_best[i])}]")


def train_model_huge_batch(model, optimizer, text_B, video_id_B, batch_size = 8, device=torch.device("cuda")):
    def get_feature(model, text_B, video_id_B, batch_size):
        text_feature, num_tokens, frames_feature, num_frames = [], [], [], []
        B = len(text_B)
        with torch.no_grad():
            for i in range(0, B, batch_size):
                j = min(i+batch_size, B)
                t, nt, f, nf = model(text_B[i:j], video_id_B[i:j], return_feature=True)
                text_feature.append(t.cpu())
                num_tokens.append(nt.cpu())
                frames_feature.append(f.cpu())
                num_frames.append(nf.cpu())
            text_feature = torch.concat(text_feature, dim=0).to(torch.float32)
            num_tokens = torch.concat(num_tokens, dim=0)
            frames_feature = torch.concat(frames_feature, dim=0).to(torch.float32)
            num_frames = torch.concat(num_frames, dim=0)
        return text_feature, num_tokens, frames_feature, num_frames
    def get_grad(model, text_feature, num_tokens, frames_feature, num_frames):
        text_feature.requires_grad_()
        frames_feature.requires_grad_()
        similarity = model.calculate_similarity(text_feature, num_tokens, frames_feature, num_frames)
        loss = cross_entropy_loss(similarity)
        loss.backward()
        text_grad = text_feature.grad.detach()
        frames_grad = frames_feature.grad.detach()
        return text_grad, frames_grad
    def update_model(model, optimizer, text_B, video_id_B, text_grad, frames_grad, batch_size, device):
        B = len(text_B)
        optimizer.zero_grad()
        for i in range(0, B, batch_size):
            j = min(i+batch_size, B)
            t, nt, f, nf = model(text_B[i:j], video_id_B[i:j], return_feature=True)
            tg = text_grad[i:j].to(device)
            fg = frames_grad[i:j].to(device)
            loss = torch.sum(t*tg) + torch.sum(f*fg)
            loss.backward()
        optimizer.step()
    text_feature, num_tokens, frames_feature, num_frames = get_feature(model, text_B, video_id_B, batch_size)
    text_grad, frames_grad = get_grad(model, text_feature, num_tokens, frames_feature, num_frames)
    update_model(model, optimizer, text_B, video_id_B, text_grad, frames_grad, batch_size, device)


def train_model_small_batch(model, optimizer, text_batch, video_id_batch):
    optimizer.zero_grad()
    # forward
    similarity = model(text_batch, video_id_batch)
    loss = cross_entropy_loss(similarity)
    # backward
    loss.backward()
    # update
    optimizer.step()


class DataConverter():
    def __init__(self):
        _, self.img_preprocess = clip.load("ViT-B/16", device="cpu")

    def video_file_to_frame_list(self, video_file, max_length=400):
        capture = cv2.VideoCapture(video_file)
        frame_list = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)
            if len(frame_list) >= max_length:
                break
        capture.release()
        return frame_list

    def frame_list_to_sampled_frame_list(self, frame_list, num_frames=12):
        index_list, frame_list_ = [], []
        j = 0
        for i in range(num_frames):
            pi = i/num_frames
            while j+1<len(frame_list) and abs((j+1)/len(frame_list) - pi) < abs(j/len(frame_list) - pi):
                j += 1
            index_list.append(j)
            frame_list_.append(frame_list[j])
        return frame_list_

    def frame_list_to_frames_tensor(self, frame_list, max_frames=400, padding=False):
        frames_tensor = [
            self.img_preprocess(Image.fromarray(i).convert("RGB")) for i in frame_list
        ]
        frames_tensor = torch.stack(frames_tensor).to(torch.float16)
        if padding:
            if frames_tensor.shape[0] < max_frames:
                padding_zeros = torch.zeros((
                    max_frames-frames_tensor.shape[0],
                    frames_tensor.shape[1],
                    frames_tensor.shape[2],
                    frames_tensor.shape[3]
                ))
                frames_tensor = torch.concat(
                    [frames_tensor, padding_zeros], axis=0)
            else:
                frames_tensor = frames_tensor[:max_frames]
        return frames_tensor

    def process_video(self, video_file, save_file):
        v = self.video_file_to_frame_list(video_file)
        v = self.frame_list_to_sampled_frame_list(v)
        v = self.frame_list_to_frames_tensor(v)
        joblib.dump(v, save_file)
