import torch, os
from PIL import Image, ImageEnhance
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm


def PIL_to_tensor(image):
    return torch.tensor(np.array(image)).to(torch.float32)/255

def tensor_to_PIL(image):
    return Image.fromarray((image * 255).detach().clamp(0, 255).cpu().numpy().astype("uint8"))


class LeftVideoGraph:
    def __init__(self, n):
        self.n = n
        self.edges = {}
        for i in range(n):
            father = self.father(i)
            if father<self.n:
                self.edges[(i, father)] = None
            for j in self.cousin_leaves(i):
                if j<self.n:
                    self.edges[(i, j)] = None

    def father(self, x):
        y = 1
        while x&y:
            y <<= 1
        return x|y

    def cousin(self, x):
        y = 1
        while (y<<1)<x:
            y <<= 1
        if (y>>1)>(x^y):
            return None
        return x^y

    def cousin_leaves(self, x):
        y = 1
        while x&y:
            y <<= 1
        x -= x & (y - 1)
        return range(x+y, x+(y<<1))

    def query_middle_node(self, x, y):
        for z in range(x+1, y):
            if (x, z) in self.edges and (z, y) in self.edges:
                return z
        return None

    def query(self, x):
        z_list = []
        z = -1
        for i in range(10):
            y = 1
            while z + (y<<1)<=x:
                y <<= 1
            z += y
            z_list.append(z)
            if z==x:
                break
        return z_list

    def query_edge(self, level):
        edge_list = []
        step = 1<<level
        for x in range(step-1, self.n, step*2):
            y = x + step
            if y<self.n:
                edge_list.append((x, y))
        return edge_list


class PatchMatcher:
    def __init__(self, height, width, patch_size, num_iter=6, device="cuda"):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_iter = num_iter
        self.device = device

        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size),
            padding=(self.patch_size//2, self.patch_size//2)
        )
        self.fold = torch.nn.Fold(
            output_size=(self.height, self.width),
            kernel_size=(self.patch_size, self.patch_size),
            padding=(self.patch_size//2, self.patch_size//2)
        )
        ones_like_image = torch.ones((1, 3, self.height, self.width), device=self.device)
        self.divisor = self.fold(self.unfold(ones_like_image))

    def image_to_patch(self, image):
        height, width, _ = image.shape
        image = rearrange(image, "H W C -> 1 C H W")
        patch = self.unfold(image)
        patch = rearrange(
            patch,
            "1 L (H W) -> H W L",
            H=self.height,
            W=self.width
        )
        return patch

    def patch_to_image(self, patch):
        patch = rearrange(patch, "H W L -> 1 L (H W)")
        image = self.fold(patch) / self.divisor
        image = rearrange(image, "1 C H W -> H W C")
        return image

    def apply_nnf_to_image(self, nnf, image):
        nnf, image = nnf.to(self.device), image.to(self.device)
        patch = self.image_to_patch(image)
        patch = patch[nnf[:,:,0], nnf[:,:,1]]
        image = self.patch_to_image(patch)
        return image

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        target_guide_pred = source_guide[nnf[:,:,0], nnf[:,:,1]]
        error_guide = (target_guide - target_guide_pred).norm(dim=-1)
        target_style_pred = source_style[nnf[:,:,0], nnf[:,:,1]]
        error_style = (target_style - target_style_pred).norm(dim=-1)
        return error_guide * 6.0 + error_style

    def clamp_bound(self, nnf, height, width):
        nnf[nnf < 0] = 0
        nnf[:, :, 0][nnf[:, :, 0] >= height] = height - 1
        nnf[:, :, 1][nnf[:, :, 1] >= width] = width - 1
        return nnf

    def random_step(self, nnf, w):
        height, width, _ = nnf.shape
        step = torch.stack(
            [
                torch.randint(-height, height, (height, width), dtype=torch.int32, device=self.device),
                torch.randint(-width, width, (height, width), dtype=torch.int32, device=self.device)
            ],
            dim=2
        )
        step = (step * w).to(torch.int32)
        upd_nnf = self.clamp_bound(nnf + step, height, width)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        height, width, _ = nnf.shape
        if d==0:
            upd_nnf = torch.concat([nnf[:1, :], nnf[:-1, :]], dim=0)
            upd_nnf[:, :, 0] += 1
        elif d==1:
            upd_nnf = torch.concat([nnf[:, :1], nnf[:, :-1]], dim=1)
            upd_nnf[:, :, 1] += 1
        elif d==2:
            upd_nnf = torch.concat([nnf[1:, :], nnf[-1:, :]], dim=0)
            upd_nnf[:, :, 0] -= 1
        elif d==3:
            upd_nnf = torch.concat([nnf[:, 1:], nnf[:, -1:]], dim=1)
            upd_nnf[:, :, 1] -= 1
        upd_nnf = self.clamp_bound(upd_nnf, height, width)
        return upd_nnf

    def update(self, source_guide, target_guide, source_style, target_style, nnf, dis, upd_nnf):
        upd_dis = self.get_error(source_guide, target_guide, source_style, target_style, upd_nnf)
        upd_idx = (upd_dis < dis)
        nnf[upd_idx] = upd_nnf[upd_idx]
        dis[upd_idx] = upd_dis[upd_idx]
        return nnf, dis

    def iteration(self, source_guide, target_guide, source_style, target_style, nnf, dis):
        height, width, _ = nnf.shape
        # propagation
        for d in range(4):
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, dis = self.update(source_guide, target_guide, source_style, target_style, nnf, dis, upd_nnf)
        # random search
        w = 1.0
        while w*max(height, width)>1:
            upd_nnf = self.random_step(nnf, w)
            nnf, dis = self.update(source_guide, target_guide, source_style, target_style, nnf, dis, upd_nnf)
            w *= 0.5
        return nnf, dis

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        target_style = self.apply_nnf_to_image(nnf, source_style)
        source_guide, target_guide, source_style, target_style = map(
            lambda x: x.to(self.device),
            (source_guide, target_guide, source_style, target_style)
        )
        source_guide, target_guide, source_style, target_style = map(
            self.image_to_patch,
            (source_guide, target_guide, source_style, target_style)
        )
        dis = self.get_error(source_guide, target_guide, source_style, target_style, nnf)
        for it in range(self.num_iter):
            nnf, dis = self.iteration(source_guide, target_guide, source_style, target_style, nnf, dis)
            target_style = source_style[nnf[:,:,0], nnf[:,:,1]]
            target_style = self.patch_to_image(target_style)
            if it<self.num_iter-1:
                target_style = self.image_to_patch(target_style)
        return nnf, target_style


class PyramidPatchMatcher:
    def __init__(self, image_height, image_width, patch_size, num_iter=6, device="cuda"):
        self.pyramid_level = int(np.log2(min(image_height, image_width) / patch_size))
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.patch_size = patch_size
        self.num_iter = num_iter
        self.device = device
        for level in range(self.pyramid_level):
            height = image_height//(2**(self.pyramid_level - 1 - level))
            width = image_width//(2**(self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(PatchMatcher(height, width, patch_size, num_iter=num_iter, device=device))

    def resample(self, image, height, width):
        image = rearrange(image, "H W C -> 1 C H W")
        image = torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear")
        image = rearrange(image, "1 C H W -> H W C")
        return image

    def resample_image_to_pyramid_level(self, image, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if level < self.pyramid_level - 1:
            image = self.resample(image, height, width)
        else:
            image = image.clone()
        return image

    def get_initial_nnf(self, nnf, level):
        height = self.pyramid_heights[level]
        width = self.pyramid_widths[level]
        if nnf is None:
            nnf = torch.stack(
                [
                    torch.randint(0, height, (height, width), dtype=torch.int32, device=self.device),
                    torch.randint(0, width, (height, width), dtype=torch.int32, device=self.device)
                ],
                dim=2
            )
        else:
            scale = height / nnf.shape[0]
            nnf = self.resample(nnf.to(torch.float32) * scale, height, width).to(torch.int32)
        return nnf

    def apply_nnf_to_image(self, nnf, image):
        return self.patch_matchers[-1].apply_nnf_to_image(nnf, image)

    def nnf_add(self, nnf_a, nnf_b):
        nnf_c = nnf_a[nnf_b[:,:,0], nnf_b[:,:,1]] # TODO:check
        return nnf_c

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf=None):
        for level in range(self.pyramid_level):
            nnf = self.get_initial_nnf(nnf, level)
            source_guide_, target_guide_, source_style_ = map(
                lambda x: self.resample_image_to_pyramid_level(x, level),
                (source_guide, target_guide, source_style)
            )
            nnf, target_style = self.patch_matchers[level].estimate_nnf(
                source_guide_, target_guide_, source_style_, nnf
            )
        return nnf, target_style


class EbsynthBinaryEngine:
    def __init__(self, bin_path, cache_path="cache", patch_size=21, n_jobs=16):
        self.bin_path = bin_path
        self.cache_path = cache_path
        self.patch_size = patch_size
        self.n_jobs = n_jobs
        os.makedirs(cache_path, exist_ok=True)

    def load_nnf(self, file_name, height, width):
        with open(file_name, "r") as f:
            nnf = list(map(int, f.read().split()))
            nnf = torch.tensor(nnf, dtype=torch.int32)
            nnf = rearrange(nnf, "(H W C) -> H W C", H=height, W=width, C=2)
            nnf = nnf[:,:,[1,0]]
        return nnf

    def run_cmd(self, cmd_list):
        cmd_list_file = os.path.join(self.cache_path, "cmd_list.txt")
        with open(cmd_list_file, "w", encoding="utf-8") as f:
            for cmd in cmd_list:
                f.write(cmd)
                f.write("\n")
        cmd = f"cat {cmd_list_file} | xargs -P {self.n_jobs} -I cmd sh -c 'cmd'"
        os.system(cmd)

    def get_nnf_dict(self, graph, guide, style):
        for i, frame in enumerate(guide):
            tensor_to_PIL(frame).save(os.path.join(self.cache_path, f"guide_{i}.jpg"))
        for i, frame in enumerate(style):
            tensor_to_PIL(frame).save(os.path.join(self.cache_path, f"style_{i}.jpg"))
        
        cmd_list = []
        for u, v in graph.edges:
            source_guide = os.path.join(self.cache_path, f"guide_{u}.jpg")
            target_guide = os.path.join(self.cache_path, f"guide_{v}.jpg")
            source_style = os.path.join(self.cache_path, f"style_{u}.jpg")
            nnf_file = os.path.join(self.cache_path, f"nnf_{u}_{v}.txt")
            cmd = f"{self.bin_path} -style {source_style} -guide {source_guide} {target_guide} -weight 100.0 -output {nnf_file} -patchsize {self.patch_size}"
            cmd_list.append(cmd)
        self.run_cmd(cmd_list)

        nnf_dict = {}
        height, width = style[0].shape[0], style[0].shape[1]
        for u, v in graph.edges:
            nnf_file = os.path.join(self.cache_path, f"nnf_{u}_{v}.txt")
            nnf = self.load_nnf(nnf_file, height, width)
            nnf_dict[(u, v)] = nnf.cpu()
        return nnf_dict


class PytorchEngine:
    def __init__(self, patch_matcher):
        self.patch_matcher = patch_matcher

    def get_nnf_dict(self, graph, guide, style):
        nnf_dict = {}
        for u, v in tqdm(graph.edges, desc="Estimating NNF"):
            nnf, _ = self.patch_matcher.estimate_nnf(
                source_guide=guide[u],
                target_guide=guide[v],
                source_style=style[u]
            )
            nnf_dict[(u, v)] = nnf.cpu()
        return nnf_dict


class VideoPatchMatchSmoother:
    def __init__(self, engine_name="Pytorch", bin_path=None, cache_path=None, num_iter=1, postprocessing={}):
        self.operating_space = "pixel"
        self.engine_name = engine_name
        self.bin_path = bin_path
        self.cache_path = cache_path
        self.num_iter = num_iter
        self.postprocessing = postprocessing

    def remap_and_blend_left(
        self,
        patch_matcher,
        guide,
        style,
        remaping_operator,
        blending_operator,
        nnf_engine
    ):
        n = len(guide)
        graph = LeftVideoGraph(n)
        nnf_dict = nnf_engine.get_nnf_dict(graph, guide, style)
        remap_table = [[style[i]] for i in range(n)]
        blend_table = [[style[i]] for i in range(n)]
        level = 0
        while True:
            edges = graph.query_edge(level)
            level += 1
            if len(edges)==0:
                break
            for u, v in edges:
                nnf = nnf_dict[(u, v)]
                remaping_result = remaping_operator(nnf, blend_table[u][-1])
                remap_table[v].append(remaping_result)
                blending_result = blending_operator(remap_table[v])
                blend_table[v].append(blending_result)
        blending_inputs = []
        for i in range(n):
            nodes = graph.query(i)
            blending_input = []
            for u in nodes:
                if u==i:
                    if len(remap_table[u])==1:
                        continue
                    else:
                        remaping_result = blending_operator(remap_table[u][1:])
                else:
                    nnf = nnf_dict[(u, i)]
                    remaping_result = remaping_operator(nnf, blend_table[u][-1])
                blending_input.append(remaping_result)
            blending_inputs.append(blending_input)
        return blending_inputs

    def remap_and_blend(
        self,
        patch_matcher,
        guide,
        style,
        remaping_operator,
        blending_operator,
        nnf_engine = None,
    ):
        if nnf_engine is None:
            nnf_engine = PytorchEngine(patch_matcher)
        blending_inputs_l = self.remap_and_blend_left(
            patch_matcher,
            guide,
            style,
            remaping_operator,
            blending_operator,
            nnf_engine
        )
        blending_inputs_r = self.remap_and_blend_left(
            patch_matcher,
            guide[::-1],
            style[::-1],
            remaping_operator,
            blending_operator,
            nnf_engine
        )[::-1]
        frames = []
        for l, m, r in zip(blending_inputs_l, style, blending_inputs_r):
            frame = blending_operator(l + [m] + r) / len(style)
            frames.append(frame)
        return frames

    def prepare(self, frames):
        self.guide = [PIL_to_tensor(frame) for frame in frames]
        height, width = self.guide[0].shape[0], self.guide[0].shape[1]
        self.patch_matcher = PyramidPatchMatcher(height, width, patch_size=11, num_iter=6)

    def postprocessing_contrast(self, style, rate):
        style = [ImageEnhance.Contrast(i).enhance(rate) for i in style]
        return style

    def postprocessing_sharpness(self, style, rate):
        style = [ImageEnhance.Sharpness(i).enhance(rate) for i in style]
        return style

    def image_postprocessing(self, images):
        for name in self.postprocessing:
            rate = self.postprocessing[name]
            if name == "contrast":
                images = self.postprocessing_contrast(images, rate)
            elif name == "sharpness":
                images = self.postprocessing_sharpness(images, rate)
        return images

    def smooth(self, frames):
        # Two operators
        remaping_operator = lambda nnf, image: self.patch_matcher.apply_nnf_to_image(nnf, image).cpu()
        blending_operator = lambda frames: torch.stack(frames).sum(dim=0)
        # NNF Estimation Engine
        if self.engine_name == "Pytorch":
            nnf_engine = PytorchEngine(self.patch_matcher)
        elif self.engine_name == "Ebsynth":
            nnf_engine = EbsynthBinaryEngine(self.bin_path, self.cache_path)
        # Postprocessing
        frames = self.image_postprocessing(frames)
        # Images to tensors
        style = [PIL_to_tensor(frame) for frame in frames]
        # Iteration
        for it in range(self.num_iter):
            # Remap and blend tensors
            frames = self.remap_and_blend(self.patch_matcher, self.guide, style, remaping_operator, blending_operator, nnf_engine)
        # Tensors to frames
        frames = [tensor_to_PIL(frame) for frame in frames]
        # Postprocessing
        frames = self.image_postprocessing(frames)
        return frames