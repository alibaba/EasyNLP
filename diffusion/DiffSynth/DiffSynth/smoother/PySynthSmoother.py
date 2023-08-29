import torch, os, cv2
from PIL import Image, ImageEnhance
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
import cupy as cp
from DiffSynth.utils import save_video, save_images


remapping_kernel = cp.RawKernel(r'''
extern "C" __global__
void remap(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const float* source_style,
    const int* nnf,
    float* target_style
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= height or y >= width) return;
    const int pid = x * width + y;
    const int min_px = x < r ? -x : -r;
    const int max_px = x + r > height - 1 ? height - 1 - x : r;
    const int min_py = y < r ? -y : -r;
    const int max_py = y + r > width - 1 ? width - 1 - y : r;
    for (int px = min_px; px <= max_px; px++){
        for (int py = min_py; py <= max_py; py++){
            const int nid = (x + px) * width + y + py;
            const int x_ = nnf[nid*2] - px;
            const int y_ = nnf[nid*2+1] - py;
            const int pid_ = (x_ + r) * (width + patch_size - 1) + (y_ + r);
            for (int c = 0; c < channel; c++){
                target_style[pid * channel + c] += source_style[pid_ * channel + c];
            }
        }
    }
    for (int c = 0; c < channel; c++){
        target_style[pid * channel + c] /= (max_px - min_px + 1) * (max_py - min_py + 1);
    }
}
''', 'remap')


patch_error_kernel = cp.RawKernel(r'''
extern "C" __global__
void patch_error(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const float* source,
    const int* nnf,
    const float* target,
    float* error
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= height or y >= width) return;
    const int x_ = nnf[(x * width + y)*2];
    const int y_ = nnf[(x * width + y)*2+1];
    float e = 0;
    for (int px = -r; px <= r; px++){
        for (int py = -r; py <= r; py++){
            const int pid = (x + r + px) * (width + patch_size - 1) + y + r + py;
            const int pid_ = (x_ + r + px) * (width + patch_size - 1) + y_ + r + py;
            for (int c = 0; c < channel; c++){
                const float diff = target[pid * channel + c] - source[pid_ * channel + c];
                e += diff * diff;
            }
        }
    }
    error[x * width + y] = e;
}
''', 'patch_error')


class PatchMatcher:
    def __init__(self, height, width, channel, patch_size, threads_per_block=8, num_iter=6, gpu_id=0, guide_weight=100.0):
        self.height = height
        self.width = width
        self.channel = channel
        self.patch_size = patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight

        self.r = (patch_size - 1) // 2
        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block
        )
        self.block = (threads_per_block, threads_per_block)
        with cp.cuda.Device(self.gpu_id):
            self.min_nnf_value = cp.zeros((self.height, self.width, 2), dtype=cp.int32)
            self.max_nnf_value = cp.stack((
                cp.ones((self.height, self.width), dtype=cp.int32) * (self.height - 1),
                cp.ones((self.height, self.width), dtype=cp.int32) * (self.width - 1),
            ), axis=2)

    def pad_image(self, image):
        return cp.pad(image, ((self.r, self.r), (self.r, self.r), (0, 0)))

    def unpad_image(self, image):
        return image[self.r: -self.r, self.r: -self.r]

    def apply_nnf_to_image(self, nnf, source):
        target = cp.zeros((self.height, self.width, self.channel), dtype=cp.float32)
        remapping_kernel(
            self.grid,
            self.block,
            (self.height, self.width, self.channel, self.patch_size, source, nnf, target)
        )
        return target

    def get_patch_error(self, source, nnf, target):
        error = cp.zeros((self.height, self.width), dtype=cp.float32)
        patch_error_kernel(
            self.grid,
            self.block,
            (self.height, self.width, self.channel, self.patch_size, source, nnf, target, error)
        )
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        error_style = self.get_patch_error(source_style, nnf, target_style)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        nnf = cp.clip(nnf, self.min_nnf_value, self.max_nnf_value)
        return nnf

    def random_step(self, nnf, w):
        step = cp.stack([
            cp.random.random((self.height, self.width), dtype=cp.float32) * (self.height * 2) - self.height,
            cp.random.random((self.height, self.width), dtype=cp.float32) * (self.width * 2) - self.width
        ], axis=2)
        step = (step * w).astype(cp.int32)
        upd_nnf = self.clamp_bound(nnf + step)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        if d==0:
            upd_nnf = cp.concatenate([nnf[:1, :], nnf[:-1, :]], axis=0)
            upd_nnf[:, :, 0] += 1
        elif d==1:
            upd_nnf = cp.concatenate([nnf[:, :1], nnf[:, :-1]], axis=1)
            upd_nnf[:, :, 1] += 1
        elif d==2:
            upd_nnf = cp.concatenate([nnf[1:, :], nnf[-1:, :]], axis=0)
            upd_nnf[:, :, 0] -= 1
        elif d==3:
            upd_nnf = cp.concatenate([nnf[:, 1:], nnf[:, -1:]], axis=1)
            upd_nnf[:, :, 1] -= 1
        upd_nnf = self.clamp_bound(upd_nnf)
        return upd_nnf

    def update(self, source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf):
        upd_err = self.get_error(source_guide, target_guide, source_style, target_style, upd_nnf)
        upd_idx = (upd_err < err)
        nnf[upd_idx] = upd_nnf[upd_idx]
        err[upd_idx] = upd_err[upd_idx]
        return nnf, err

    def propagation(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for d in cp.random.permutation(4):
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err
        
    def random_search(self, source_guide, target_guide, source_style, target_style, nnf, err):
        w = 1.0
        while w*max(self.height, self.width)>1:
            upd_nnf = self.random_step(nnf, w)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
            w *= 0.5
        return nnf, err

    def iteration(self, source_guide, target_guide, source_style, target_style, nnf, err):
        nnf, err = self.propagation(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.random_search(source_guide, target_guide, source_style, target_style, nnf, err)
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        with cp.cuda.Device(self.gpu_id):
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)
            target_style = self.pad_image(self.apply_nnf_to_image(nnf, source_style))

            err = self.get_error(source_guide, target_guide, source_style, target_style, nnf)
            for it in range(self.num_iter):
                nnf, err = self.iteration(source_guide, target_guide, source_style, target_style, nnf, err)
                target_style = self.pad_image(self.apply_nnf_to_image(nnf, source_style))
        target_style = self.unpad_image(target_style)
        return nnf, target_style


class PyramidPatchMatcher:
    def __init__(self, image_height, image_width, channel, patch_size, threads_per_block=8, num_iter=6, gpu_id=0, guide_weight=100.0):
        self.pyramid_level = int(np.log2(min(image_height, image_width) / patch_size))
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.patch_size = patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        for level in range(self.pyramid_level):
            height = image_height//(2**(self.pyramid_level - 1 - level))
            width = image_width//(2**(self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(PatchMatcher(
                height, width, channel, patch_size,
                threads_per_block=threads_per_block, num_iter=num_iter, gpu_id=gpu_id, guide_weight=guide_weight
            ))

    def resample_image(self, image, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        image = cv2.resize(image.get(), (width, height), interpolation=cv2.INTER_AREA)
        image = cp.array(image, dtype=cp.float32)
        return image

    def initialize_nnf(self):
        height, width = self.pyramid_heights[0], self.pyramid_widths[0]
        nnf = cp.stack([
            cp.random.randint(0, height, (height, width), dtype=cp.int32),
            cp.random.randint(0, width, (height, width), dtype=cp.int32)
        ], axis=2)
        return nnf

    def update_nnf(self, nnf, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        scale = (height / nnf.shape[0] + width / nnf.shape[1]) / 2
        nnf = nnf.astype(cp.float32) * scale
        nnf = cv2.resize(nnf.get(), (width, height), interpolation=cv2.INTER_LINEAR)
        nnf = cp.array(nnf, dtype=cp.int32)
        return nnf

    def apply_nnf_to_image(self, nnf, image):
        with cp.cuda.Device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            image = self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
        return image

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf=None):
        with cp.cuda.Device(self.gpu_id):
            if not isinstance(source_guide, cp.ndarray):
                source_guide = cp.array(source_guide)
            if not isinstance(target_guide, cp.ndarray):
                target_guide = cp.array(target_guide)
            if not isinstance(source_style, cp.ndarray):
                source_style = cp.array(source_style)
            for level in range(self.pyramid_level):
                nnf = self.initialize_nnf() if level==0 else self.update_nnf(nnf, level)
                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)
                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )
        return nnf, target_style


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


class NNFCache:
    def __init__(self):
        pass

    def get_nnf_dict(self, graph, frames_guide, frames_style):
        nnf_dict = {}
        for u, v in tqdm(graph.edges, desc="Estimating NNF"):
            nnf, _ = self.patch_matcher.estimate_nnf(
                source_guide=frames_guide[u],
                target_guide=frames_guide[v],
                source_style=frames_style[u]
            )
            nnf_dict[(u, v)] = nnf.cpu()
        return nnf_dict


class PySynthSmoother:
    def __init__(self, patch_size=21, threads_per_block=8, num_iter=6, gpu_id=0, guide_weight=100.0, speed="slowest", window_size=10, postprocessing={}):
        self.patch_size = patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.speed = speed
        self.window_size = window_size
        self.postprocessing = postprocessing
        self.operating_space = "pixel"

    def reset(self, image_height, image_width):
        self.patch_match_engine = PyramidPatchMatcher(
            image_height, image_width, channel=3, patch_size=self.patch_size,
            threads_per_block=self.threads_per_block, num_iter=self.num_iter,
            gpu_id=self.gpu_id, guide_weight=self.guide_weight
        )

    def prepare(self, images):
        self.frames_guide = images
        image_width, image_height = images[0].size
        self.reset(image_height, image_width)

    def PIL_to_numpy(self, frames):
        return [np.array(frame).astype(np.float32)/255 for frame in frames]

    def numpy_to_PIL(self, frames):
        return [Image.fromarray(np.clip((frame * 255), 0, 255).astype("uint8")) for frame in frames]

    def remapping_operator(self, nnf, image):
        with cp.cuda.Device(self.gpu_id):
            nnf = cp.array(nnf, dtype=cp.int32)
            image = cp.array(image, dtype=cp.float32)
            image = self.patch_match_engine.apply_nnf_to_image(nnf, image)
            image = image.get()
        return image

    def blending_operator(self, frames):
        frame = np.stack(frames).sum(axis=0)
        return frame

    def smooth_slowest(self, frames_guide, frames_style):
        frames_output = []
        for frame_id in tqdm(range(len(frames_style))):
            remapped_frames = [frames_style[frame_id]]
            for i in range(frame_id - self.window_size, frame_id + self.window_size + 1):
                if i<0 or i>=len(frames_style) or i==frame_id:
                    continue
                _, remapped_frame = self.patch_match_engine.estimate_nnf(frames_guide[i], frames_guide[frame_id], frames_style[i])
                remapped_frames.append(remapped_frame.get())
            blended_frame = self.blending_operator(remapped_frames) / len(remapped_frames)
            frames_output.append(blended_frame)
        return frames_output

    def remap_and_blend_left(self, frames_guide, frames_style):
        n = len(frames_guide)
        graph = LeftVideoGraph(n)
        # Estimate NNF
        nnf_dict = {}
        for u, v in tqdm(graph.edges, desc="Estimating NNF"):
            nnf, _ = self.patch_match_engine.estimate_nnf(
                source_guide=frames_guide[u],
                target_guide=frames_guide[v],
                source_style=frames_style[u]
            )
            nnf_dict[(u, v)] = nnf.get()
        # remap_table and blend_table
        remap_table = [[frames_style[i]] for i in range(n)]
        blend_table = [[frames_style[i]] for i in range(n)]
        level = 0
        while True:
            edges = graph.query_edge(level)
            level += 1
            if len(edges)==0:
                break
            for u, v in edges:
                nnf = nnf_dict[(u, v)]
                remaping_result = self.remapping_operator(nnf, blend_table[u][-1])
                remap_table[v].append(remaping_result)
                blending_result = self.blending_operator(remap_table[v])
                blend_table[v].append(blending_result)
        # calculate remapping prefix sum
        blending_inputs = []
        for i in tqdm(range(n), desc="Remapping frames"):
            blending_input = []
            # sum of 0...i-1
            nodes = graph.query(i)
            for u in nodes:
                if u==i:
                    if len(remap_table[u])==1:
                        continue
                    else:
                        remaping_result = self.blending_operator(remap_table[u][1:])
                else:
                    nnf = nnf_dict[(u, i)]
                    remaping_result = self.remapping_operator(nnf, blend_table[u][-1])
                blending_input.append(remaping_result)
            blending_inputs.append(blending_input)
        return blending_inputs

    def smooth_fastest(self, frames_guide, frames_style):
        n = len(frames_guide)
        prefix_sum = self.remap_and_blend_left(frames_guide, frames_style)
        suffix_sum = self.remap_and_blend_left(frames_guide[::-1], frames_style[::-1])[::-1]
        frames_output = []
        for i, l, m, r in zip(range(n), prefix_sum, frames_style, suffix_sum):
            window_size = min(i + self.window_size, n - 1) - max(i - self.window_size, 0) + 1
            frame = self.blending_operator(l + [m] + r) / n
            frames_output.append(frame)
        return frames_output

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

    def smooth(self, frames_style):
        frames_guide = self.PIL_to_numpy(self.frames_guide)
        frames_style = self.PIL_to_numpy(frames_style)
        if self.speed == "slowest":
            frames_output = self.smooth_slowest(frames_guide, frames_style)
        elif self.speed == "fastest":
            frames_output = self.smooth_fastest(frames_guide, frames_style)
        else:
            raise NotImplementedError()
        frames_output = self.numpy_to_PIL(frames_output)
        frames_output = self.image_postprocessing(frames_output)
        return frames_output
