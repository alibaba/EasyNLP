import cv2
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import cupy as cp


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


class VideoWithOperator:
    def __init__(self, frames_guide, frames_style, patch_size=21, threads_per_block=8, num_iter=6, gpu_id=0, guide_weight=100.0):
        self.frames_guide = frames_guide
        self.frames_style = frames_style
        image_height, image_width, _ = frames_style[0].shape
        self.patch_match_engine = PyramidPatchMatcher(
            image_height, image_width, channel=3, patch_size=patch_size,
            threads_per_block=threads_per_block, num_iter=num_iter,
            gpu_id=gpu_id, guide_weight=guide_weight
        )

    def remap(self, x, i, j):
        source_style, num_blend = x
        nnf, target_style = self.patch_match_engine.estimate_nnf(
            source_guide=self.frames_guide[i],
            target_guide=self.frames_guide[j],
            source_style=source_style
        )
        target_style = target_style.get()
        return target_style, num_blend

    def blend(self, x):
        sum_num_blend = sum([num_blend for style, num_blend in x])
        weighted_frames = [style * (num_blend / sum_num_blend) for style, num_blend in x]
        mean_frame = np.stack(weighted_frames).sum(axis=0)
        return mean_frame, sum_num_blend

    def __call__(self, i):
        return self.frames_style[i], 1

    def __len__(self):
        return len(self.frames_style)


class FastBlendingAlgorithm:
    def __init__(self, data):
        self.data = data
        n = len(self.data)
        self.remapping_table = [[self.data(i)] for i in range(n)]
        self.blending_table = [[self.data(i)] for i in range(n)]
        level = 1
        while (1<<level)<=n:
            for i in tqdm(range((1<<level)-1, n, 1<<level), desc=f"Building remapping table (level-{level})"):
                source, target = i - (1<<level-1), i
                remapping_result = self.data.remap(self.blending_table[source][-1], source, target)
                self.remapping_table[target].append(remapping_result)
                blending_result = self.data.blend(self.remapping_table[target])
                self.blending_table[target].append(blending_result)
            level += 1

    def tree_query(self, leftbound, rightbound):
        node_list = []
        node_index = rightbound
        while node_index>=leftbound:
            node_level = 0
            while (1<<node_level)&node_index and node_index-(1<<node_level+1)+1>=leftbound:
                node_level += 1
            node_list.append((node_index, node_level))
            node_index -= 1<<node_level
        return node_list
    
    def query(self, leftbound, rightbound):
        node_list = self.tree_query(leftbound, rightbound)
        result = []
        for node_index, node_level in node_list:
            node_value = self.blending_table[node_index][node_level]
            if node_index!=rightbound:
                node_value = self.data.remap(node_value, node_index, rightbound)
            result.append(node_value)
        result = self.data.blend(result)
        return result


class ImagePostProcessor:
    def __init__(self, postprocessing):
        self.postprocessing = postprocessing

    def postprocessing_contrast(self, style, rate):
        style = [ImageEnhance.Contrast(i).enhance(rate) for i in style]
        return style

    def postprocessing_sharpness(self, style, rate):
        style = [ImageEnhance.Sharpness(i).enhance(rate) for i in style]
        return style

    def __call__(self, images):
        for name in self.postprocessing:
            rate = self.postprocessing[name]
            if name == "contrast":
                images = self.postprocessing_contrast(images, rate)
            elif name == "sharpness":
                images = self.postprocessing_sharpness(images, rate)
        return images


class PySynthSmoother:
    def __init__(self, speed="slowest", window_size=3, postprocessing={}, ebsynth_config={}):
        self.speed = speed
        self.window_size = window_size
        self.postprocessor = ImagePostProcessor(postprocessing)
        self.ebsynth_config = ebsynth_config
        self.operating_space = "pixel"

    def prepare(self, images):
        self.frames_guide = images

    def PIL_to_numpy(self, frames):
        return [np.array(frame).astype(np.float32)/255 for frame in frames]

    def numpy_to_PIL(self, frames):
        return [Image.fromarray(np.clip((frame * 255), 0, 255).astype("uint8")) for frame in frames]

    def smooth_slowest(self, frames_guide, frames_style):
        data = VideoWithOperator(frames_guide, frames_style, **self.ebsynth_config)
        frames_output = []
        for frame_id in tqdm(range(len(data))):
            remapped_frames = [data(frame_id)]
            for i in range(frame_id - self.window_size, frame_id + self.window_size + 1):
                if i<0 or i>=len(data) or i==frame_id:
                    continue
                remapped_frame = data.remap(data(i), i, frame_id)
                remapped_frames.append(remapped_frame)
            blended_frame, _ = data.blend(remapped_frames)
            frames_output.append(blended_frame)
        return frames_output

    def smooth_fastest(self, frames_guide, frames_style):
        # left
        data = VideoWithOperator(frames_guide, frames_style, **self.ebsynth_config)
        algo = FastBlendingAlgorithm(data)
        remapped_frames_l = []
        for frame_id in tqdm(range(len(data)), desc="Remapping and blending (left part)"):
            bound = max(frame_id - self.window_size, 0)
            remapped_frames_l.append(algo.query(bound, frame_id))
        # right
        data = VideoWithOperator(frames_guide[::-1], frames_style[::-1], **self.ebsynth_config)
        algo = FastBlendingAlgorithm(data)
        remapped_frames_r = []
        for frame_id in tqdm(range(len(data)), desc="Remapping and blending (right part)"):
            bound = max(frame_id - self.window_size, 0)
            remapped_frames_r.append(algo.query(bound, frame_id))
        remapped_frames_r = remapped_frames_r[::-1]
        # merge
        frames_output = []
        data = VideoWithOperator(frames_guide, frames_style, **self.ebsynth_config)
        for frame_id in range(len(data)):
            frame, _ = data(frame_id)
            frame_output, _ = data.blend([
                remapped_frames_l[frame_id],
                (frame, -1),
                remapped_frames_r[frame_id]
            ])
            frames_output.append(frame_output)
        return frames_output

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
        frames_output = self.postprocessor(frames_output)
        return frames_output

