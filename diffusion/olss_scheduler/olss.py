import torch
from tqdm import tqdm


class OLSSSchedulerModel(torch.nn.Module):

    def __init__(self, wx, we):
        super(OLSSSchedulerModel, self).__init__()
        assert len(wx.shape)==1 and len(we.shape)==2
        T = wx.shape[0]
        assert T==we.shape[0] and T==we.shape[1]
        self.register_parameter("wx", torch.nn.Parameter(wx))
        self.register_parameter("we", torch.nn.Parameter(we))

    def forward(self, t, xT, e_prev):
        assert t - len(e_prev) + 1 == 0
        x = xT*self.wx[t]
        for e, we in zip(e_prev, self.we[t]):
            x += e*we
        return x.to(xT.dtype)


class OLSSScheduler():

    def __init__(self, timesteps, model):
        self.timesteps = timesteps
        self.model = model
        self.init_noise_sigma = 1.0
        self.order = 1

    @staticmethod
    def load(path):
        timesteps, wx, we = torch.load(path, map_location="cpu")
        model = OLSSSchedulerModel(wx, we)
        return OLSSScheduler(timesteps, model)

    def save(self, path):
        timesteps, wx, we = self.timesteps, self.model.wx, self.model.we
        torch.save((timesteps, wx, we), path)

    def set_timesteps(self, num_inference_steps, device = "cuda"):
        self.xT = None
        self.e_prev = []
        self.t_prev = -1
        self.model = self.model.to(device)
        self.timesteps = self.timesteps.to(device)

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs):
        return sample

    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        *args, **kwargs
    ):
        t = self.timesteps.tolist().index(timestep)
        assert self.t_prev==-1 or t==self.t_prev+1
        if self.t_prev==-1:
            self.xT = sample
        self.e_prev.append(model_output)
        x = self.model(t, self.xT, self.e_prev)
        if t+1==len(self.timesteps):
            self.xT = None
            self.e_prev = []
            self.t_prev = -1
        else:
            self.t_prev = t
        return (x,)


class OLSSSolver:

    def __init__(self):
        pass

    def solve_linear_regression(self, X, Y):
        X = X.to(torch.float64)
        Y = Y.to(torch.float64)
        # coef = torch.linalg.pinv(X.T @ X) @ X.T @ Y
        coef = torch.linalg.lstsq(X, Y).solution
        return coef

    def solve_scheduer_parameters(self, xT, e_prev, x):
        # prepare
        xe_prev = torch.concat([xT, e_prev], dim=0)
        xe_prev = xe_prev.reshape(xe_prev.shape[0], -1)
        x = x.flatten()
        # solve the ordinary least squares problem
        coef = self.solve_linear_regression(xe_prev.T, x)
        # split the parameters
        wx, we = coef[:1], coef[1:]
        # error
        x_pred = torch.matmul(coef.unsqueeze(0), xe_prev.to(torch.float64)).squeeze(0)
        err = torch.nn.functional.mse_loss(x_pred, x).tolist()
        return wx, we, err

    @torch.no_grad()
    def resolve_diffusion_process(self,
                                  steps_accelerate,
                                  t_path,
                                  x_path,
                                  e_path,
                                  i_path=None):
        steps_inference = t_path.shape[0]
        # accelerate path
        if i_path is None:
            i_path = torch.arange(0, steps_inference, steps_inference//steps_accelerate)[:steps_accelerate]
        t_path = t_path[i_path]
        x_path = torch.concat([x_path[i_path], x_path[-1:]])
        e_path = e_path[i_path]
        # parameters
        wx = torch.zeros(steps_accelerate, dtype=torch.float64)
        we = torch.zeros((steps_accelerate, steps_accelerate), dtype=torch.float64)
        for i in range(steps_accelerate):
            x = x_path[i+1]
            xT = x_path[0:1]
            e_prev = e_path[:i+1]
            wx[i], we[i, :i+1], _ = self.solve_scheduer_parameters(xT, e_prev, x)
        return t_path, wx, we

    def search_next_step_with_error_limit(self, x_prev, e_prev, x_flat, i_lowerbound, max_error):
        i_next = i_lowerbound
        i_upperbound = len(x_flat)-1
        while i_upperbound>i_lowerbound:
            i_next = (i_lowerbound + i_upperbound + 1)//2
            x_goal = x_flat[i_next]
            _, _, err_step = self.solve_scheduer_parameters(x_prev, e_prev, x_goal)
            if err_step>max_error:
                i_upperbound = i_next - 1
            else:
                i_lowerbound = i_next
        i_next = i_lowerbound
        return i_next

    def search_path_with_error_limit(self,
                                    max_steps,
                                    t_path,
                                    x_path,
                                    e_path,
                                    max_error):
        # prepare for calculation
        num_inference_steps = t_path.shape[0]
        x_flat = x_path.reshape(num_inference_steps+1, -1)
        e_flat = e_path.reshape(num_inference_steps, -1)
        # search (greedy)
        i_path_acc = [0]
        for step in range(max_steps):
            x_prev = x_flat[i_path_acc[step:step+1]]
            e_prev = e_flat[i_path_acc]
            i_lowerbound = i_path_acc[step] + 1
            i_next = self.search_next_step_with_error_limit(x_prev, e_prev, x_flat, i_lowerbound, max_error)
            if i_next == num_inference_steps:
                return i_path_acc
            else:
                i_path_acc.append(i_next)
        return None

    @torch.no_grad()
    def resolve_diffusion_process_graph(self,
                                        num_accelerate_steps,
                                        t_path,
                                        x_path,
                                        e_path,
                                        max_iter = 30,
                                        verbose = 0):
        error_l, error_r = 0.0, 10.0
        for it in tqdm(range(max_iter), desc="OLSS is solving the parameters"):
            error_m = (error_l + error_r) / 2
            path = self.search_path_with_error_limit(num_accelerate_steps, t_path, x_path, e_path, error_m)
            if path is None:
                error_l = error_m
            else:
                error_r = error_m
            if verbose>0:
                print(f"search for path with maximum error: {error_m}")
                if path is None:
                    print("    cannot find such path")
                else:
                    print(f"    find a path with length {len(path)}: {path}")
        path = self.search_path_with_error_limit(num_accelerate_steps, t_path, x_path, e_path, error_r)
        timesteps, wx, we = self.resolve_diffusion_process(num_accelerate_steps, t_path, x_path, e_path, i_path=path)
        return timesteps, wx, we


class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.catch_x, self.catch_e, self.catch_x_ = {}, {}, {}
        self.olss_scheduler = None

    def set_timesteps(self, num_inference_steps, **kwargs):
        if self.olss_scheduler is None:
            result = self.scheduler.set_timesteps(num_inference_steps, **kwargs)
            self.timesteps = self.scheduler.timesteps
            self.init_noise_sigma = self.scheduler.init_noise_sigma
            self.order = self.scheduler.order
            return result
        else:
            result = self.olss_scheduler.set_timesteps(num_inference_steps, **kwargs)
            self.timesteps = self.olss_scheduler.timesteps
            self.init_noise_sigma = self.scheduler.init_noise_sigma
            self.order = self.scheduler.order
            return result

    def step(self, model_output, timestep, sample, **kwargs):
        if self.olss_scheduler is None:
            result = self.scheduler.step(model_output, timestep, sample, **kwargs)
            timestep = timestep.tolist()
            if timestep not in self.catch_x:
                self.catch_x[timestep] = []
                self.catch_e[timestep] = []
                self.catch_x_[timestep] = []
            self.catch_x[timestep].append(sample.clone().detach().cpu())
            self.catch_e[timestep].append(model_output.clone().detach().cpu())
            self.catch_x_[timestep].append(result[0].clone().detach().cpu())
            return result
        else:
            result = self.olss_scheduler.step(model_output, timestep, sample, **kwargs)
            return result
    
    def scale_model_input(self, sample, timestep):
        return sample
    
    def add_noise(self, original_samples, noise, timesteps):
        result = self.scheduler.add_noise(original_samples, noise, timesteps)
        return result
    
    def get_path(self):
        t_path = sorted([t for t in self.catch_x], reverse=True)
        x_path, e_path = [], []
        for t in t_path:
            x = torch.cat(self.catch_x[t], dim=0)
            x_path.append(x)
            e = torch.cat(self.catch_e[t], dim=0)
            e_path.append(e)
        t_final = t_path[-1]
        x_final = torch.cat(self.catch_x_[t_final], dim=0)
        x_path.append(x_final)
        t_path = torch.tensor(t_path, dtype=torch.int32)
        x_path = torch.stack(x_path)
        e_path = torch.stack(e_path)
        return t_path, x_path, e_path
    
    def prepare_olss(self, num_accelerate_steps):
        solver = OLSSSolver()
        t_path, x_path, e_path = self.get_path()
        timesteps, wx, we = solver.resolve_diffusion_process_graph(
            num_accelerate_steps, t_path, x_path, e_path)
        self.olss_model = OLSSSchedulerModel(wx, we)
        self.olss_scheduler = OLSSScheduler(timesteps, self.olss_model)
        