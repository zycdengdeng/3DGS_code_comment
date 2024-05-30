"""
高斯model的基本框架：
1.导入必要的模块和函数
这些模块和函数主要用于数学运算、点云数据处理和文件操作等。
2.类初始化和设置函数
__init__ 方法初始化模型的基本参数和张量，并调用 setup_functions 方法设置各种激活和变换函数。
3.捕获和恢复模型状态
提供保存和恢复模型状态的方法，以便模型的持久化和继续训练。
4.创建模型参数
从给定的点云数据初始化模型的参数，包括位置、颜色、特征、尺度、旋转和不透明度等。
4.设置训练参数
设置训练过程中的参数，包括初始化学习率、优化器和学习率调度器。
5. 其他辅助方法
包括调整球谐函数阶数、获取模型参数、保存和加载点云文件等辅助功能。
"""
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
#反sigmoid 就是把0-1再返回原始数据
#生成指数学习率的，随时间减小
#生产旋转矩阵，就可以控制物体的转动方向和角度
from torch import nn
import os
from utils.system_utils import mkdir_p
#创建目录的实用函数 简化文件夹管理

from plyfile import PlyData, PlyElement
#专门用于处理 PLY 文件格式
#PLY（Polygon File Format 或 Stanford Triangle Format）是一种用于表示三维模型数据的文件格式，常用于存储从3D扫描仪获得的三维数据。
from utils.sh_utils import RGB2SH
#用于将 RGB 颜色空间数据转换为球谐函数系数的函数。
from simple_knn._C import distCUDA2
#distCUDA2 是一个使用 CUDA 加速计算的函数，用于计算两个集合之间的距离。
from utils.graphics_utils import BasicPointCloud
#加载，操作，可视化点云数据
from utils.general_utils import strip_symmetric, build_scaling_rotation
#它可能用于提取对称矩阵的特定部分或简化对称数据的处理
#这个函数可能是用于构建包含缩放和旋转变换的矩阵。
class GaussianModel:

    def setup_functions(self): #用于设置一些激活函数和变换函数  这一步就是配置工具
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):#构建协方差矩阵，该函数接受 scaling（尺度）、scaling_modifier（尺度修正因子）、rotation（旋转）作为参数
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)   #乘转置
            symm = strip_symmetric(actual_covariance)  #取上三角部分
            return symm #最终返回对称的协方差矩阵。

        #在神经网络中，激活函数（activation function）是用来对输入信号进行非线性变换的。
        self.scaling_activation = torch.exp #将尺度激活函数设置为指数函数。
        self.scaling_inverse_activation = torch.log #将尺度逆激活函数设置为对数函数。

        self.covariance_activation = build_covariance_from_scaling_rotation #将协方差激活函数设置为上述定义的 build_covariance_from_scaling_rotation 函数。

        self.opacity_activation = torch.sigmoid #将不透明度激活函数设置为 sigmoid 函数。
        self.inverse_opacity_activation = inverse_sigmoid #将不透明度逆激活函数设置为一个名为 inverse_sigmoid 的函数

        self.rotation_activation = torch.nn.functional.normalize #用于归一化旋转矩阵，方向和原始tensor相同


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0  #球谐阶数
        self.max_sh_degree = sh_degree   #最大球谐阶数
        # 存储不同信息的张量（tensor）
        self._xyz = torch.empty(0) #空间位置
        self._features_dc = torch.empty(0) #存储特征数据
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)  #椭球的形状尺度  半轴长度
        self._rotation = torch.empty(0) #椭球的旋转
        self._opacity = torch.empty(0)  #不透明度
        self.max_radii2D = torch.empty(0) #最大半径
        self.xyz_gradient_accum = torch.empty(0) #位置梯度
        self.denom = torch.empty(0)  #需要的分母数据
        self.optimizer = None  #初始化优化器为 None。
        self.percent_dense = 0  #初始化百分比密度为0。
        self.spatial_lr_scale = 0 #初始化空间学习速率缩放为0。
        self.setup_functions() #调用 setup_functions 方法设置各种激活和变换函数
        
    #捕获当前对象的状态
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self): #每 1000 次迭代，增加球谐函数的阶数。
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float): #用于从给定的点云数据 pcd 创建对象的初始化状态。
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() #转换点云数据为浮点型 CUDA 张量
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()#三个维度的张量大小
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
    #clamp_min 是 PyTorch 中的一个内置函数，用于将张量中的最小值限制在一个指定的值之上
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
    #初始化不透明度张量
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
#我们把点的位置（x, y, z）变成模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
#为训练过程做准备
    def training_setup(self, training_args):
        #初始化训练参数
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#优化器参数列表：定义了需要优化的参数、对应的初始学习率和名称。
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration): #更新学习率。
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):# 构建属性列表
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path): #保存为 PLY 文件
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):#重置不透明度 
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path): #这个方法的目的是从PLY文件中加载各种数据，并将这些数据存储为类中的属性，以便后续的操作和训练。
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name): #替换优化器中的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):#修剪优化器中的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将新的密集化点的相关特征保存在一个字典中。
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) #将字典中的张量连接（concatenate）成可优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便于在优化器中进行处理。
        # 更新模型中原始点集的相关特征，使用新的密集化后的特征。
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重新初始化一些用于梯度计算和密集化操作的变量。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] #获取初始点的数量。
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") #创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中。
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) #创建一个掩码，标记那些梯度大于等于指定阈值的点。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # 一步过滤掉那些缩放（scaling）大于一定百分比的场景范围的点。

        # 为每个点生成新的样本，其中 stds 是点的缩放，means 是均值。
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds) #使用均值和标准差生成样本。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) #为每个点构建旋转矩阵，并将其重复 N 次。
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) #将旋转后的样本点添加到原始点的位置。
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) #生成新的缩放参数。
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1) #将旋转矩阵重复 N 次。
        # 将原始点的特征重复 N 次。
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作（此处跟densify_and_clone一样）。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 根据修剪过滤器，修剪模型中的一些参数。
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) #建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据。
        
        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等。
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom #计算密度估计的梯度
        grads[grads.isnan()] = 0.0 #将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性。

        self.densify_and_clone(grads, max_grad, extent) #对under reconstruction的区域进行稠密化和复制操作
        self.densify_and_split(grads, max_grad, extent) #对over reconstruction的区域进行稠密化和分割操作

        prune_mask = (self.get_opacity < min_opacity).squeeze() #创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度。
        if max_screen_size: #如何设置了相机的范围，
            big_points_vs = self.max_radii2D > max_screen_size #创建一个掩码，标记在图像空间中半径大于指定阈值的点。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent #创建一个掩码，标记在世界空间中尺寸大于指定阈值的点。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) #将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码。
        self.prune_points(prune_mask) #：根据修剪掩码，修剪模型中的一些参数。

        torch.cuda.empty_cache() #清理 GPU 缓存，释放一些内存

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
