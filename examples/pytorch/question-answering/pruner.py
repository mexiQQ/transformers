import torch
import numpy

__all__ = ["Prune"]


class Prune:
    def __init__(
        self,
        model,
        pretrain_step: int = 0,
        sparse_step: int = 0,
        current_step: int = 0,
        frequency: int = 100,
        prune_dict: dict = {},
        restore_sparsity: bool = False,
        fix_sparsity: bool = False,
        prune_device: str = "default",
        deploy_device: str = "none",
        group_size: int = 64,
        fixed_mask=None,
        mask=None
    ):
        self._model = model
        self._t = current_step 
        self._initial_sparsity = {}
        self._pretrain_step = pretrain_step
        self._sparse_step = sparse_step
        self._frequency = frequency
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._prune_device = prune_device
        self._deploy_device = deploy_device
        self._fpga_input_group = 4
        # self._asic_input_gloup = 8
        self._group_size = group_size
        self._asic_input_gloup = 512 // group_size
        self._check_parameter()
        self.fixed_mask = fixed_mask
        self.mask = mask 
        self._mask = {}
        self._prepare()

        device = self._model.device
        if self.fixed_mask:
            self._mask = torch.load(self.fixed_mask)
            self._fix_sparsity = True
            for key,v in self._mask.items():
                self._mask[key] = torch.as_tensor(
                    v, dtype=torch.float32, device=device
                ) 

        if self.mask:
            self._mask = torch.load(self.mask)
            for key,v in self._mask.items():
                self._mask[key] = torch.as_tensor(
                    v, dtype=torch.float32, device=device
                )

    def _check_parameter(self):
        assert isinstance(self._pretrain_step, int)
        assert isinstance(self._sparse_step, int)
        assert isinstance(self._frequency, int)
        assert isinstance(self._prune_dict, dict)
        assert isinstance(self._restore_sparsity, bool)
        assert isinstance(self._fix_sparsity, bool)
        assert self._prune_device in ["default", "cpu"]
        assert self._deploy_device in ["none", "fpga", "asic"]

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    if (
                        (self._deploy_device == "fpga")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._fpga_input_group)
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    elif (
                        (self._deploy_device == "asic")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._asic_input_gloup)
                        and ([parameter.shape[2], parameter.shape[3]] == [1, 1])
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(
                            weight == 0,
                            torch.zeros_like(weight),
                            torch.ones_like(weight),
                        )
                        self._initial_sparsity[name] = (
                            1
                            - mask.cpu().numpy().astype(numpy.float32).sum()
                            / weight.cpu().numpy().size
                        )
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)

    def _update_mask(self, name, weight, keep_k):
        if keep_k >= 1:
            reshape_weight = weight.reshape(-1)
            index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
            mask = numpy.zeros(reshape_weight.shape)
            mask[index] = 1
            mask = mask.reshape(weight.shape)
            mask = torch.as_tensor(mask, dtype=weight.dtype, device=weight.device)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_fpga(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k, inc_group):
            reshape_weight = transpose_weight.reshape(
                [
                    -1,
                    transpose_weight.shape[-2]
                    * transpose_weight.shape[-1]
                    // inc_group,
                ]
            )
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = mask.reshape(transpose_weight.shape)
            mask = mask.transpose([0, 2, 1, 3])
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        if keep_k >= 1:
            transpose_weight = weight.permute([0, 2, 1, 3])
            if transpose_weight.shape[-2] % self._fpga_input_group == 0:
                mask = _block_sparsity_balance(
                    transpose_weight, keep_k, self._fpga_input_group
                )
            else:
                temp1 = transpose_weight.shape[-2]
                temp4 = (self._fpga_input_group - 1) * (
                    temp1 // self._fpga_input_group + 1
                )
                keep_k_1 = int(temp4 / temp1 * keep_k)
                keep_k_2 = keep_k - keep_k_1
                transpose_weight_1 = transpose_weight[:, :, :temp4, :]
                transpose_weight_2 = transpose_weight[:, :, temp4:, :]
                mask_1 = _block_sparsity_balance(
                    transpose_weight_1, keep_k_1, self._fpga_input_group - 1
                )
                mask_2 = _block_sparsity_balance(transpose_weight_2, keep_k_2, 1)
                mask = torch.cat([mask_1, mask_2], 1)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_asic_4d(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k):
            reshape_weight = transpose_weight.reshape([-1, transpose_weight.shape[-1]])
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = mask.reshape(transpose_weight.shape)
            mask = mask.transpose([0, 3, 1, 2])
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        def _block_1x1(transpose_weight, keep_k):
            temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
            temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
            for i in range(self._asic_input_gloup):
                locals()["list%s" % i] = []
            for i in range(temp1):
                for j in range(
                    i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                ):
                    locals()["list%s" % (j % self._asic_input_gloup)].append(j)
            for i in range(temp1 * self._asic_input_gloup, transpose_weight.shape[-1]):
                locals()["list%s" % (i % self._asic_input_gloup)].append(i)
            temp3 = []
            for i in range(self._asic_input_gloup):
                temp3.append(
                    int(
                        len(locals()["list%s" % i])
                        / transpose_weight.shape[-1]
                        * keep_k
                    )
                )
            group_mask = numpy.ones(transpose_weight.shape).transpose([0, 3, 1, 2])
            for i in range(self._asic_input_gloup):
                temp4 = torch.cat(
                    [
                        transpose_weight[:, :, :, one : one + 1]
                        for one in locals()["list%s" % i]
                    ],
                    3,
                )
                mask = _block_sparsity_balance(temp4, temp3[i])
                for one, two in enumerate(locals()["list%s" % i]):
                    group_mask[:, two : two + 1, :, :] = (
                        mask[:, one : one + 1, :, :].cpu().numpy()
                    )
            group_mask = torch.as_tensor(
                group_mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return group_mask

        if keep_k >= 1:
            transpose_weight = weight.permute([0, 2, 3, 1])
            if transpose_weight.shape[1] == 1 and transpose_weight.shape[2] == 1:
                group_size = 512
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                mask = numpy.ones(weight.shape)
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        mask_1 = _block_1x1(transpose_weight_1, keep_k_1 // temp1)
                        mask[
                            :, i * group_size : (i + 1) * group_size, :, :
                        ] = mask_1.cpu().numpy()
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                        mask_2 = _block_1x1(transpose_weight_2, keep_k_2)
                        mask[:, temp1 * group_size :, :, :] = mask_2.cpu().numpy()
                    else:
                        pass
                mask = torch.as_tensor(
                    mask, dtype=transpose_weight.dtype, device=transpose_weight.device
                )
            else:
                group_size = self._group_size
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                mask = numpy.ones(weight.shape)
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        mask_1 = _block_sparsity_balance(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                        mask[
                            :, i * group_size : (i + 1) * group_size, :, :
                        ] = mask_1.cpu().numpy()
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    mask_2 = _block_sparsity_balance(transpose_weight_2, keep_k_2)
                    mask[:, temp1 * group_size :, :, :] = mask_2.cpu().numpy()
                mask = torch.as_tensor(
                    mask, dtype=transpose_weight.dtype, device=transpose_weight.device
                )
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_asic_2d(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k):
            reshape_weight = transpose_weight
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        def _block_1x1(transpose_weight, keep_k):
            temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
            temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
            for i in range(self._asic_input_gloup):
                locals()["list%s" % i] = []
            for i in range(temp1):
                for j in range(
                    i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                ):
                    locals()["list%s" % (j % self._asic_input_gloup)].append(j)
            for i in range(temp1 * self._asic_input_gloup, transpose_weight.shape[-1]):
                locals()["list%s" % (i % self._asic_input_gloup)].append(i)
            temp3 = []
            for i in range(self._asic_input_gloup):
                temp3.append(
                    int(
                        len(locals()["list%s" % i])
                        / transpose_weight.shape[-1]
                        * keep_k
                    )
                )
            group_mask = numpy.ones(transpose_weight.shape)
            for i in range(self._asic_input_gloup):
                temp4 = torch.cat(
                    [
                        transpose_weight[:, one : one + 1]
                        for one in locals()["list%s" % i]
                    ],
                    1,
                )
                mask = _block_sparsity_balance(temp4, temp3[i])
                for one, two in enumerate(locals()["list%s" % i]):
                    group_mask[:, two : two + 1] = mask[:, one : one + 1].cpu().numpy()
            group_mask = torch.as_tensor(
                group_mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return group_mask

        if keep_k >= 1:
            transpose_weight = weight
            group_size = 512
            temp1 = transpose_weight.shape[-1] // group_size
            temp2 = transpose_weight.shape[-1] % group_size
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            mask = numpy.ones(weight.shape)
            if temp1 > 0:
                for i in range(temp1):
                    transpose_weight_1 = transpose_weight[
                        :, i * group_size : (i + 1) * group_size
                    ]
                    mask_1 = _block_1x1(transpose_weight_1, keep_k_1 // temp1)
                    mask[
                        :, i * group_size : (i + 1) * group_size
                    ] = mask_1.cpu().numpy()
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, temp1 * group_size :]
                if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                    mask_2 = _block_1x1(transpose_weight_2, keep_k_2)
                    mask[:, temp1 * group_size :] = mask_2.cpu().numpy()
                else:
                    pass
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = (
            self._pretrain_step < self._t <= self._pretrain_step + self._sparse_step
        )
        condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        return condition1 and condition2 and condition3

    def _get_weight(self, parameter):
        if self._prune_device == "default":
            weight = parameter.data
        elif self._prune_device == "cpu":
            weight = parameter.data.to(device=torch.device("cpu"))
        return weight

    def prune(self):
        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._update_mask_conditions():
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (
                            self._t - self._pretrain_step
                        ) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = (
                            target_sparsity
                            + (self._initial_sparsity[name] - target_sparsity)
                            * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        )
                        keep_k = int(
                            weight.cpu().numpy().size * (1.0 - current_sparsity)
                        )
                        if self._deploy_device == "none":
                            self._update_mask(name, weight, keep_k)
                        elif self._deploy_device == "fpga":
                            if len(weight.shape) == 4:
                                self._update_mask_fpga(name, weight, keep_k)
                            else:
                                self._update_mask(name, weight, keep_k)
                        elif self._deploy_device == "asic":
                            if len(weight.shape) == 4:
                                self._update_mask_asic_4d(name, weight, keep_k)
                            elif len(weight.shape) == 2:
                                self._update_mask_asic_2d(name, weight, keep_k)
                            else:
                                self._update_mask(name, weight, keep_k)

                    parameter.mul_(self._mask[name])

    def sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}

        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + numpy.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - numpy.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate

    def check(self):
        def _check_weight(weight, keep_k):
            qualify = numpy.flatnonzero(weight).size <= keep_k
            return qualify

        def _check_weight_fpga(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k, inc_group):
                reshape_weight = transpose_weight.reshape(
                    [
                        -1,
                        transpose_weight.shape[-2]
                        * transpose_weight.shape[-1]
                        // inc_group,
                    ]
                )
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            transpose_weight = weight.transpose([0, 2, 1, 3])
            if transpose_weight.shape[-2] % self._fpga_input_group == 0:
                qualify = _check_block_sparsity_balance(
                    transpose_weight, keep_k, self._fpga_input_group
                )
            else:
                temp1 = transpose_weight.shape[-2]
                temp4 = (self._fpga_input_group - 1) * (
                    temp1 // self._fpga_input_group + 1
                )
                keep_k_1 = int(temp4 / temp1 * keep_k)
                keep_k_2 = keep_k - keep_k_1
                transpose_weight_1 = transpose_weight[:, :, :temp4, :]
                transpose_weight_2 = transpose_weight[:, :, temp4:, :]
                qualify_1 = _check_block_sparsity_balance(
                    transpose_weight_1, keep_k_1, self._fpga_input_group - 1
                )
                qualify_2 = _check_block_sparsity_balance(
                    transpose_weight_2, keep_k_2, 1
                )
                qualify = all([qualify_1, qualify_2])
            return qualify

        def _check_weight_asic_4d(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k):
                reshape_weight = transpose_weight.reshape(
                    [-1, transpose_weight.shape[-1]]
                )
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            def _check_block_1x1(transpose_weight, keep_k):
                temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
                temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
                for i in range(self._asic_input_gloup):
                    locals()["list%s" % i] = []
                for i in range(temp1):
                    for j in range(
                        i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                    ):
                        locals()["list%s" % (j % self._asic_input_gloup)].append(j)
                for i in range(
                    temp1 * self._asic_input_gloup, transpose_weight.shape[-1]
                ):
                    locals()["list%s" % (i % self._asic_input_gloup)].append(i)
                temp3 = []
                for i in range(self._asic_input_gloup):
                    temp3.append(
                        int(
                            len(locals()["list%s" % i])
                            / transpose_weight.shape[-1]
                            * keep_k
                        )
                    )
                qualify_list = []
                for i in range(self._asic_input_gloup):
                    temp4 = numpy.concatenate(
                        [
                            transpose_weight[:, :, :, one : one + 1]
                            for one in locals()["list%s" % i]
                        ],
                        3,
                    )
                    qualify_list.append(_check_block_sparsity_balance(temp4, temp3[i]))
                return all(qualify_list)

            transpose_weight = weight.transpose([0, 2, 3, 1])
            if transpose_weight.shape[1] == 1 and transpose_weight.shape[2] == 1:
                group_size = 512
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        qualify_1 = _check_block_1x1(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                        qualify_2 = _check_block_1x1(transpose_weight_2, keep_k_2)
                    else:
                        pass
                qualify_list = []
                try:
                    qualify_list.append(qualify_1)
                except:
                    pass
                try:
                    qualify_list.append(qualify_2)
                except:
                    pass
                qualify = all(qualify_list)
            else:
                group_size = self._group_size
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        qualify_1 = _check_block_sparsity_balance(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    qualify_2 = _check_block_sparsity_balance(
                        transpose_weight_2, keep_k_2
                    )
                qualify_list = []
                try:
                    qualify_list.append(qualify_1)
                except:
                    pass
                try:
                    qualify_list.append(qualify_2)
                except:
                    pass
                qualify = all(qualify_list)
            return qualify

        def _check_weight_asic_2d(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k):
                reshape_weight = transpose_weight
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            def _check_block_1x1(transpose_weight, keep_k):
                temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
                temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
                for i in range(self._asic_input_gloup):
                    locals()["list%s" % i] = []
                for i in range(temp1):
                    for j in range(
                        i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                    ):
                        locals()["list%s" % (j % self._asic_input_gloup)].append(j)
                for i in range(
                    temp1 * self._asic_input_gloup, transpose_weight.shape[-1]
                ):
                    locals()["list%s" % (i % self._asic_input_gloup)].append(i)
                temp3 = []
                for i in range(self._asic_input_gloup):
                    temp3.append(
                        int(
                            len(locals()["list%s" % i])
                            / transpose_weight.shape[-1]
                            * keep_k
                        )
                    )
                qualify_list = []
                for i in range(self._asic_input_gloup):
                    temp4 = numpy.concatenate(
                        [
                            transpose_weight[:, one : one + 1]
                            for one in locals()["list%s" % i]
                        ],
                        1,
                    )
                    qualify_list.append(_check_block_sparsity_balance(temp4, temp3[i]))
                return all(qualify_list)

            transpose_weight = weight
            group_size = 512
            temp1 = transpose_weight.shape[-1] // group_size
            temp2 = transpose_weight.shape[-1] % group_size
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            if temp1 > 0:
                for i in range(temp1):
                    transpose_weight_1 = transpose_weight[
                        :, i * group_size : (i + 1) * group_size
                    ]
                    qualify_1 = _check_block_1x1(transpose_weight_1, keep_k_1 // temp1)
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, temp1 * group_size :]
                if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                    qualify_2 = _check_block_1x1(transpose_weight_2, keep_k_2)
                else:
                    pass
            qualify_list = []
            try:
                qualify_list.append(qualify_1)
            except:
                pass
            try:
                qualify_list.append(qualify_2)
            except:
                pass
            qualify = all(qualify_list)
            return qualify

        with torch.no_grad():
            layer_sparse_qualify = {}
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = parameter.data.cpu().numpy()
                    target_sparsity = self._prune_dict[name]
                    keep_k = int(weight.size * (1.0 - target_sparsity))
                    if self._deploy_device == "none":
                        qualify = _check_weight(weight, keep_k)
                    elif self._deploy_device == "fpga":
                        if len(weight.shape) == 4:
                            qualify = _check_weight_fpga(weight, keep_k)
                        else:
                            qualify = _check_weight(weight, keep_k)
                    elif self._deploy_device == "asic":
                        if len(weight.shape) == 4:
                            qualify = _check_weight_asic_4d(weight, keep_k)
                        elif len(weight.shape) == 2:
                            qualify = _check_weight_asic_2d(weight, keep_k)
                        else:
                            qualify = _check_weight(weight, keep_k)
                    layer_sparse_qualify[name] = qualify
        total_sparse_qualify = all(one for one in layer_sparse_qualify.values())
        return layer_sparse_qualify, total_sparse_qualify


if __name__ == "__main__":
    import os
    import torchvision

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv0 = torch.nn.Conv2d(1, 599, 3, 2)
            self.bn0 = torch.nn.BatchNorm2d(599)
            self.conv1 = torch.nn.Conv2d(599, 100, 1, 1)
            self.bn1 = torch.nn.BatchNorm2d(100)
            self.conv2 = torch.nn.Conv2d(100, 127, 3, 2)
            self.bn2 = torch.nn.BatchNorm2d(127)
            self.conv3 = torch.nn.Conv2d(127, 200, 3, 2)
            self.bn3 = torch.nn.BatchNorm2d(200)
            self.linear1 = torch.nn.Linear(800, 10)

        def forward(self, x):
            x = self.conv0(x)
            x = self.bn0(x)
            x = torch.nn.functional.relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.nn.functional.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.nn.functional.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.nn.functional.relu(x)
            x = torch.flatten(x, 1)
            x = self.linear1(x)
            output = torch.nn.functional.softmax(x, dim=1)
            return output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(
        "~/.pytorch/datasets", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        "~/.pytorch/datasets", train=False, transform=transform
    )

    batch_size = 256
    epoch = 10
    step = train_data.data.shape[0] // batch_size
    lr = 0.1 / 256 * batch_size

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    model = Net().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step * epoch)

    ########################################################
    prune_dict = {}
    for k, v in model.named_parameters():
        if len(v.shape) < 2:
            continue
        if k == "conv1.weight":
            prune_dict[k] = 0.95
        else:
            prune_dict[k] = 0.99
    prune = Prune(model, step * 0, step * 8, 10, prune_dict, deploy_device="asic")
    ########################################################

    for idx in range(epoch):
        model.train()
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            #############
            prune.prune()
            #############

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += torch.nn.functional.cross_entropy(
                    output, target, reduction="sum"
                ).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        # print(torch.cuda.memory_allocated(torch.device('cuda')))
        #######################################################
        layer_sparse_rate, total_sparse_rate = prune.sparsity()
        #######################################################
        print(
            "Epoch %d: Accuracy=%f; weight sparsity=%s"
            % (idx, test_acc, total_sparse_rate)
        )

    ##########################################################
    layer_sparse_qualify, total_sparse_qualify = prune.check()
    ##########################################################
    print("deploy qualified: %s" % total_sparse_qualify)
    torch.save(model.state_dict(), "pytorch_mnist")
