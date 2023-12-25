import os
import sys
from functools import partial
from collections import defaultdict
import acl
from tbe import tik
import te.platform as tbe_platform
from tbe.common.platform import get_soc_spec

sys.path.append(os.path.dirname(__file__))

from constants import FP16
from constants import GM
from constants import L1
from constants import UB
from tik_ops_utils import TikOpsUtils


class FlashAttentionFwd:
    """The implementation of flash attention forward
    This function contains the flash attention forward implementation used in flash attention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`
    """

    def __init__(self, q, k, v, kernel_name, disable_debug=True):
        self.tik_instance = tik.Tik(disable_debug=disable_debug)
        self.core_num = get_soc_spec(tbe_platform.CORE_NUM)
        self.soc_version = acl.get_soc_name()
        self.kernel_name = kernel_name
        self.cont_data_mv_1_bust = partial(self.tik_instance.data_move, sid=0, nburst=1,
                                           src_stride=0,
                                           dst_stride=0)
        self.tik_ops_utils = TikOpsUtils(self.tik_instance)
        self.max_repeat_times = 255
        self.max_head_num = 64
        self.max_batch_head = 4096
        self.max_seqlen = 8192
        self.head_dim_list = [40, 64, 80, 160]  # TODO:need test to confirm head_dim range
        self.min_batch = 1
        self.max_batch = 64
        self.valid = True

        if q is None or k is None or v is None:
            self.valid = False
            return

        # NZ
        self.q_shape = q["shape"]
        self.k_shape = k["shape"]
        self.v_shape = v["shape"]
        # ND
        self.q_ori_shape = q["ori_shape"]
        self.k_ori_shape = k["ori_shape"]
        self.v_ori_shape = v["ori_shape"]
        self.check_shape()
        if not self.valid:
            return
        if len(self.q_ori_shape) == 4:
            batch_size, head_num, Nq, actual_d = self.q_ori_shape
            N = self.k_ori_shape[2]
            self.bs = batch_size
            self.heads = head_num
            self.B = batch_size * head_num
        else:
            batch_size, Nq, actual_d = self.q_ori_shape
            N = self.k_ori_shape[1]
            self.bs = batch_size
            self.heads = 1
            self.B = batch_size

        self.actual_d = actual_d
        self.actual_Nq = Nq
        self.actual_N = N
        self.N0 = 16
        self.M0 = 16
        self.d = (actual_d + self.N0 - 1) // self.N0 * self.N0
        self.Nq = (Nq + self.M0 - 1) // self.M0 * self.M0
        self.N = (N + self.M0 - 1) // self.M0 * self.M0
        self.N1 = self.d // self.N0

        self.check_params()
        if not self.valid:
            return

        self.O_shape = self.q_shape
        self.l_shape = (self.B, self.Nq)
        self.m_shape = (self.B, self.Nq)

    def check_params(self):
        if not self.valid:
            return
        if self.heads <= 0 or self.heads > self.max_head_num:
            self.valid = False
            return
        if self.bs <= 0:
            self.valid = False
            return
        if len(self.q_ori_shape) == 4 and (self.bs < self.min_batch or self.bs > self.max_batch) :
            self.valid = False
            return
        if len(self.q_ori_shape) == 3 and self.bs > self.max_batch_head:
            self.valid = False
            return
        if self.actual_d not in self.head_dim_list:
            self.valid = False
            return
        if self.actual_Nq <= 0 or self.actual_Nq > self.max_seqlen:
            self.valid = False
            return
        if self.actual_N <= 0 or self.actual_N > self.max_seqlen:
            self.valid = False
            return
        if self.Nq <= 0 or self.Nq > self.max_seqlen:
            self.valid = False
            return
        if self.N <= 0 or self.N > self.max_seqlen:
            self.valid = False
            return
        return

    def check_shape(self):
        if not self.valid:
            return
        if self.q_shape is None or self.k_shape is None or self.v_shape is None:
            self.valid = False
            return
        if self.q_ori_shape is None or self.k_ori_shape is None or self.v_ori_shape is None:
            self.valid = False
            return
        # ori shape should be 3 or 4
        if len(self.q_ori_shape) != 3 and len(self.q_ori_shape) != 4:
            self.valid = False
            return
        if len(self.k_ori_shape) != len(self.q_ori_shape):
            self.valid = False
            return
        if len(self.v_ori_shape) != len(self.q_ori_shape):
            self.valid = False
            return
        return

    def init_gm(self):
        if not self.valid:
            return
        # input gm
        self.Q_gm = self.tik_instance.Tensor(FP16, self.q_shape, name="Q_gm", scope=GM)
        self.K_gm = self.tik_instance.Tensor(FP16, self.k_shape, name="K_gm", scope=GM)
        self.V_gm = self.tik_instance.Tensor(FP16, self.v_shape, name="V_gm", scope=GM)

        # workspace gm
        self.l_gm = self.tik_instance.Tensor(FP16, self.l_shape, name="l_gm", scope=GM, is_workspace=True)
        self.m_gm = self.tik_instance.Tensor(FP16, self.m_shape, name="m_gm", scope=GM, is_workspace=True)
        # output gm
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM)

    def tiling(self):
        """Tiling for FlashAttention"""
        if not self.valid:
            return
        self.Br = min(64, self.actual_Nq)
        self.Bc = min(512, self.actual_N)

        if self.d >= 192:
            self.Bc = min(self.Bc, 128)
        elif self.d >= 96:
            self.Bc = min(self.Bc, 256)

        if self.Bc <= 64 and self.d <= 64:
            self.Br = min(512, self.Nq)
        elif self.Bc <= 160 and self.d <= 160:
            self.Br = min(256, self.Nq)
        elif self.Bc <= 256 and self.d <= 256:
            self.Br = min(128, self.Nq)

        self.Tr = self.actual_Nq // self.Br
        self.Tc = self.actual_N // self.Bc

        if self.actual_Nq % self.Br != 0:
            self.last_Br = self.actual_Nq - self.Tr * self.Br
            self.Tr += 1
        else:
            self.last_Br = self.Br
        if self.actual_N % self.Bc != 0:
            self.last_Bc = self.actual_N - self.Tc * self.Bc
            self.Tc += 1
        else:
            self.last_Bc = self.Bc

    def prepare_global_ones(self):
        """Prepare ones tensor for cube impl row_sum"""
        if not self.valid:
            return
        Bc_aligned = self.tik_ops_utils.up_align_to_K0(self.Bc)
        last_Bc_aligned = self.tik_ops_utils.up_align_to_K0(self.last_Bc)
        self.ones_l1 = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_l1", scope=L1)
        self.last_ones_l1 = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_l1", scope=L1)
        with self.tik_instance.new_stmt_scope():
            ones_ub = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_ub", scope=UB)
            last_ones_ub = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_ub", scope=UB)
            self.tik_ops_utils.vec_duplicate(ones_ub, 1.0, Bc_aligned * 16)
            self.tik_ops_utils.vec_duplicate(last_ones_ub, 1.0, last_Bc_aligned * 16)
            self.cont_data_mv_1_bust(dst=self.ones_l1, src=ones_ub, burst=Bc_aligned)
            self.cont_data_mv_1_bust(dst=self.last_ones_l1, src=last_ones_ub, burst=last_Bc_aligned)

    def get_total_block_num(self):
        """Calc total block num"""
        if not self.valid:
            return
        block_num = 0
        for _ in range(self.B):
            for tr_idx in range(self.Tr):
                block_num += self.Tc
        return block_num

    def update_core_task_map(self,
                             core_b_map,
                             core_b_tr_map,
                             core_idx,
                             b_start,
                             b_end,
                             tr_start,
                             tr_end):
        """Update each core task map"""
        if not self.valid:
            return
        core_b_map[core_idx][0] = min(core_b_map[core_idx][0], b_start)
        if tr_end == 0:  # 跨head，但跨过的head不会被当前的core处理
            core_b_map[core_idx][1] = max(core_b_map[core_idx][1], b_end - 1)
        else:
            core_b_map[core_idx][1] = max(core_b_map[core_idx][1], b_end)
        for b_idx in range(b_start, b_end + 1):
            if b_idx == b_end and tr_end == 0:  # 跨head，但跨过的head不会被当前的core处理
                break
            elif b_idx == b_start and b_idx == b_end:  # 没跨head
                core_b_tr_map[core_idx][b_idx] = (tr_start, tr_end)
            elif b_idx == b_start:  # 跨head，第一个head
                core_b_tr_map[core_idx][b_idx] = (tr_start, self.Tr)
            elif b_idx == b_end:  # 跨head，最后一个head
                core_b_tr_map[core_idx][b_idx] = (0, tr_end)
            else:  # 跨head，中间的head
                core_b_tr_map[core_idx][b_idx] = (0, self.Tr)

    def convert_py_dict_to_tik_tensor(self, core_b_map, core_b_tr_map):
        """Convert python dict to tik tensor"""
        if not self.valid:
            return
        core_idx_to_batch_info = self.tik_instance.Tensor(
            "int32", (self.core_num, 2), name="core_idx_to_batch_info", scope=UB
        )
        core_idx_to_tr_info = self.tik_instance.Tensor(
            "int32", (self.core_num, self.B, 2), name="core_idx_to_tr_info", scope=UB
        )
        for core_idx in core_b_map.keys():
            batch_start, batch_end = core_b_map[core_idx]
            core_idx_to_batch_info[core_idx, 0] = batch_start
            core_idx_to_batch_info[core_idx, 1] = batch_end - batch_start + 1
            for batch_idx in core_b_tr_map[core_idx].keys():
                tr_start, tr_end = core_b_tr_map[core_idx][batch_idx]
                core_idx_to_tr_info[core_idx, batch_idx, 0] = tr_start
                core_idx_to_tr_info[core_idx, batch_idx, 1] = tr_end

        return core_idx_to_batch_info, core_idx_to_tr_info

    def get_core_task_info(self):
        """Get batch start, batch number, outer loop start and end of each NPU core"""
        if not self.valid:
            return
        if self.core_num > self.B * self.Tr:
            self.core_num = self.B * self.Tr

        total_blk_num = self.get_total_block_num()
        b_start = 0
        tr_start = 0
        remain_blk_num = total_blk_num
        core_b_map = defaultdict(lambda: [100000, -1])
        core_b_tr_map = defaultdict(lambda: defaultdict(list))
        for core_idx in range(self.core_num):
            cur_core_blk_num = 0
            cur_each_core_blk_num = remain_blk_num // (self.core_num - core_idx)
            cur_core_finished = False
            b_end = b_start
            tr_end = tr_start
            while b_end < self.B:
                while tr_end < self.Tr:
                    cur_tr_blk_num = self.Tc
                    if abs(cur_core_blk_num - cur_each_core_blk_num) <= \
                            (cur_core_blk_num + cur_tr_blk_num - cur_each_core_blk_num):
                        self.update_core_task_map(core_b_map, core_b_tr_map, core_idx, b_start, b_end, tr_start, tr_end)
                        remain_blk_num -= cur_core_blk_num
                        cur_core_finished = True
                        break
                    else:
                        cur_core_blk_num += cur_tr_blk_num
                        tr_end += 1
                        if tr_end == self.Tr:
                            tr_end = 0
                            b_end += 1
                if cur_core_finished:
                    b_start = b_end
                    tr_start = tr_end
                    break
        core_idx_to_batch_info, core_idx_to_tr_info = self.convert_py_dict_to_tik_tensor(core_b_map, core_b_tr_map)
        return core_idx_to_batch_info, core_idx_to_tr_info

    def get_gm_offset(self, batch_start, batch_idx, h, w, block_h, block_idx):
        """Calc gm offset"""
        if not self.valid:
            return
        gm_offset = (batch_start + batch_idx) * h * w + block_idx * block_h * self.N0
        return gm_offset

    def softmax_compute(self, Sij_ub, mij_ub, lij_ub, m, n):
        """Refer to Algorithm 2 line12
        Calculate softmax.
        :param Sij_ub: with shape of (N1, M, N0), 使用Sij空间返回Pij 提高UB利用率
        :param mij_ub:
        :param lij_ub:
        :param m:
        :param n:
        :return:
        """
        if not self.valid:
            return
        m_aligned = self.tik_ops_utils.up_align_to_K0(m)
        n_aligned = self.tik_ops_utils.up_align_to_K0(n)
        n0 = 16
        n1 = n_aligned // 16
        # calc rowmax of Sij
        loop_times = n1
        if n != n_aligned:
            loop_times = n1 - 1
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mn0_block_max = self.tik_instance.Tensor(FP16, (m_aligned, n0), name="mn0_block_max", scope=UB)
            self.cont_data_mv_1_bust(dst=mn0_block_max, src=Sij_ub, burst=m_aligned) # 用第一个(m, n0)始化，少一次duplicate和max
            # with self.tik_instance.for_range(1, loop_times) as idx:
            for idx in range(1, loop_times):
                offset = idx * m_aligned * n0
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_max,
                                                        mn0_block_max,
                                                        mn0_block_max,
                                                        Sij_ub[offset],
                                                        m_aligned * n0)
            # 为rowmax过滤尾块后面的无效数据
            if n != n_aligned:
                offset1 = (n1 - 1) * m_aligned * n0
                mask = n0 - (n_aligned - n)
                if m_aligned > self.max_repeat_times:
                    max_repeat_times_loops, remain_repeat_times = divmod(m_aligned, self.max_repeat_times)
                    for idx in range(max_repeat_times_loops):
                        offset2 = idx * self.max_repeat_times * n0
                        self.tik_instance.vec_max(mask, mn0_block_max[offset2], mn0_block_max[offset2],
                                                  Sij_ub[offset1+offset2], self.max_repeat_times, 1, 1, 1)
                    if remain_repeat_times > 0:
                        offset2 = max_repeat_times_loops * self.max_repeat_times * n0
                        self.tik_instance.vec_max(mask, mn0_block_max[offset2], mn0_block_max[offset2],
                                                  Sij_ub[offset1 + offset2], remain_repeat_times, 1, 1, 1)
                else:
                    self.tik_instance.vec_max(mask, mn0_block_max, mn0_block_max, Sij_ub[offset1], m_aligned, 1, 1, 1)
            self.tik_ops_utils.vec_reduce(self.tik_instance.vcgmax, mij_ub, mn0_block_max, m_aligned * n0)
        # Sij - mij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_mij_ub = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(mij_ub)
            for idx in range(n1):
                offset = idx * m_aligned * n0
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_sub,
                                                        Sij_ub[offset],
                                                        Sij_ub[offset],
                                                        broadcast_mij_ub,
                                                        m_aligned * n0)

        if self.soc_version == "Ascend910A":
            # exp
            self.tik_ops_utils.vec_ele_wise(self.tik_instance.vec_exp,
                                            Sij_ub, Sij_ub, m_aligned * n_aligned)
        elif self.soc_version == "Ascend310P3":
            # fast exp
            self.tik_ops_utils.vec_and_scalar_ele_wise(self.tik_instance.vec_muls,
                                                       Sij_ub, Sij_ub, 1 / 256.0, m_aligned * n_aligned)
            self.tik_ops_utils.vec_and_scalar_ele_wise(self.tik_instance.vec_adds,
                                                       Sij_ub, Sij_ub, 1.0, m_aligned * n_aligned)
            for _ in range(8):
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                        Sij_ub,
                                                        Sij_ub,
                                                        Sij_ub,
                                                        m_aligned * n_aligned)

        # cube impl rowsum
        Sij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (n_aligned // 16, m_aligned, 16),
                                                    name="Sij_l1_K1MK0_ed", scope=L1)
        self.cont_data_mv_1_bust(dst=Sij_l1_K1MK0_ed, src=Sij_ub, burst=m_aligned * n_aligned // 16)
        if n == self.last_Bc:
            Sij_row_sum_ub = self.tik_ops_utils.row_sum_cube_impl(Sij_l1_K1MK0_ed, self.last_ones_l1, lij_ub, m, n)
        else:
            Sij_row_sum_ub = self.tik_ops_utils.row_sum_cube_impl(Sij_l1_K1MK0_ed, self.ones_l1, lij_ub, m, n)

        return Sij_l1_K1MK0_ed, mij_ub, Sij_row_sum_ub

    def update_m_l(self, mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len):
        """Refer to Algorithm 2 line13
        mi_new = max(mi, mij), li_new = exp(mi-mi_new)*li + exp(mij - mi_new) * lij
        """
        if not self.valid:
            return
        vec_len_aligned = self.tik_ops_utils.up_align_to_K0(vec_len)
        mi_new_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), name="mi_new_ub", scope=UB)
        li_new_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), name="li_new_ub", scope=UB)
        # 1 calculate mi_new = max(mi, mij)
        self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_max,
                                                mi_new_ub,
                                                mi_old_ub,
                                                mij_ub,
                                                vec_len_aligned)

        # 2 calculate li_new = exp(mi-mi_new)*li + exp(mij-mi_new)*lij
        # 2.1 相减，求指数
        self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_sub,
                                                mi_old_ub,
                                                mi_old_ub,
                                                mi_new_ub,
                                                vec_len_aligned) # mi-mi_new
        self.tik_ops_utils.vec_ele_wise(self.tik_instance.vec_exp,
                                        mi_old_ub,
                                        mi_old_ub,
                                        vec_len_aligned)  # exp(mi-mi_new)

        # 2.2 相减，求指数
        self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_sub,
                                                mij_ub,
                                                mij_ub,
                                                mi_new_ub,
                                                vec_len_aligned) # mij-mi_new
        self.tik_ops_utils.vec_ele_wise(self.tik_instance.vec_exp,
                                        mij_ub,
                                        mij_ub,
                                        vec_len_aligned) # exp(mij-mi_new)

        # 2.3 相乘，相加 exp(m_ij_ub-mi_new)*li + exp(m_ij_ub-mi_new) * lij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mul_li_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), scope=UB, name="mul_li_ub")
            mul_lij_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), scope=UB, name="mul_lij_ub")
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                    mul_li_ub,
                                                    mi_old_ub,
                                                    li_old_ub,
                                                    vec_len_aligned)
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                    mul_lij_ub,
                                                    mij_ub,
                                                    lij_ub,
                                                    vec_len_aligned)
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_add,
                                                    li_new_ub,
                                                    mul_li_ub,
                                                    mul_lij_ub,
                                                    vec_len_aligned)
        return mi_new_ub, li_new_ub

    def update_o_m_l(self,
                     Pij_l1_K1MK0_ed,
                     Vj_l1_K1NK0_ed,
                     mij_ub,
                     lij_ub,
                     batch_start,
                     batch_idx,
                     kv_blk_idx,
                     kv_blk_h,
                     q_blk_idx,
                     q_blk_h,
                     o_gm_offset):
        """Refer to Algorithm 2 line13 and line15 in FlashAttention
        load o m l from gm and update them in ub, then write them back to gm
        :param Pij_l1_K1MK0_ed: input tensor with shape of (Bc // 16, Br, 16)
        :param Vj_l1_K1NK0_ed: input tensor with shape of (d // 16, Bc, 16)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param kv_blk_h:
        :param q_blk_idx:
        :param q_blk_h:
        :param o_gm_offset:
        :return: None
        """
        if not self.valid:
            return
        vec_gm_offset = (batch_start + batch_idx) * self.Nq + q_blk_idx * self.Br
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_h)
        Pij_Vj_matmul_res_ub = self.tik_ops_utils.matmul_compute(Pij_l1_K1MK0_ed, Vj_l1_K1NK0_ed, q_blk_h,
                                                                 kv_blk_h, self.actual_d,
                                                                 N1MN0_to_MN=False)  # Pij*Vj
        n1, m, n0 = Pij_Vj_matmul_res_ub.shape
        # 第一次外循环时，不用走更新逻辑， 减少的无用计算
        # calculate mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, q_blk_h_aligned)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, q_blk_h_aligned)
            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(lij_ub, q_blk_h_aligned)
            # TODO：NZ版double buffer
            broadcast_li_new_rec_ub = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(li_new_rec_ub)
            # with self.tik_instance.for_range(0, n1) as idx:
            for idx in range(0, n1):
                offset = idx * m * n0
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                        Pij_Vj_matmul_res_ub[offset],
                                                        Pij_Vj_matmul_res_ub[offset],
                                                        broadcast_li_new_rec_ub,
                                                        m * n0)
            self.tik_instance.data_move(dst=self.O_gm[o_gm_offset], src=Pij_Vj_matmul_res_ub, sid=0,
                                        nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                        src_stride=0, dst_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16)


        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), name="li_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset, q_blk_h_aligned)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset, q_blk_h_aligned)

            # 更新 l, m
            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, q_blk_h_aligned)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, q_blk_h_aligned)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, q_blk_h_aligned)

            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub
            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(li_new_ub, q_blk_h_aligned)
            # TODO：double buffer
            # scale1 <- li * exp(mi - mi_new) / li_new
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                    li_ub,
                                                    li_ub,
                                                    exp_mi_sub_mi_new,
                                                    q_blk_h_aligned)
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                    li_ub,
                                                    li_ub,
                                                    li_new_rec_ub,
                                                    q_blk_h_aligned)
            scale1 = li_ub
            # scale2 <- exp(mij - mi_new) / li_new
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                    exp_mij_sub_mi_new,
                                                    exp_mij_sub_mi_new,
                                                    li_new_rec_ub,
                                                    q_blk_h_aligned)
            scale2 = exp_mij_sub_mi_new
            Oi_ub = self.tik_instance.Tensor(FP16, (n1, m, n0), name="Oi_ub", scope=UB)
            self.tik_instance.data_move(dst=Oi_ub, src=self.O_gm[o_gm_offset],
                                        sid=0, nburst=self.N1, burst=m * self.N0 // 16,
                                        src_stride=(self.Nq - m) * self.N0 // 16, dst_stride=0)
            broadcast_scale1 = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(scale1)
            # with self.tik_instance.for_range(0, n1) as idx:
            for idx in range(0, n1):
                offset = idx * m * n0
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                        Oi_ub[offset],
                                                        Oi_ub[offset],
                                                        broadcast_scale1,
                                                        m * n0)
            # vec: scale2 * Pij_Vj
            broadcast_scale2 = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(scale2)
            # with self.tik_instance.for_range(0, n1) as idx:
            for idx in range(0, n1):
                offset = idx * m * n0
                self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_mul,
                                                        Pij_Vj_matmul_res_ub[offset],
                                                        Pij_Vj_matmul_res_ub[offset],
                                                        broadcast_scale2,
                                                        m * n0)
            self.tik_ops_utils.vec_and_vec_ele_wise(self.tik_instance.vec_add,
                                                    Oi_ub,
                                                    Oi_ub,
                                                    Pij_Vj_matmul_res_ub,
                                                    m * n1 * n0)
            self.tik_instance.data_move(dst=self.O_gm[o_gm_offset], src=Oi_ub, sid=0,
                                        nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                        src_stride=0, dst_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16)

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_h,
                                 core_idx_to_tr_info, core_idx):
        """The forward computation in each outer loop"""
        if not self.valid:
            return
        kv_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_h)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc,
                                          kv_blk_idx)
        # load Kj (kv_blk_idx_th block of K_gm)
        KjT_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.N1, kv_blk_h_aligned, self.N0),
                                                   name="KjT_l1_K1MK0_ed", scope=L1)
        self.tik_instance.data_move(dst=KjT_l1_K1MK0_ed, src=self.K_gm[kv_gm_offset],
                                    sid=0, nburst=self.N1, burst=kv_blk_h_aligned * self.N0 // 16,
                                    src_stride=(self.N - kv_blk_h_aligned) * self.N0 // 16, dst_stride=0)

        # load Vj (kv_blk_idx_th block of V_gm), then reorder for Pij*Vj
        Vj_l1 = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned, self.d), name="Vj_l1", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # TODO: 外部转置或V的快速转换方案探索
            Vj_ub = self.tik_instance.Tensor(FP16, (self.N1, kv_blk_h_aligned, self.N0),
                                             name="Vj_ub", scope=UB)
            self.tik_instance.data_move(dst=Vj_ub, src=self.V_gm[kv_gm_offset],
                                        sid=0, nburst=self.N1, burst=kv_blk_h_aligned * self.N0 // 16,
                                        src_stride=(self.N - kv_blk_h_aligned) * self.N0 // 16, dst_stride=0)
            # (N1, K, N0) -> (K, N)
            Vj_ub = self.tik_ops_utils.N1MN0_TO_MN(Vj_ub)
            # (K, N) -> (K1, N, K0)
            Vj_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Vj_ub, workspace_tensor=Vj_l1)

        tr_start_s = self.tik_instance.Scalar("int32",
                                              init_value=core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0],
                                              name="tr_start_s")
        tr_end_s = self.tik_instance.Scalar("int32",
                                            init_value=core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1],
                                            name="tr_end_s")
        with self.tik_instance.if_scope(tr_start_s != tr_end_s - 1):
            qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, tr_start_s)
            q_blk_h_aligned = (self.Br + 15) // 16 * 16
            lij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="lij_ub")
            mij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="mij_ub")
            Sij_ub_N1MN0 = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16),
                                                    scope=UB, name="Sij_ub_N1MN0")
            # load Qi (q_blk_idx_th block of Q_gm)
            Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.N1, q_blk_h_aligned, self.N0),
                                                      scope=L1, name="Qi_l1_K1MK0_ed")
            self.tik_instance.data_move(dst=Qi_l1_K1MK0_ed, src=self.Q_gm[qo_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                        src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16, dst_stride=0)

            Sij_ub_N1MN0 = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1MK0_ed, m=self.Br,
                                                             k=self.actual_d, n=kv_blk_h,
                                                             N1MN0_to_MN=False, workspace=Sij_ub_N1MN0)  # Qi*KjT

            with self.tik_instance.for_range(tr_start_s, tr_end_s - 1, name="q_blk_idx") as q_blk_idx:
                self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                             batch_start,
                                             kv_blk_h, self.Br, q_blk_idx, kv_blk_idx,
                                             Sij_ub_ws=Sij_ub_N1MN0, Qi_l1_ws=Qi_l1_K1MK0_ed,
                                             lij_ub_ws=lij_ub, mij_ub_ws=mij_ub, tr_end=tr_end_s-2)
        with self.tik_instance.if_scope(tr_end_s - 1 == self.Tr - 1):
            self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                         batch_start,
                                         kv_blk_h, self.last_Br, tr_end_s - 1, kv_blk_idx)
        with self.tik_instance.else_scope():
            self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                         batch_start,
                                         kv_blk_h, self.Br, tr_end_s - 1, kv_blk_idx)

    def compute_in_each_q_block(self, KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx, batch_start,
                                kv_blk_h, q_blk_h, q_blk_idx, kv_blk_idx, Sij_ub_ws=None, Qi_l1_ws=None,
                                lij_ub_ws=None, mij_ub_ws=None, tr_end=None):
        """The forward computation in each inner loop"""
        if not self.valid:
            return
        kv_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_h)
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_h)
        qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        if Sij_ub_ws is None:
            # load Qi (q_blk_idx_th block of Q_gm)
            Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.N1, q_blk_h_aligned, self.N0),
                                                      scope=L1, name="Qi_l1_K1MK0_ed")
            self.tik_instance.data_move(dst=Qi_l1_K1MK0_ed, src=self.Q_gm[qo_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                        src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16, dst_stride=0)

            lij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="lij_ub")
            mij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="mij_ub")

            # QK^T Q shape: (q_blk_h_aligned, self.d), K^T shape: (self.d, kv_blk_h_aligned)
            # M, N (q_blk_h_aligned, kv_blk_h_aligned)
            Sij_ub_N1MN0 = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1MK0_ed, m=q_blk_h,
                                                             k=self.actual_d, n=kv_blk_h,
                                                             N1MN0_to_MN=False)  # Qi*KjT
        else:
            Sij_ub_N1MN0 = Sij_ub_ws
            Qi_l1_K1MK0_ed = Qi_l1_ws
            lij_ub = lij_ub_ws
            mij_ub = mij_ub_ws

        Sij_l1_K1MK0_ed, mij_ub, lij_ub = self.softmax_compute(Sij_ub_N1MN0, mij_ub, lij_ub, q_blk_h, kv_blk_h)
        if tr_end is not None:
            with self.tik_instance.if_scope(q_blk_idx + 1 <= tr_end):
                nxt_qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx + 1)
                self.tik_instance.data_move(dst=Qi_l1_K1MK0_ed, src=self.Q_gm[nxt_qo_gm_offset],
                                            sid=0, nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                            src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16, dst_stride=0)
                Sij_ub_N1MN0 = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1MK0_ed, m=q_blk_h,
                                                                 k=self.actual_d, n=kv_blk_h,
                                                                 N1MN0_to_MN=False, workspace=Sij_ub_N1MN0)  # Qi*KjT

        self.update_o_m_l(
            Sij_l1_K1MK0_ed,
            Vj_l1_K1NK0_ed,
            mij_ub,
            lij_ub,
            batch_start,
            batch_idx,
            kv_blk_idx,
            kv_blk_h,
            q_blk_idx,
            q_blk_h,
            qo_gm_offset
        )

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention forward on each core"""
        if not self.valid:
            return
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc - 1, name="kv_blk_idx") as kv_blk_idx:
                self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                              core_idx_to_tr_info, core_idx)
            # last kv block
            self.compute_in_each_kv_block(batch_start_sc, batch_idx, self.Tc - 1, self.last_Bc,
                                          core_idx_to_tr_info, core_idx)

    def compute_process(self):
        """The compute process of FlashAttention forward"""
        if not self.valid:
            return
        self.init_gm()
        self.tiling()
        self.prepare_global_ones()
        core_idx_to_batch_info, core_idx_to_tr_info = self.get_core_task_info()
        with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                         block_num=self.core_num) as core_idx:
            batch_start_s = self.tik_instance.Scalar("int32", init_value=core_idx_to_batch_info[core_idx, 0],
                                                     name="batch_start_s")
            batch_num_s = self.tik_instance.Scalar("int32", init_value=core_idx_to_batch_info[core_idx, 1],
                                                   name="batch_num_s")

            self.compute_one_core(batch_start_s, batch_num_s, core_idx_to_tr_info, core_idx)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.Q_gm, self.K_gm, self.V_gm],
            outputs=[self.O_gm],
            config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
            enable_l2=True
        )


def flash_attention_compute(q, k, v, y, kernel_name="flash_attention", disable_debug=True):
    fa = FlashAttentionFwd(q=q, k=k, v=v, kernel_name=kernel_name, disable_debug=disable_debug)
    fa.compute_process()
    return fa.tik_instance
