# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The forward tik ops of flash attention"""

from tbe import tik

from accspeed.ops.flash_attention.attention import FlashAttention
from accspeed.ops.flash_attention.constants import DTYPE_SIZE
from accspeed.ops.flash_attention.constants import FP16
from accspeed.ops.flash_attention.constants import FP32
from accspeed.ops.flash_attention.constants import GM
from accspeed.ops.flash_attention.constants import L1
from accspeed.ops.flash_attention.constants import UB
from accspeed.ops.flash_attention.tiling_strategy.strategy import TilingStrategy


class FlashAttentionFwd(FlashAttention):
    """The implementation of flash attention forward
    This function contains the flash attention forward implementation used in flash attention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
    """

    def __init__(self, query, key, value,
                 attn_mask, dropout_mask, alibi_mask,
                 kernel_name,
                 tiling_stgy: TilingStrategy,
                 prev_block_num=65536,
                 next_block_num=65536, high_precision=False, disable_debug=True):
        super(FlashAttentionFwd, self).__init__(query, key, value, attn_mask, dropout_mask, alibi_mask,
                                                kernel_name,
                                                tiling_stgy, prev_block_num, next_block_num, high_precision,
                                                disable_debug)
        self.O_gm = None
        self.l_gm = None
        self.m_gm = None
        self.O_gm_workspace = None

    def define_custom_inputs(self):
        """define custom inputs"""

    def define_outputs(self):
        """define outputs"""
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM, is_atomic_add=True)
        if self.high_precision:  # 累加更新O FP32临时空间
            self.O_gm_workspace = self.tik_instance.Tensor(FP32, self.O_shape, name="O_gm_workspace", scope=GM,
                                                           is_workspace=True, is_atomic_add=True)
        self.l_gm = self.tik_instance.Tensor(self.precision_type, self.l_shape, name="l_gm", scope=GM,
                                             is_atomic_add=True)
        self.m_gm = self.tik_instance.Tensor(FP16, self.m_shape, name="m_gm", scope=GM, is_atomic_add=True)

    def prepare_global_ones(self):
        """Prepare global ones tensor in L1 for cube impl row_sum"""
        Bc_aligned = (self.Bc + 15) // 16 * 16
        last_Bc_aligned = (self.last_Bc + 15) // 16 * 16
        self.ones_l1 = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_l1", scope=L1)
        self.last_ones_l1 = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_l1", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            ones_ub = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_ub", scope=UB)
            self.tik_instance.h_duplicate(ones_ub, 1.0)
            self.cont_data_mv_1_bust(dst=self.ones_l1, src=ones_ub, burst=Bc_aligned)
            last_ones_ub = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_ub", scope=UB)
            self.tik_instance.h_duplicate(ones_ub, 1.0)
            self.cont_data_mv_1_bust(dst=self.last_ones_l1, src=last_ones_ub, burst=last_Bc_aligned)

    def softmax_compute(self, Sij_ub, mij_ub, lij_ub, m, n):
        """Refer to Algorithm 2 line12
        Calculate softmax.
        :param Sij_ub: with shape M x N or N1 x M x N0, 使用Sij空间返回Pij 提高UB利用率
        :param mij_ub:
        :param lij_ub:
        :param m:
        :param n:
        :return:
        """
        m_aligned = self.tik_ops_utils.up_align_to_K0(m)
        n_aligned = self.tik_ops_utils.up_align_to_K0(n)
        n0 = 16
        n1 = n // 16
        # 当前只支持 n % 16 == 0
        # TODO：当 n % 16 ！=0时，无效的尾块不能参与计算rowmax
        # calc rowmax of Sij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mn0_block_max = self.tik_instance.Tensor(FP16, (1, m, n0), name="mn0_block_max", scope=UB)
            self.cont_data_mv_1_bust(dst=mn0_block_max, src=Sij_ub, burst=m)  # 用第一个(m, n0)始化，少一次duplicate和max
            with self.tik_instance.for_range(1, n1) as idx:
                self.tik_instance.h_max(mn0_block_max, mn0_block_max, Sij_ub[idx, :, :])
            mn0_block_max = mn0_block_max.reshape((m, n0))
            self.tik_instance.h_reduce_max(mij_ub, mn0_block_max, 1)
        # Sij - mij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_mij_ub = self.tik_ops_utils.broadcast(mij_ub, (m, n0))
            broadcast_mij_ub = broadcast_mij_ub.reshape((1, m, n0))
            for idx in range(n1):
                self.tik_instance.h_sub(Sij_ub[idx, :, :], Sij_ub[idx, :, :], broadcast_mij_ub)
        # exp
        Sij_ub_fp32 = None
        if self.high_precision:
            # fp16 -> fp32
            Sij_ub_fp32 = self.tik_instance.Tensor(
                FP32, (n_aligned // 16, m_aligned, 16), name="Sij_ub_fp32", scope=UB
            )
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                self.tik_instance.h_cast(Sij_ub_fp32, Sij_ub, "none")
                self.tik_instance.h_exp(Sij_ub_fp32, Sij_ub_fp32)
                self.tik_instance.h_cast(Sij_ub, Sij_ub_fp32, "none")
        else:
            self.tik_instance.h_exp(Sij_ub, Sij_ub)

        # cube impl rowsum
        Sij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (n_aligned // 16, m_aligned, 16),
                                                   name="Sij_l1_K1MK0_ed", scope=L1)
        self.cont_data_mv_1_bust(dst=Sij_l1_K1MK0_ed, src=Sij_ub, burst=m * n // 16)
        if n == self.Bc:
            Sij_row_sum_ub = self.tik_ops_utils.row_sum_cube_impl(Sij_l1_K1MK0_ed, self.ones_l1,
                                                                  lij_ub, m, n, self.precision_type)
        else:
            Sij_row_sum_ub = self.tik_ops_utils.row_sum_cube_impl(Sij_l1_K1MK0_ed, self.last_ones_l1,
                                                                  lij_ub, m, n, self.precision_type)

        if self.high_precision:
            return Sij_ub_fp32, mij_ub, Sij_row_sum_ub
        if self.has_drop_mask:
            return Sij_ub, mij_ub, Sij_row_sum_ub
        return Sij_l1_K1MK0_ed, mij_ub, Sij_row_sum_ub

    def update_m_l(self, mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len):
        """Refer to Algorithm 2 line13
        mi_new = max(mi, mij), li_new = exp(mi-mi_new)*li + exp(mij - mi_new) * lij
        """
        dtype = li_old_ub.dtype
        vec_len_aligned = self.tik_ops_utils.up_align_to_K0(vec_len)
        mi_new_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), name="mi_new_ub", scope=UB)
        li_new_ub = self.tik_instance.Tensor(dtype, (vec_len_aligned,), name="li_new_ub", scope=UB)
        # 1 calculate mi_new = max(mi, mij)
        self.tik_instance.h_max(mi_new_ub, mi_old_ub, mij_ub)

        # 2 calculate li_new = exp(mi-mi_new)*li + exp(mij-mi_new)*lij
        # 2.1 相减，求指数
        self.tik_instance.h_sub(mi_old_ub, mi_old_ub, mi_new_ub)  # mi-mi_new
        self.tik_instance.h_exp(mi_old_ub, mi_old_ub)  # exp(mi-mi_new)

        # 2.2 相减，求指数
        self.tik_instance.h_sub(mij_ub, mij_ub, mi_new_ub)  # mij-mi_new
        self.tik_instance.h_exp(mij_ub, mij_ub)  # exp(mij-mi_new)

        # 2.3 相乘，相加 exp(m_ij_ub-mi_new)*li + exp(m_ij_ub-mi_new) * lij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mul_li_ub = self.tik_instance.Tensor(dtype, (vec_len_aligned,), scope=UB, name="mul_li_ub")
            mul_lij_ub = self.tik_instance.Tensor(dtype, (vec_len_aligned,), scope=UB, name="mul_lij_ub")
            if self.high_precision:
                self.tik_instance.h_cast(mul_li_ub, mi_old_ub, "none")
                self.tik_instance.h_cast(mul_lij_ub, mij_ub, "none")
                self.tik_instance.h_mul(mul_li_ub, mul_li_ub, li_old_ub)
                self.tik_instance.h_mul(mul_lij_ub, mul_lij_ub, lij_ub)
            else:
                self.tik_instance.h_mul(mul_li_ub, mi_old_ub, li_old_ub)
                self.tik_instance.h_mul(mul_lij_ub, mij_ub, lij_ub)
            self.tik_instance.h_add(li_new_ub, mul_li_ub, mul_lij_ub)
        return mi_new_ub, li_new_ub

    def update_o_m_l_fp32(self,
                          Pij_ub_fp32,
                          Vj_l1_zN,
                          Pij_ub,
                          mij_ub,
                          lij_ub,
                          batch_start,
                          batch_idx,
                          kv_blk_idx,
                          kv_blk_height,
                          q_blk_idx,
                          block_h):
        """ load o m l from gm and update them in ub, then write them back to gm
        :param Pij_ub_fp32: input tensor with format zN, shape (q_blk_h_aligned, k_blk_h_aligned)
        :param Vj_l1_zN: input tensor with format zN, (v_blk_h_aligned, self.d)
        :param Pij_ub: shape (q_blk_h_aligned, k_blk_h_aligned)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param q_blk_idx:
        :param block_h:
        :return: None
        """
        vec_gm_offset = self.get_l_m_gm_offset(batch_start, batch_idx, self.Nq, self.Br, q_blk_idx)
        o_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        block_h_aligned = self.tik_ops_utils.up_align_to_K0(block_h)
        block_k_aligned_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        n1 = block_k_aligned_aligned // self.N0
        # 第一次外循环时，不用走更新逻辑， 减少的无用计算
        # mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(tik.any(kv_blk_idx == 0, kv_blk_idx + self.prev_block_num == q_blk_idx)):
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(lij_ub, block_h)
            # TODO：FP32广播接口，去掉循环 => 提升性能
            vec_ub = self.tik_instance.Tensor(FP32, (block_h, self.N0), name="vec_ub", scope=UB)
            for i in range(block_h):
                src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP32)
                self.tik_instance.h_duplicate(vec_ub[i, :], src_scalar)
            vec_ub = vec_ub.reshape((1, block_h, self.N0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Pij_ub_fp32[idx, :, :],
                                        Pij_ub_fp32[idx, :, :],
                                        vec_ub)
            self.tik_instance.h_cast(Pij_ub, Pij_ub_fp32, "none")
            # ub -> l1
            Pij_l1_zN = self.tik_instance.Tensor(
                FP16, (block_k_aligned_aligned // 16, block_h_aligned, 16), name="Pij_l1_K1MK0_ed", scope=L1
            )
            # Pij_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Pij_ub, workspace_tensor=Pij_l1_K1MK0_ed)
            self.cont_data_mv_1_bust(dst=Pij_l1_zN, src=Pij_ub,
                                     burst=block_k_aligned_aligned * block_h_aligned // 16)
            # UB 使用: Pij_ub + Pij_ub_f32 * 2
            Pij_Vj_matmul_res_ub = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, block_h,
                                                                   kv_blk_height, self.actual_d, N1MN0_to_MN=False,
                                                                   precision_type=self.precision_type)  # Pij*Vj
            self.unroll_data_move(dst=self.O_gm_workspace,
                                  dst_offset=o_gm_offset,
                                  src=Pij_Vj_matmul_res_ub,
                                  src_offset=0,
                                  nburst=self.N1,
                                  burst=block_h * self.N0 // 8,
                                  src_stride=0,
                                  dst_stride=(self.Nq - block_h_aligned) * self.N0 // 8,
                                  dst_offset_inc=self.Nq * self.N0,
                                  src_offset_inc=block_h * self.N0)
        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h_aligned,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP32, (block_h_aligned,), name="li_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset, block_h)

            # 更新 l, m
            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

            exp_m_old_fp32 = self.tik_instance.Tensor(FP32, (block_h_aligned,), scope=UB, name="exp_m_old_fp32")
            exp_m_cur_fp32 = self.tik_instance.Tensor(FP32, (block_h_aligned,), scope=UB, name="exp_m_cur_fp32")
            self.tik_instance.h_cast(exp_m_old_fp32, mi_ub, "none")
            self.tik_instance.h_cast(exp_m_cur_fp32, mij_ub, "none")

            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(li_new_ub, block_h)
            self.tik_instance.h_mul(exp_m_cur_fp32, exp_m_cur_fp32, li_new_rec_ub)
            exp_m_cur_fp32_vec_ub = self.tik_instance.Tensor(FP32, (block_h, self.N0), name="exp_m_cur_fp32_vec_ub",
                                                             scope=UB)
            for i in range(block_h):
                src_scalar = self.tik_instance.Scalar(init_value=exp_m_cur_fp32[i], dtype=FP32)
                self.tik_instance.h_duplicate(exp_m_cur_fp32_vec_ub[i, :], src_scalar)
            exp_m_cur_fp32_vec_ub = exp_m_cur_fp32_vec_ub.reshape((1, block_h, self.N0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Pij_ub_fp32[idx, :, :],
                                        Pij_ub_fp32[idx, :, :],
                                        exp_m_cur_fp32_vec_ub)
            self.tik_instance.h_cast(Pij_ub, Pij_ub_fp32, "none")
            # ub -> l1
            Pij_l1_zN = self.tik_instance.Tensor(
                FP16, (block_k_aligned_aligned // 16, block_h_aligned, 16), name="Pij_l1_K1MK0_ed", scope=L1
            )
            self.cont_data_mv_1_bust(dst=Pij_l1_zN, src=Pij_ub,
                                     burst=block_k_aligned_aligned * block_h_aligned // 16)
            Pij_Vj_matmul_res_ub = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, block_h,
                                                                   kv_blk_height, self.actual_d, N1MN0_to_MN=False,
                                                                   precision_type=self.precision_type)  # Pij*Vj
            n1, m, n0 = Pij_Vj_matmul_res_ub.shape
            # 载入Oi 到 UB
            Oi_ub = self.tik_instance.Tensor(FP32, (n1, m, n0), name="Oi_ub", scope=UB)
            self.unroll_data_move(dst=Oi_ub,
                                  dst_offset=0,
                                  src=self.O_gm_workspace,
                                  src_offset=o_gm_offset,
                                  nburst=self.N1,
                                  burst=m * self.N0 // 8,
                                  src_stride=(self.Nq - m) * self.N0 // 8,
                                  dst_stride=0,
                                  dst_offset_inc=m * self.N0,
                                  src_offset_inc=self.Nq * self.N0)

            self.tik_instance.h_mul(li_new_rec_ub, li_new_rec_ub, li_ub)
            self.tik_instance.h_mul(li_new_rec_ub, li_new_rec_ub, exp_m_old_fp32)
            # li_new_rec_vec_ub = self.tik_instance.Tensor(FP32, (block_h, self.N0), name="li_new_rec_vec_ub",
            #                                              scope=UB)
            li_new_rec_vec_ub = exp_m_cur_fp32_vec_ub.reshape((block_h, self.N0))
            for i in range(block_h):
                src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP32)
                self.tik_instance.h_duplicate(li_new_rec_vec_ub[i, :], src_scalar)
            li_new_rec_vec_ub = li_new_rec_vec_ub.reshape((1, block_h, self.N0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Oi_ub[idx, :, :],
                                        Oi_ub[idx, :, :],
                                        li_new_rec_vec_ub)
            self.tik_instance.h_add(Oi_ub, Oi_ub, Pij_Vj_matmul_res_ub)

            self.unroll_data_move(dst=self.O_gm_workspace,
                                  dst_offset=o_gm_offset,
                                  src=Oi_ub,
                                  src_offset=0,
                                  nburst=self.N1,
                                  burst=block_h * self.N0 // 8,
                                  src_stride=0,
                                  dst_stride=(self.Nq - block_h_aligned) * self.N0 // 8,
                                  dst_offset_inc=self.Nq * self.N0,
                                  src_offset_inc=block_h * self.N0)

    def exp_Pij_Vj(self, exp_mij_sub_mi_new, Pij_Vj_ub, block_h_aligned):
        """Refer to Algorithm 2 line15
        exp(mij - mi_new) * Pij * Vj
        """
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_exp_mij_sub_mi_new = self.tik_ops_utils.broadcast(exp_mij_sub_mi_new, (block_h_aligned, self.d))
            self.tik_instance.h_mul(Pij_Vj_ub, Pij_Vj_ub, broadcast_exp_mij_sub_mi_new)
        return Pij_Vj_ub

    def update_o_m_l(self,
                     Pij_l1_zN,
                     Vj_l1_zN,
                     mij_ub,
                     lij_ub,
                     batch_start,
                     batch_idx,
                     kv_blk_idx,
                     kv_blk_height,
                     q_blk_idx,
                     block_h):
        """Refer to Algorithm 2 line13 and line15 in FlashAttention
        load o m l from gm and update them in ub, then write them back to gm
        :param Pij_l1_zN: input tensor with format zN, shape (q_blk_h_aligned, k_blk_h_aligned)
        :param Vj_l1_zN: input tensor with format zN, (v_blk_h_aligned, self.d)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param q_blk_idx:
        :param block_h:
        :return: None
        """
        vec_gm_offset = self.get_l_m_gm_offset(batch_start, batch_idx, self.Nq, self.Br, q_blk_idx)
        o_gm_offset = self.get_gm_offset(
            batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx
        )
        block_h_aligned = self.tik_ops_utils.up_align_to_K0(block_h)

        Pij_Vj_matmul_res_ub = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, block_h,
                                                                 kv_blk_height, self.actual_d,
                                                                 N1MN0_to_MN=False)  # Pij*Vj
        n1, m, n0 = Pij_Vj_matmul_res_ub.shape
        # 第一次外循环时，不用走更新逻辑， 减少的无用计算
        # calculate mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(tik.any(kv_blk_idx == 0, kv_blk_idx + self.prev_block_num == q_blk_idx)):
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(lij_ub, block_h)
            # TODO：NZ版double buffer
            broadcast_li_new_rec_ub = self.tik_ops_utils.broadcast(li_new_rec_ub, (m, n0))
            broadcast_li_new_rec_ub = broadcast_li_new_rec_ub.reshape((1, m, n0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Pij_Vj_matmul_res_ub[idx, :, :],
                                        Pij_Vj_matmul_res_ub[idx, :, :],
                                        broadcast_li_new_rec_ub)
            self.unroll_data_move(dst=self.O_gm,
                                  dst_offset=o_gm_offset,
                                  src=Pij_Vj_matmul_res_ub,
                                  src_offset=0,
                                  nburst=self.N1,
                                  burst=block_h * self.N0 // 16,
                                  src_stride=0,
                                  dst_stride=(self.Nq - block_h_aligned) * self.N0 // 16,
                                  dst_offset_inc=self.Nq * self.N0,
                                  src_offset_inc=block_h * self.N0)

        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h_aligned,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (block_h_aligned,), name="li_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset, block_h)

            # 更新 l, m
            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub

            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(li_new_ub, block_h)
            # TODO：NZ版double buffer
            # scale1 <- li * exp(mi - mi_new) / li_new
            self.tik_instance.h_mul(li_ub, li_ub, exp_mi_sub_mi_new)
            self.tik_instance.h_mul(li_ub, li_ub, li_new_rec_ub)
            scale1 = li_ub
            # scale2 <- exp(mij - mi_new) / li_new
            self.tik_instance.h_mul(exp_mij_sub_mi_new, exp_mij_sub_mi_new, li_new_rec_ub)
            scale2 = exp_mij_sub_mi_new
            Oi_ub = self.tik_instance.Tensor(FP16, (n1, m, n0), name="Oi_ub", scope=UB)
            self.unroll_data_move(dst=Oi_ub,
                                  dst_offset=0,
                                  src=self.O_gm,
                                  src_offset=o_gm_offset,
                                  nburst=self.N1,
                                  burst=m * self.N0 // 16,
                                  src_stride=(self.Nq - m) * self.N0 // 16,
                                  dst_stride=0,
                                  dst_offset_inc=m * self.N0,
                                  src_offset_inc=self.Nq * self.N0)

            broadcast_scale1 = self.tik_ops_utils.broadcast(scale1, (m, n0))
            broadcast_scale1 = broadcast_scale1.reshape((1, m, n0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Oi_ub[idx, :, :], Oi_ub[idx, :, :], broadcast_scale1)
            # vec: scale2 * Pij_Vj
            broadcast_scale2 = self.tik_ops_utils.broadcast(scale2, (m, n0))
            broadcast_scale2 = broadcast_scale2.reshape((1, m, n0))
            with self.tik_instance.for_range(0, n1) as idx:
                self.tik_instance.h_mul(Pij_Vj_matmul_res_ub[idx, :, :],
                                        Pij_Vj_matmul_res_ub[idx, :, :],
                                        broadcast_scale2)
            self.tik_instance.h_add(Oi_ub, Oi_ub, Pij_Vj_matmul_res_ub)
            self.unroll_data_move(dst=self.O_gm,
                                  dst_offset=o_gm_offset,
                                  src=Oi_ub,
                                  src_offset=0,
                                  nburst=self.N1,
                                  burst=block_h * self.N0 // 16,
                                  src_stride=0,
                                  dst_stride=(self.Nq - block_h_aligned) * self.N0 // 16,
                                  dst_offset_inc=self.Nq * self.N0,
                                  src_offset_inc=block_h * self.N0)

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info, core_idx):
        """The forward computation in each outer loop"""
        kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc, kv_blk_idx)
        # load Kj (kv_blk_idx_th block of K_gm)
        KjT_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_height_aligned, self.N0),
                                                   name="KjT_l1_K1MK0_ed", scope=L1)
        self.unroll_data_move(dst=KjT_l1_K1MK0_ed,
                              dst_offset=0,
                              src=self.K_gm,
                              src_offset=kv_gm_offset,
                              nburst=self.N1,
                              burst=kv_blk_height_aligned * self.N0 // 16,
                              src_stride=(self.N - kv_blk_height_aligned) * self.N0 // 16,
                              dst_stride=0,
                              dst_offset_inc=kv_blk_height_aligned * self.N0,
                              src_offset_inc=self.N * self.N0)

        # load Vj (kv_blk_idx_th block of V_gm), then reorder for Pij*Vj
        Vj_l1_zN = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Vj_l1_zN", scope=L1)
        self.unroll_data_move(dst=Vj_l1_zN,
                              dst_offset=0,
                              src=self.V_gm,
                              src_offset=kv_gm_offset,
                              nburst=self.N1,
                              burst=kv_blk_height_aligned * self.N0 // 16,
                              src_stride=(self.N - kv_blk_height_aligned) * self.N0 // 16,
                              dst_stride=0,
                              dst_offset_inc=kv_blk_height_aligned * self.N0,
                              src_offset_inc=self.N * self.N0)

        tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
        tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
        tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
        tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
        with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
            # 根据atten_mask倒三角特性，过滤无效计算
            with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,
                                                    q_blk_idx <= kv_blk_idx + self.prev_block_num)):
                with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                    self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_zN, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.Br, q_blk_idx, kv_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_zN, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.last_Br, q_blk_idx, kv_blk_idx)

    def compute_in_each_q_block(self, KjT_l1_K1MK0_ed, Vj_l1_zN, batch_idx, batch_start,
                                kv_blk_height, q_blk_height, q_blk_idx, kv_blk_idx):
        """The forward computation in each inner loop"""
        kv_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
        # load Qi (q_blk_idx_th block of Q_gm), then reorder it fo Qi*KjT
        q_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_h_aligned, self.N0),
                                                  scope=L1, name="Qi_l1_K1MK0_ed")
        self.unroll_data_move(dst=Qi_l1_K1MK0_ed,
                              dst_offset=0,
                              src=self.Q_gm,
                              src_offset=q_gm_offset,
                              nburst=self.N1,
                              burst=q_blk_h_aligned * self.N0 // 16,
                              src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16,
                              dst_stride=0,
                              dst_offset_inc=q_blk_h_aligned * self.N0,
                              src_offset_inc=self.Nq * self.N0)

        lij_ub = self.tik_instance.Tensor(self.precision_type, (q_blk_h_aligned,), scope=UB, name="lij_ub")
        mij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="mij_ub")

        # QK^T Q shape: (q_blk_h_aligned, self.d), K^T shape: (self.d, kv_blk_h_aligned)
        # M, N (q_blk_h_aligned, kv_blk_h_aligned)
        Sij_ub_N1MN0 = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1MK0_ed, m=q_blk_height,
                                                         k=self.actual_d, n=kv_blk_height,
                                                         N1MN0_to_MN=False)  # Qi*KjT
        if self.has_alibi_mask:  # TODO NZ需要维护
            alibi_mask_gm_offset = self.get_alibi_gm_offset(batch_start, batch_idx, self.N, self.Bc, kv_blk_idx)
            self.do_alibi_mask(Sij_ub_N1MN0, alibi_mask_gm_offset, q_blk_h_aligned, kv_blk_h_aligned)

        # att_mask
        if self.has_attn_mask:
            with self.tik_instance.if_scope(q_blk_idx == kv_blk_idx):
                self.do_att_mask(Sij_ub_N1MN0, q_blk_height, kv_blk_height,
                                 q_blk_h_aligned, kv_blk_h_aligned)

        Pij_N1MN0, mij_ub, lij_ub = self.softmax_compute(
            Sij_ub_N1MN0, mij_ub, lij_ub, q_blk_height, kv_blk_height
        )  # self.high_precision=True, Pij_ub return type fp32
        # dropout_mask
        if self.has_drop_mask:
            dropout_mask_gm_offset = self.get_drop_mask_gm_offset(batch_start, batch_idx, self.Nq,
                                                                  self.N, self.Br, q_blk_idx, self.Bc, kv_blk_idx)
            self.do_dropout_mask(Pij_N1MN0, dropout_mask_gm_offset, kv_blk_h_aligned, kv_blk_height,
                                 q_blk_h_aligned, q_blk_height, precision_type=self.precision_type)
            if not self.high_precision:
                Pij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16,
                                                           (kv_blk_h_aligned // self.N0, q_blk_h_aligned, self.N0),
                                                           name="Pij_l1_K1MK0_ed", scope=L1)
                self.cont_data_mv_1_bust(dst=Pij_l1_K1MK0_ed, src=Pij_N1MN0,
                                         burst=kv_blk_h_aligned * q_blk_h_aligned // 16)
                Pij_N1MN0 = Pij_l1_K1MK0_ed
        if self.high_precision:
            self.update_o_m_l_fp32(
                Pij_N1MN0,
                Vj_l1_zN,
                Sij_ub_N1MN0,  # fp16, 与Pij_ub内存共用, fp32, 内存保留做类型转换使用，避免重新申请;
                mij_ub,
                lij_ub,
                batch_start,
                batch_idx,
                kv_blk_idx,
                kv_blk_height,
                q_blk_idx,
                q_blk_height
            )
        else:
            self.update_o_m_l(
                Pij_N1MN0,
                Vj_l1_zN,
                mij_ub,
                lij_ub,
                batch_start,
                batch_idx,
                kv_blk_idx,
                kv_blk_height,
                q_blk_idx,
                q_blk_height
            )

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention forward on each core"""
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.Tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_Bc,
                                                  core_idx_to_tr_info, core_idx)
            if self.high_precision:
                # O_gm_workspace -> O_gm
                block_h = 128
                gm_offset = (batch_start_sc + batch_idx) * (self.Nq * self.d)
                if self.Nq // block_h // 2 > 0:
                    with self.tik_instance.for_range(0, self.Nq // block_h // 2) as i:
                        with self.tik_instance.for_range(0, 2, thread_num=2) as t_idx:
                            temp_ub = self.tik_instance.Tensor(FP32, (block_h, self.d), name="temp_ub", scope=UB)
                            temp_ub_fp16 = self.tik_instance.Tensor(FP16, (block_h, self.d), name="temp_ub_fp16",
                                                                    scope=UB)
                            index = i * 2 + t_idx
                            gm_offset += index * (block_h * self.d)
                            self.cont_data_mv_1_bust(dst=temp_ub, src=self.O_gm_workspace[gm_offset],
                                                     burst=block_h * self.d * DTYPE_SIZE[FP32] // 32)
                            self.tik_instance.h_cast(temp_ub_fp16, temp_ub, "none")
                            self.cont_data_mv_1_bust(dst=self.O_gm[gm_offset], src=temp_ub_fp16,
                                                     burst=block_h * self.d * DTYPE_SIZE[FP16] // 32)
                if self.Nq % (block_h * 2) > 0:  # 尾块处理
                    temp_ub = self.tik_instance.Tensor(FP32, (block_h, self.d), name="temp_ub", scope=UB)
                    temp_ub_fp16 = self.tik_instance.Tensor(FP16, (block_h, self.d), name="temp_ub_fp16", scope=UB)
                    gm_offset = (batch_start_sc + batch_idx) * (self.Nq * self.d) + \
                                (self.Nq // (block_h * 2) * 2) * (block_h * self.d)
                    last_block_h = self.Nq % (block_h * 2)
                    self.cont_data_mv_1_bust(dst=temp_ub, src=self.O_gm_workspace[gm_offset],
                                             burst=last_block_h * self.d * DTYPE_SIZE[FP32] // 32)
                    self.tik_instance.h_cast(temp_ub_fp16, temp_ub, "none")
                    self.cont_data_mv_1_bust(dst=self.O_gm[gm_offset], src=temp_ub_fp16,
                                             burst=last_block_h * self.d * DTYPE_SIZE[FP16] // 32)

    def collect_inputs(self):
        """collect all input gm tensors into input_gm_list,
        the input list should keep order with the para order in Primitive and init
        """
        input_gm_list = [self.Q_gm, self.K_gm, self.V_gm]
        if self.has_attn_mask:
            input_gm_list.append(self.att_mask_gm)
        if self.has_drop_mask:
            input_gm_list.append(self.drop_mask_gm)
        if self.has_alibi_mask:
            input_gm_list.append(self.alibi_mask_gm)

        return input_gm_list

    def collect_outputs(self):
        """collect all output gm tensors into output_gm_list,
        the output list should keep order with the para order in Primitive and init
        """
        output_gm_list = [self.O_gm, self.l_gm, self.m_gm]
        return output_gm_list


def flash_attention(query, key, value, attn_mask, dropout_mask, alibi_mask, output, rowsum, rowmax,
                    prev_block_num=65536, next_block_num=65536, high_precision=False, tiling_stgy_name='sparse',
                    kernel_name="flash_attention", disable_debug=True):
    """
    algorithm: flash_attention_backward

    Parameters
    ----------
    query : dict. shape and dtype of input, only support float16
    key : dict. shape and dtype of input, only support float16
    value: dict. shape and dtype of input, only support float16
    attn_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    alibi_mask: dict. shape and dtype of input, only support float16
    output: dict. shape and dtype of output, only support float16
    rowsum: dict. shape and dtype of output, only support float16
    rowmax: dict. shape and dtype of output, only support float16
    prev_block_num: int. an attribute used to define sparse attention
    next_block_num: int. an attribute used to define sparse attention
    high_precision: whether enable high precision mode.
    tiling_stgy_name: str. an attribute used to choose the tiling strategy
    kernel_name: str. cce kernel name, default value is real_div
    disable_debug: bool. whether disable debug

    Returns
    -------
    tik_instance
    """
    fa = FlashAttentionFwd(query=query, key=key, value=value, attn_mask=attn_mask,
                           dropout_mask=dropout_mask, alibi_mask=alibi_mask, kernel_name=kernel_name,
                           tiling_stgy=TilingStrategy.from_strategy_name(tiling_stgy_name),
                           prev_block_num=prev_block_num, next_block_num=next_block_num,
                           high_precision=high_precision, disable_debug=disable_debug)
    fa.compute_process()
    return fa.tik_instance
