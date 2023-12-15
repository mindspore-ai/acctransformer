import os
import sys
from pathlib import Path

import numpy as np
from tbe.common.platform import set_current_compile_soc_info

from common import data_compare
from common import np_impl_attention_forward, np_impl_attention_grad_Vfa

set_current_compile_soc_info("Ascend910")

PRJ_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent
sys.path.append(str(PRJ_ROOT.joinpath("wukong_huahua", "ops", "tbe")))

from flash_attention.flash_attention_fwd import flash_attention
from flash_attention.flash_attention_bwd import flash_attention_grad


def run_forward_debug(q, k, v, mask, att_mask, drop_mask, alibi_mask):
    batch, head_num, q_seq_len, hidden_dim = q.shape
    o = np.random.random((batch, head_num, q_seq_len, hidden_dim)).astype("float16")
    np_O, np_Sim, np_P, np_row_sum, np_row_max = np_impl_attention_forward(q, k, v, att_mask)

    tik_instance = flash_attention({"shape": q.shape, "dtype": "float16"},
                                   {"shape": k.shape, "dtype": "float16"},
                                   {"shape": v.shape, "dtype": "float16"},
                                   {"shape": mask.shape, "dtype": "int8"},
                                   {"shape": (1, q_seq_len, k_seq_len), "dtype": "float16"},
                                   {"shape": (batch, head_num, q_seq_len, k_seq_len), "dtype": "float16"},
                                   {"shape": (batch, head_num, 1, q_seq_len), "dtype": "float16"},
                                   {"shape": o.shape, "dtype": "float16"},
                                   {"shape": (batch, head_num, q_seq_len), "dtype": "float16"},
                                   {"shape": (batch, head_num, q_seq_len), "dtype": "float16"},
                                   disable_debug=False)
    feed_dict = {"Q_gm": q, "K_gm": k, "V_gm": v, "mask_gm": mask, "att_mask_gm": att_mask,
                 "drop_mask_gm": drop_mask, "alibi_mask_gm": alibi_mask}
    (tik_O, l, m) = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
    # O, = tik_instance.StartProfiling(feed_dict=feed_dict)  # 接口不能用

    print(f"tik output shape: {tik_O.shape}, {tik_O}")
    print(f"numpy result vs tik result equal: {np.allclose(tik_O, np_O, 1e-3)}")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count \
        = data_compare(tik_O, np_O)
    print(f"Compare result: {result}, error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
          f"error_greater_than_tenth_count: {error_greater_than_tenth_count}")


def run_backward_debug(Q, K, V, dim_mask, att_mask, drop_mask, alibi_mask):
    B, h, Nq, d = Q.shape
    dO = 10 * np.random.random((B, h, Nq, d)).astype("float16")
    O, Sim, P, l, m = np_impl_attention_forward(Q, K, V, att_mask)
    m_ = np.squeeze(m)
    l_ = np.squeeze(l).astype("float16")

    tik_instance = flash_attention_grad(
        {"shape": Q.shape, "dtype": Q.dtype},
        {"shape": K.shape, "dtype": K.dtype},
        {"shape": V.shape, "dtype": V.dtype},
        {"shape": O.shape, "dtype": O.dtype},
        {"shape": dO.shape, "dtype": dO.dtype},
        {"shape": l_.shape, "dtype": l.dtype},
        {"shape": m_.shape, "dtype": m.dtype},
        {"shape": dim_mask.shape, "dtype": dim_mask.dtype},
        {"shape": (1, Nq, Nq), "dtype": "float16"},
        {"shape": (B, h, Nq, Nq), "dtype": "float16"},
        {"shape": (B, h, 1, Nq), "dtype": "float16"},
        {"shape": Q.shape, "dtype": Q.dtype},
        {"shape": K.shape, "dtype": K.dtype},
        {"shape": V.shape, "dtype": V.dtype},
        disable_debug=False
    )
    feed_dict = {
        "Q_gm": Q,
        "K_gm": K,
        "V_gm": V,
        "O_gm": O,
        "dO_gm": dO,
        "l_gm": l_,
        "m_gm": m_,
        "mask_gm": dim_mask,
        "att_mask_gm": att_mask,
        "drop_mask_gm": drop_mask,
        "alibi_mask_gm": alibi_mask
    }
    tik_dQ, tik_dK, tik_dV = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

    np_dQ, np_dK, np_dV = np_impl_attention_grad_Vfa(Q, K, V, att_mask, l, m, O, dO, drop_mask, alibi_mask)
    print("--------------- tik_dQ -----------------")
    print(tik_dQ)
    print("--------------- np_dQ -----------------")
    print(np_dQ)
    print("\n")

    print("--------------- tik_dK -----------------")
    print(tik_dK)
    print("--------------- np_dK -----------------")
    print(np_dK)
    print("\n")

    print("--------------- tik_dV -----------------")
    print(tik_dV)
    print("--------------- np_dV -----------------")
    print(np_dV)
    print("\n")

    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(np_dQ, tik_dQ)
    print(
        f"Compare result: {result}, dQ error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dQ error_greater_than_tenth_count: {error_greater_than_tenth_count}")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(np_dK, tik_dK)
    print(
        f"Compare result: {result}, dK error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dK error_greater_than_tenth_count: {error_greater_than_tenth_count}")
    result, error_less_than_thousandth_proportion, error_greater_than_tenth_count = data_compare(np_dV, tik_dV)
    print(
        f"Compare result: {result}, dV error_less_than_thousandth_proportion: {error_less_than_thousandth_proportion}, "
        f"dV error_greater_than_tenth_count: {error_greater_than_tenth_count}")


if __name__ == "__main__":
    set_current_compile_soc_info("Ascend910")
    np.random.seed(1)

    batch = 5
    head_num = 8
    q_seq_len = 256
    k_seq_len = 256
    hidden_dim = 80
    np.random.seed(1)
    q = np.random.random((batch, head_num, q_seq_len, hidden_dim)).astype("float16")
    k = np.random.random((batch, head_num, k_seq_len, hidden_dim)).astype("float16")
    v = np.random.random((batch, head_num, k_seq_len, hidden_dim)).astype("float16")
    mask = np.array([1 for _ in range(hidden_dim)]).astype("int8")
    att_mask = np.triu(np.ones(shape=(1, q_seq_len, k_seq_len), dtype=np.float16), k=1)
    alibi_mask = np.zeros((batch, head_num, 1, q_seq_len)).astype("float16")
    print(att_mask)
    drop_mask = np.ones((batch, head_num, q_seq_len, k_seq_len)).astype("float16")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad", action="store_true")
    args = parser.parse_args()

    if args.grad:
        run_backward_debug(q, k, v, mask, att_mask, drop_mask, alibi_mask)
    else:
        run_forward_debug(q, k, v, mask, att_mask, drop_mask, alibi_mask)
