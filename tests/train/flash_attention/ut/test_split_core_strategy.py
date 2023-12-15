import mindspore
import mindspore.context as context
import numpy as np
import pytest
from tbe import tik

from flash_attention_test_agent import FlashAttentionTestAgent

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


class SplitCoreComputer(FlashAttentionTestAgent):
    def define_ios(self, *args, **kwargs):
        B, Tr, Tc, core_num, next_block_num = (
            kwargs.get("B"),
            kwargs.get("Tr"),
            kwargs.get("Tc"),
            kwargs.get("core_num"),
            kwargs.get("next_block_num"),
        )
        self.fa.B = B
        self.fa.Tr = Tr
        self.fa.Tc = Tc
        self.fa.core_num = core_num
        self.fa.next_block_num = next_block_num
        self.core_b_map_gm = self.tik_instance.Tensor("int32", (core_num, 2),
                                                      name="core_b_map_gm", scope=tik.scope_gm)
        self.core_b_tr_map_gm = self.tik_instance.Tensor("int32", (core_num, B, 2),
                                                          name="core_b_tr_map_gm", scope=tik.scope_gm)

    def compute(self, *args, **kwargs):
        core_b_map, core_b_tr_map = self.fa.get_core_task_info()

        self.tik_instance.data_move(self.core_b_map_gm, core_b_map, 0, 1,
                                    self.fa.core_num * 2 // 8, 0, 0)
        self.tik_instance.data_move(self.core_b_tr_map_gm, core_b_tr_map, 0, 1,
                                    self.fa.core_num * self.fa.B * 2 // 8, 0, 0)
        self.tik_instance.BuildCCE(
            kernel_name="mock_scale",
            inputs=[],
            outputs=[self.core_b_map_gm, self.core_b_tr_map_gm],
        )
        return self.tik_instance

@pytest.mark.parametrize(
    "B, Tr, Tc, core_num, next_block_num",
    [
        (12, 32, 32, 32, 0),
    ],
)
def test_split_core_compute(B, Tr, Tc, core_num, next_block_num):
    mfa = SplitCoreComputer(**dict(B=B, Tr=Tr, Tc=Tc, core_num=core_num, next_block_num=next_block_num))
    tik_instance = mfa.compute()
    core_b_map, core_b_tr_map = tik_instance.tikdb.start_debug(feed_dict={}, interactive=False)
    print("\n")
    for core_idx in range(core_num):
        b_start, b_num = core_b_map[core_idx]
        print(f"core_idx: {core_idx}, b_start: {b_start}, b_num: {b_num}")
        for b_idx in range(b_num):
            b_offset = b_start + b_idx
            tr_start, tr_end = core_b_tr_map[core_idx, b_offset]
            print(f"core_idx: {core_idx}, b_idx: {b_offset}, tr_start: {tr_start}, tr_end: {tr_end}")
        print("\n")
