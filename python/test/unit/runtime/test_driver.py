import sys
from concurrent.futures import ThreadPoolExecutor
import torch

import triton
import triton.language as tl
from triton.backends.driver import expand_signature, wrap_descriptors


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()


def test_expand_signature_with_aggregate_tensordesc():
    signature = (
        "i32",
        ("tensordesc<fp16[16,32]>", "i64"),
        "tensordesc_im2col<fp32[1,16],input_rank=4,'layout'>",
    )
    expanded = expand_signature(signature, [None, {"lowered": True}], "nvTmaDesc")

    assert expanded[0] == "i32"
    assert expanded[1] == (
        "*fp16",
        "i64",
        "i64",
        "i64",
        "i64",
        "i1",
        "i1",
        "i32",
        "i32",
        "i64",
        "i64",
        "i64",
    )
    # input_rank=4 drives the number of shape/stride entries for im2col.
    assert expanded[2:] == ["nvTmaDesc", "i32", "i32", "i32", "i32", "i64", "i64", "i64", "i64"]


def test_wrap_descriptors_handles_aggregate_arguments():
    signature = {0: ("tensordesc<fp16[16,16]>", "i32"), 1: "i64", 2: "tensordesc<fp16[16,16]>"}
    outer_meta = {"tag": "outer"}
    launcher_calls = []

    def launcher(*args):
        launcher_calls.append(args)
        return "ok"

    def make_descriptor(arg, meta, base_args):
        return [("desc", arg, meta, base_args[0]), ("shape", arg)]

    wrapped = wrap_descriptors(launcher, signature, [None, outer_meta], make_descriptor)
    assert wrapped("meta0", "meta1", (("A", 7), 9, "B")) == "ok"

    assert len(launcher_calls) == 1
    assert launcher_calls[0][0] == "meta0"
    assert launcher_calls[0][1] == "meta1"
    assert launcher_calls[0][2] == (
        (("desc", "A", None, "meta0"), ("shape", "A"), 7),
        9,
        ("desc", "B", outer_meta, "meta0"),
        ("shape", "B"),
    )


def test_wrap_descriptors_is_noop_without_tensordesc():

    def launcher(*args):
        return args

    wrapped = wrap_descriptors(launcher, {0: "i32", 1: ("i64", "constexpr")}, None, lambda *_: [])
    assert wrapped is launcher
