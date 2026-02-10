from abc import ABCMeta, abstractmethod
import re
from typing import Callable, List, Protocol, Sequence


def decompose_descriptor(arg):
    # Currently host-side tensor descriptors are passed as tensor desc + shape + strides.
    # We still need to pass shape/strides after descriptor lowering, so they appear twice.
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", arg.round_f32_to_tf32, *arg.shape, *arg.strides]


def _is_descriptor(arg):
    return isinstance(arg, str) and arg.startswith("tensordesc")


def _count_descriptors(signature):
    if _is_descriptor(signature):
        return 1
    if isinstance(signature, tuple):
        return sum(_count_descriptors(sig) for sig in signature)
    return 0


def _visit_descriptors(args, signature, tensordesc_meta, wrap_descriptor):

    def visit(arg, sig, index):
        if _is_descriptor(sig):
            return (index + 1, wrap_descriptor(arg, tensordesc_meta[index] if tensordesc_meta else None))
        if isinstance(sig, tuple):
            assert isinstance(arg, (list, tuple))
            assert len(arg) == len(sig)
            result = []
            for a, s in zip(arg, sig):
                index, processed = visit(a, s, index)
                if _is_descriptor(s):
                    result.extend(processed)
                else:
                    result.append(processed)
            return (index, tuple(result))
        return (index, arg)

    index, result = visit(tuple(args), tuple(signature), 0)
    assert not tensordesc_meta or index == len(tensordesc_meta)
    return result


def wrap_descriptors(launcher, signature, tensordesc_meta, make_descriptor):
    signature = tuple(signature.values()) if hasattr(signature, "values") else tuple(signature)
    descriptor_count = _count_descriptors(signature)
    if descriptor_count == 0:
        return launcher

    assert not tensordesc_meta or len(tensordesc_meta) == descriptor_count
    if not tensordesc_meta:
        tensordesc_meta = [None] * descriptor_count

    def inner(*args):
        base_args = args[:-1]
        kernel_args = args[-1]
        wrapped = _visit_descriptors(
            kernel_args,
            signature,
            tensordesc_meta,
            lambda a, m: tuple(make_descriptor(a, m, base_args)),
        )
        return launcher(*base_args, wrapped)

    return inner


def _parse_descriptor(descriptor):
    match = re.match(r"tensordesc(?:_im2col)?<([^[>]*)\[([^\]]*)\]", descriptor)
    assert match, f"Malformed tensor descriptor type: {descriptor}"

    dtype = match.group(1)
    block_shape = match.group(2)
    block_ndim = block_shape.count(",") + 1

    rank_match = re.search(r",input_rank=(\d+)", descriptor)
    ndim = int(rank_match.group(1)) if rank_match else block_ndim
    return (dtype, ndim)


def _expand_descriptor(descriptor, meta, descriptor_type):
    dtype, ndim = _parse_descriptor(descriptor)
    expanded = []

    # If there is no descriptor metadata, the descriptor was decomposed to:
    # base pointer, shape, strides, padding, round_f32_to_tf32.
    if meta is None:
        expanded.append("*" + dtype)
        for _ in range(2 * ndim):
            expanded.append("i64")
        expanded.append("i1")
        expanded.append("i1")
    else:
        expanded.append(descriptor_type)

    for _ in range(ndim):
        expanded.append("i32")
    for _ in range(ndim):
        expanded.append("i64")
    return expanded


def expand_signature(signature, tensordesc_meta, descriptor_type):
    signature = tuple(signature)
    expanded = _visit_descriptors(
        signature,
        signature,
        tensordesc_meta,
        lambda a, m: _expand_descriptor(a, m, descriptor_type),
    )
    return list(expanded)


class Benchmarker(Protocol):

    def __call__(self, kernel_call: Callable, *, quantiles: List[float], **kwargs) -> Sequence[float]:
        pass


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for this backend.

        Args:
            ty (str): The Triton type string. e.g., 'i32', '*fp16', 'fp32'.

        Returns:
            str: The C++ type string.
        """
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """
        Return the benchmarking function that this backend should use by default.
        """
        raise NotImplementedError

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
