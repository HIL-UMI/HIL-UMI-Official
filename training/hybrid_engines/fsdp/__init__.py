# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.


import torch
from packaging import version

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor  # noqa: F401

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        ShardingStrategy,
        fully_shard,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

elif version.parse(torch.__version__) >= version.parse("2.4.0"):
    from torch.distributed._composable.fsdp import (  # noqa: F401
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )
    from torch.distributed.fsdp import (  # noqa: F401
        BackwardPrefetch,
        CPUOffload,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,  # noqa: F401
    )
else:
    raise ImportError(f"Unsupported torch version: {version.parse(torch.__version__)}")
