from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    vocab_size: int = 512
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 256
    block_size: int = 16
    num_gpu_blocks: int = 128
    max_batch_seqs: int = 16
    max_Step_tokens: int = 256
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks must be > 0, got {self.num_gpu_blocks}")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def total_kv_token_capacity(self) -> int:
        return self.num_gpu_blocks * self.block_size
