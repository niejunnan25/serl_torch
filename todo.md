

#### Part 1 Agents

Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/agents/continuous
这个文件夹下面是四个 policy，Agent 策略（如 DRQ、SAC、BC），都是 jax 实现，需要首先将这几个 policy 转换为 torch

1.根据 bc.py 创建 bc_torch.py
2.根据 drq.py 创建 drq_torch.py
3.根据 sac.py 创建 sac_torch.py
4.根据 vice.py 创建 vice_torch.py


###### 疑问1：
/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/agents/__init__.py
这个 __init__.py 是在干嘛？

###### 疑问2：
/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/agents/continuous/drq.py
这个 drq 是什么东西？


#### Part 2 Common

/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/common/common.py
这是在干嘛？

/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/common/encoding.py

```python
    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

```
为什么这里会停止梯度？

为什么是继承 nn.Module（PyTorch 用法），但是实际上走的是 jax？

这里 @nn.compact 又是在干嘛？

```python
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
```

nn.Dense 是什么用法？


```python
class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

```

这个 GCE 是什么用法？

/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/common/optimizers.py
这里的 import optax 是在干嘛？

@optax.inject_hyperparams 是在干嘛？

整个文件有什么用？jax -> torch，这是 jax 的实现吗？如果是 torch 的话，会怎么组织呢？

/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/common

整个 common 都需要 jax -> torch。

#### Part 3 Data

/Users/niejunnan.25/Documents/codebase/serl/serl_launcher/serl_launcher/data/data_store.py
这是在干嘛？


#### Part n Question
agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d

这个 agentlace 是什么？？