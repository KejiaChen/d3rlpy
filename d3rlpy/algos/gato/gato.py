import dataclasses

from torch import nn

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace
from ...models.builders import create_gato_transformer
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.torch import SeparatorTokenEmbedding
from ...types import Shape
from .base import GatoAlgoBase, GatoBaseConfig
from .torch import GatoImpl, GatoModules

__all__ = ["GatoConfig", "Gato"]


@dataclasses.dataclass()
class GatoConfig(GatoBaseConfig):
    optim_factory: OptimizerFactory = make_optimizer_field()
    initial_learning_rate: float = 1e-7
    maximum_learning_rate: float = 1e-4
    layer_width: int = 3072
    max_observation_length: int = 512
    action_vocab_size: int = 32
    num_heads: int = 24
    context_size: int = 1024
    num_layers: int = 8
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    embed_activation_type: str = "tanh"
    clip_grad_norm: float = 1.0
    warmup_steps: int = 15000
    final_steps: int = 1000000

    def create(self, device: DeviceArg = False) -> "Gato":
        return Gato(self, device)

    @staticmethod
    def get_type() -> str:
        return "gato"


class Gato(GatoAlgoBase[GatoImpl, GatoConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_gato_transformer(
            layer_width=self._config.layer_width,
            max_observation_length=self._config.max_observation_length,
            action_vocab_size=self._config.action_vocab_size,
            num_heads=self._config.num_heads,
            context_size=self._config.context_size,
            num_layers=self._config.num_layers,
            attn_dropout=self._config.attn_dropout,
            resid_dropout=self._config.resid_dropout,
            embed_dropout=self._config.embed_dropout,
            embed_activation_type=self._config.embed_activation_type,
            device=self._device,
        )

        # instantiate embedding modules
        embedding_modules = nn.ModuleDict(
            {
                key: factory.create()
                for key, factory in self._config.embedding_modules.items()
            }
        )
        embedding_modules.to(self._device)
        embedding_module_dict = dict(embedding_modules)

        # instantiate token embeddings
        token_embeddings = {
            key: factory.create(embedding_module_dict)
            for key, factory in self._config.token_embeddings.items()
        }

        # create separator token embedding
        separator_token_embedding = SeparatorTokenEmbedding(
            self._config.layer_width
        )
        separator_token_embedding.to(self._device)

        optim = self._config.optim_factory.create(
            list(transformer.named_modules())
            + list(embedding_modules.named_modules())
            + list(separator_token_embedding.named_modules()),
            lr=self._config.initial_learning_rate,
        )

        modules = GatoModules(
            transformer=transformer,
            embedding_modules=embedding_modules,
            separator_token_embedding=separator_token_embedding,
            optim=optim,
        )

        self._impl = GatoImpl(
            modules=modules,
            token_embeddings=token_embeddings,
            clip_grad_norm=self._config.clip_grad_norm,
            warmup_steps=self._config.warmup_steps,
            final_steps=self._config.final_steps,
            initial_learning_rate=self._config.initial_learning_rate,
            maximum_learning_rate=self._config.maximum_learning_rate,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.TOKEN


register_learnable(GatoConfig)
