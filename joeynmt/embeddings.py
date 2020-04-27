import math
import torch
from torch import nn, Tensor
from joeynmt.helpers import freeze_params


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs) -> None:
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)


def concatenate_embeddings(src_embedded: Tensor, factor_embedded: Tensor) -> Tensor:
    """
    Concatenate embeddings to combine source words and their factors.

    :param src_embedded: embedded src inputs,
        shape (batch_size, src_len, src_embed_size)
    :param factor_embedded:
        shape (batch_size, src_len, factor_embed_size)
    :return:
    """

    # check shapes
    assert src_embedded.shape[0] == factor_embedded.shape[0]
    assert src_embedded.shape[1] == factor_embedded.shape[1]

    print(src_embedded.shape)
    print(factor_embedded.shape)

    concat_embedded = torch.cat((src_embedded, factor_embedded), -1)

    print(concat_embedded.shape)

    return concat_embedded


def sum_embeddings(src_embedded: Tensor, factor_embedded: Tensor) -> Tensor:
    """
    Sum embeddings to combine source words and their factors.

    :param src_embedded: embedded src inputs,
        shape (batch_size, src_len, src_embed_size)
    :param factor_embedded:
        shape (batch_size, src_len, factor_embed_size)
    :return:
    """

    # check shapes
    assert src_embedded.shape[0] == factor_embedded.shape[0]
    assert src_embedded.shape[1] == factor_embedded.shape[1]
    assert src_embedded.shape[2] == factor_embedded.shape[2]

    print(src_embedded.shape)
    print(factor_embedded.shape)

    sum_embedded = src_embedded + factor_embedded

    print(sum_embedded.shape)

    return sum_embedded
