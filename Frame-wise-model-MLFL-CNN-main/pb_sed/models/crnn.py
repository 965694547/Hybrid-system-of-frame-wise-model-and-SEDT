import numpy as np
import torch
from einops import rearrange
from padertorch import Model
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.conv_utils import Pad
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import Mean, TakeLast
from padertorch.contrib.je.modules.rnn import GRU, reverse_sequence
from pb_sed.evaluation.instance_based import get_optimal_thresholds
from torch import nn
from torchvision.utils import make_grid


class CRNN(Model):
    """
    >>> config = CRNN.get_config({\
            'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
            'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
            'rnn_fwd': {'hidden_size': 64},\
            'clf_fwd': {'out_channels':[32,10], 'kernel_size': 1},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'fft_length': 512,\
                'n_mels': 80,\
            },\
        })
    >>> crnn = CRNN.from_config(config)
    >>> inputs = {'stft': torch.randn((4, 1, 15, 257, 2)), 'seq_len': [15, 14, 13, 12], 'events': torch.zeros((4,10))}
    >>> outputs = crnn(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 10, 15])
    >>> review = crnn.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn_2d, cnn_1d,
            rnn_fwd, clf_fwd, rnn_bwd, clf_bwd, *,
            framewise_training=True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._rnn_fwd = rnn_fwd
        self._clf_fwd = clf_fwd
        self._rnn_bwd = rnn_bwd
        self._clf_bwd = clf_bwd
        self.framewise_training = framewise_training

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x, seq_len = self._cnn_2d(x, seq_len)
        if x.dim() != 3:
            assert x.dim() == 4, x.shape
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x, seq_len = self._cnn_1d(x, seq_len)
        return x, seq_len

    def fwd_classification(self, x, seq_len=None):
        x = rearrange(x, 'b f t -> b t f')
        x = self._rnn_fwd(x, seq_len)
        x = rearrange(x, 'b t f -> b f t')
        y, seq_len_y = self._clf_fwd(x, seq_len)
        return nn.Sigmoid()(y), seq_len_y

    def bwd_classification(self, x, seq_len=None):
        x = rearrange(x, 'b f t -> b t f')
        x = reverse_sequence(
            self._rnn_bwd(reverse_sequence(x, seq_len), seq_len), seq_len
        )
        x = rearrange(x, 'b t f -> b f t')
        y, seq_len_y = self._clf_bwd(x, seq_len)
        return nn.Sigmoid()(y), seq_len_y

    def prediction_pooling(self, y_fwd, y_bwd, seq_len):
        if y_bwd is None:
            if self.training and self.framewise_training:
                y = y_fwd
            else:
                y = TakeLast(axis=-1)(y_fwd, seq_len=seq_len)
                seq_len = None
        elif self.training:
            if self.framewise_training:
                y = torch.max(y_fwd, y_bwd)
            else:
                y = torch.stack((TakeLast(axis=-1)(y_fwd, seq_len=seq_len), y_bwd[:, ..., 0]), dim=-1)
                seq_len = None
        else:
            y = (TakeLast(axis=-1)(y_fwd, seq_len=seq_len) + y_bwd[:, ..., 0]) / 2
            seq_len = None
        return y, seq_len

    def sed(self, x, context, seq_len=None):
        """
        SED by applying the model to small segments around each frame

        Args:
            x:
            context:
            seq_len:

        Returns:

        """
        x, seq_len = self.feature_extractor(x, seq_len=seq_len)
        h, seq_len = self.cnn_2d(x, seq_len)
        h, seq_len = self.cnn_1d(h, seq_len)
        b, f, t = h.shape
        h = Pad()(h, 2 * context)
        h = torch.cat([h[..., i:i + 1 + 2 * context] for i in range(t)])
        y_fwd, _ = self.fwd_classification(h)
        if self._rnn_bwd is None:
            y_sed = y_fwd[..., -1]
        else:
            y_bwd, _ = self.bwd_classification(h)
            y_sed = (y_fwd[..., -1] + y_bwd[..., 0]) / 2
        y_sed = rearrange(y_sed, '(t b) k -> b t k', b=b, t=t)
        return y_sed, seq_len

    def forward(self, inputs):
        """
        forward used in trainer

        Args:
            inputs: example dict

        Returns:

        """
        x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        x, seq_len = self.feature_extractor(x, seq_len=seq_len)
        h, seq_len = self.cnn_2d(x, seq_len)
        h, seq_len = self.cnn_1d(h, seq_len)
        y_fwd, seq_len_y = self.fwd_classification(h, seq_len=seq_len)
        if self._rnn_bwd is None:
            assert self._clf_bwd is None
            y_bwd = None
        else:
            y_bwd, _ = self.bwd_classification(h, seq_len=seq_len)
        return (y_fwd, y_bwd, seq_len_y), x

    def review(self, inputs, outputs):
        """
        compute loss and metrics

        Args:
            inputs:
            outputs:

        Returns:

        """
        (y_fwd, y_bwd, seq_len_y), x = outputs

        y, seq_len_y = self.prediction_pooling(y_fwd, y_bwd, seq_len_y)
        targets = inputs['events']

        if y.dim() == 3 and targets.dim() == 2:   # (B, K)
            targets = targets.unsqueeze(-1).expand(y.shape)
        assert targets.dim() == y.dim(), (targets.shape, y.shape)
        bce = nn.BCELoss(reduction='none')(y, targets).sum(1)
        if bce.dim() > 1:
            assert bce.dim() == 2, bce.shape
            bce = Mean(axis=-1)(bce, seq_len_y)
        bce = bce.mean()

        if y.dim() == 3:
            if y_bwd is None:
                y = TakeLast(axis=-1)(y_fwd, seq_len=seq_len_y)
            else:
                y = (TakeLast(axis=-1)(y_fwd, seq_len=seq_len_y) + y_bwd[:, ..., 0]) / 2
            targets = targets.max(-1)[0]
        review = dict(
            loss=bce,
            images=dict(
                features=x[:3],
            ),
            buffers=dict(
                predictions=y.data.cpu().numpy(),
                targets=targets.data.cpu().numpy(),
            ),
        )
        return review

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary

        Args:
            summary:

        Returns:

        """
        if f'predictions' in summary['buffers']:
            # Computes fscores from predictions and targets
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            k = predictions.shape[-1]
            targets = np.concatenate(summary['buffers'].pop('targets'))
            best_thresholds, best_f = get_optimal_thresholds(
                targets, predictions, metric='f1'
            )
            for i in range(k):
                summary['scalars'][f'fscores/{i}'] = best_f[i]
                summary['scalars'][f'thresholds/{i}'] = np.minimum(np.maximum(best_thresholds[i], 0), 1)
            summary['scalars'][f'mean_fscore'] = best_f.mean()

        for key, scalar in summary['scalars'].items():
            # average scalar metrics over batches
            summary['scalars'][key] = np.mean(scalar)

        for key, image in summary['images'].items():
            # prepare image grid for tensorboard
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Automatically prepares/completes the configuration of the model.

        You do not need to understand how this is working as there is a lot of
        magic in the background which serves convenience and is not crucial to
        run the model.

        Args:
            config:

        Returns:

        """
        config['feature_extractor'] = {'factory': NormalizedLogMelExtractor}
        config['cnn_2d'] = {'factory': CNN2d}
        config['cnn_1d'] = {'factory': CNN1d}
        config['rnn_fwd'] = {'factory': GRU}
        config['clf_fwd'] = {'factory': CNN1d}
        config['rnn_bwd'] = {'factory': GRU}
        config['clf_bwd'] = {'factory': CNN1d}
        input_size = config['feature_extractor']['n_mels']
        if config['cnn_2d'] is not None and input_size is not None:
            config['cnn_2d']['in_channels'] = 1
            in_channels = config['cnn_2d']['in_channels']
            cnn_2d = config['cnn_2d']['factory'].from_config(config['cnn_2d'])
            output_size = cnn_2d.get_shapes((1, in_channels, input_size, 1000))[-1][2]
            input_size = cnn_2d.out_channels[-1] * output_size

        if config['cnn_1d'] is not None:
            if input_size is not None:
                config['cnn_1d']['in_channels'] = input_size
            input_size = config['cnn_1d']['out_channels'][-1]

        if config['rnn_fwd'] is not None:
            if config['rnn_fwd']['factory'] == GRU:
                config['rnn_fwd'].update({
                    'num_layers': 1,
                    'bias': True,
                    'dropout': 0.,
                    'bidirectional': False
                })

            if input_size is not None:
                config['rnn_fwd']['input_size'] = input_size

        if config['rnn_bwd'] is not None:
            if config['rnn_fwd'] is not None and config['rnn_bwd']['factory'] == config['rnn_fwd']['factory']:
                config['rnn_bwd'].update(config['rnn_fwd'].to_dict())
            elif config['rnn_bwd']['factory'] == GRU:
                config['rnn_bwd'].update({
                    'num_layers': 1,
                    'bias': True,
                    'dropout': 0.,
                    'bidirectional': False
                })

            if input_size is not None:
                config['rnn_bwd']['input_size'] = input_size

        if config['rnn_fwd'] is not None:
            input_size = config['rnn_fwd']['hidden_size']
        elif config['rnn_bwd'] is not None:
            input_size = config['rnn_bwd']['hidden_size']

        if config['clf_fwd'] is not None and config['clf_fwd']['factory'] == CNN1d:
            config['clf_fwd']['in_channels'] = input_size

        if config['clf_bwd'] is not None:
            if config['clf_fwd'] is not None and config['clf_bwd']['factory'] == config['clf_fwd']['factory']:
                config['clf_bwd'].update(config['clf_fwd'].to_dict())
            elif config['clf_bwd']['factory'] == CNN1d:
                config['clf_bwd']['in_channels'] = input_size
