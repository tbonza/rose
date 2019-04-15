# (generated with --quick)

import logging
from typing import Any, List, Tuple, TypeVar

F: module
PIL: module
bah: SheepGAN
dhp: DiscriminatorHParams
ghp: GeneratorHParams
logger: logging.Logger
logging: module
nn: module
np: module
optim: module
plt: module
sdp: SketchDataPipeline
torch: module
use_cuda: Any

_T0 = TypeVar('_T0')
_T1 = TypeVar('_T1')

class DecoderRNN(Any):
    fc_hc: Any
    fc_params: Any
    lstm: Any
    sdp: SketchDataPipeline
    vhp: Any
    def __init__(self, vhp) -> None: ...
    def forward(self, inputs, z, hidden_cell = ...) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]: ...

class Discriminator(Sequence2SequenceVaeUtils):
    __doc__: str
    decoder: Any
    decoder_optimizer: Any
    dhp: Any
    encoder: Any
    encoder_optimizer: Any
    eta_step: Any
    mu: Any
    mu_x: Any
    mu_y: Any
    pi: Any
    q: Any
    rho_xy: Any
    sdp: SketchDataPipeline
    sigma: Any
    sigma_x: Any
    sigma_y: Any
    vhp: Any
    def __init__(self, vhp) -> None: ...
    def discstrokes(self, gen_batch) -> Any: ...
    def load(self, encoder_name, decoder_name) -> None: ...
    def save(self, epoch) -> None: ...
    def train(self, epoch) -> None: ...

class DiscriminatorHParams(SharedHParams):
    KL_min: float
    M: int
    Nz: int
    R: float
    dec_hidden_size: int
    dropout: float
    enc_hidden_size: int
    eta_min: float
    grad_clip: float
    max_seq_length: int
    temperature: float
    wKL: float

class EncoderRNN(Any):
    fc_mu: Any
    fc_sigma: Any
    lstm: Any
    sdp: SketchDataPipeline
    vhp: Any
    def __init__(self, vhp) -> None: ...
    def forward(self, inputs, batch_size, hidden_cell = ...) -> Tuple[Any, Any, Any]: ...

class Generator(Sequence2SequenceVaeUtils):
    __doc__: str
    decoder: Any
    decoder_optimizer: Any
    encoder: Any
    encoder_optimizer: Any
    eta_step: Any
    ghp: Any
    mu: Any
    mu_x: Any
    mu_y: Any
    pi: Any
    q: Any
    rho_xy: Any
    sdp: SketchDataPipeline
    sigma: Any
    sigma_x: Any
    sigma_y: Any
    vhp: Any
    def __init__(self, vhp) -> None: ...
    def genstrokes(self, gan_batch) -> Any: ...
    def load(self, encoder_name, decoder_name) -> None: ...
    def save(self, epoch) -> None: ...
    def train(self, epoch) -> None: ...

class GeneratorHParams(SharedHParams):
    KL_min: float
    M: int
    Nz: int
    R: float
    dec_hidden_size: int
    dropout: float
    enc_hidden_size: int
    eta_min: float
    grad_clip: float
    max_seq_length: int
    temperature: float
    wKL: float

class Sequence2SequenceVaeUtils(object):
    eta_step: Any
    mu: None
    mu_x: Any
    mu_y: Any
    pi: Any
    q: Any
    rho_xy: Any
    sdp: Any
    sigma: None
    sigma_x: Any
    sigma_y: Any
    vhp: Any
    def __init__(self, vhp, sdp) -> None: ...
    def bivariate_normal_pdf(self, dx, dy) -> Any: ...
    def conditional_generation(self, epoch, encoder, decoder) -> None: ...
    def gan_conditional_generation(self, encoder, decoder, gan_batch) -> Any: ...
    def kullback_leibler_loss(self) -> Any: ...
    def ler_decay(self, optimizer: _T0) -> _T0: ...
    def make_image(self, sequence, epoch, name = ...) -> None: ...
    def make_target(self, batch, lengths) -> Tuple[Any, Any, Any, Any]: ...
    def reconstruction_loss(self, mask, dx, dy, p, epoch) -> Any: ...
    def sample_bivariate_normal(self, mu_x: _T0, mu_y: _T1, sigma_x, sigma_y, rho_xy, greedy = ...) -> Tuple[Any, Any]: ...
    def sample_next_state(self) -> Tuple[Any, Any, Any, Any, Any]: ...

class SharedHParams(object):
    Nmax: int
    batch_size: int
    lr: float
    lr_decay: float
    max_seq_length: int
    min_lr: float
    test_set: str
    train_set: str
    val_set: str

class SheepGAN(object):
    __doc__: str
    dhp: Any
    ghp: Any
    num_epochs: int
    sdp: Any
    def __init__(self, num_epochs: int, ghp, dhp, sdp) -> None: ...
    def train(self) -> None: ...

class SketchDataPipeline(object):
    Nmax: Any
    __doc__: str
    data: Any
    data_location: Any
    hp: Any
    def __init__(self, hparams, data_location) -> None: ...
    def calculate_normalizing_scale_factor(self, strokes) -> Any: ...
    def gencast_tensor(self, strokes: list, lengths: _T0) -> Tuple[Any, _T0]: ...
    def get_clean_data(self) -> Tuple[list, Any]: ...
    def make_batch(self, batch_size) -> Tuple[Any, List[int]]: ...
    def make_gan_batch(self, batch_size) -> Tuple[Any, Any, List[int]]: ...
    def max_size(self, data) -> int: ...
    def normalize(self, strokes) -> list: ...
    def purify(self, strokes) -> list: ...


def enable_cloud_log(level = ...) -> None: ...
