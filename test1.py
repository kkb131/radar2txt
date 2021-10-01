# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from AAA import Wav2Vec2Model
from AAA import Radar2VecModel
from AAA.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[]
class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x

# In[]
# In[]
def ConfigLoad():
    model_args = ()
    kwargs = {}
    config = kwargs.pop("config", None)
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)

    config_path = 'facebook/wav2vec2-base-960h'
    config, model_kwargs = Wav2Vec2Config.from_pretrained(
                    config_path,
                    *model_args,
                    cache_dir=cache_dir,
                    return_unused_kwargs=True,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
    return config

# In[]
config = ConfigLoad()
# model = Davenet();
model = Radar2VecModel(config).to(device)
print(model)

model2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
# print(model2)

dummy = torch.tensor(np.random.randn(1, 1, 256, 1000), dtype=torch.float32)
dummy = dummy.to(device)

output = model(dummy).extract_features

dmy_shape = np.shape(dummy)
output_shape = np.shape(output)

print(dmy_shape)
print(output_shape)
