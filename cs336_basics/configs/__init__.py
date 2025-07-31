from .gpt2_small import _C as gpt2_small


config_factory = {
    'gpt2_small': gpt2_small,

}


def build_config(mode):

    assert mode in config_factory.keys(), 'unknown config'

    return config_factory[mode]