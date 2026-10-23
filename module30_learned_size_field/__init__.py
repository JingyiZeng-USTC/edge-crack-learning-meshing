"""Final absolute size-field model (single non-branching edge crack)."""
__all__ = ['SingleCrackLocalSizeField', 'DirectNormalizedLogSizeMLP']


def __getattr__(name):
    if name == 'SingleCrackLocalSizeField':
        from .single_crack_local_field import SingleCrackLocalSizeField
        return SingleCrackLocalSizeField
    if name == 'DirectNormalizedLogSizeMLP':
        from .model import DirectNormalizedLogSizeMLP
        return DirectNormalizedLogSizeMLP
    raise AttributeError(name)
