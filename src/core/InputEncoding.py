"""
How an input vector is put on the unsigned 16-bit input drivers of the crossbar tiles.

The root encodes the whole vector once (`encode`) and later reverses the output with Root.addCorrectionY, which
needs the offset and the scale the encoding used. Each tile applies its chunk of the encoded vector (`apply`).
Select the encoding with `inputEncoding` in exp_params, or with the environment variable INPUT_ENCODING.
"""
import numpy as np


class MinMaxInput:
    """x_s = (x - min x) / range(x), in [0, 1], one pass per MVM (the original MELISO+ encoding).

    Every cell is driven by the offset, and the reversal adds x_min * (row sums) digitally, so the product with a
    constant vector is computed exactly and the hardware error depends on min(x).
    """

    def encode(self, x):
        offset = float(x.min())
        scale = float(np.ptp(x))
        if scale == 0:
            scale = 1.0                 # constant input: scales to zeros, the reversal is still exact
        return (x - offset) / scale, offset, scale

    def apply(self, tile, x_chunk):
        return tile.localMatVec(x_chunk)


class SignedInput:
    """x_s = x / max|x|, in [-1, 1], applied as two unsigned passes x_s = x+ - x-.

    Each half is scaled to its own peak so that it uses the full input range; a zero entry drives nothing, and the
    second pass (with its read energy and latency) only happens when the chunk has entries of both signs.
    """

    def encode(self, x):
        scale = float(np.max(np.abs(x)))
        if scale == 0:
            scale = 1.0                 # zero input: stays zero
        return x / scale, 0.0, scale

    def apply(self, tile, x_chunk):
        positive = np.clip(x_chunk, 0.0, None)
        negative = np.clip(-x_chunk, 0.0, None)
        if not negative.any():
            return self.unsignedPass(tile, positive)
        if not positive.any():
            return -self.unsignedPass(tile, negative)
        return self.unsignedPass(tile, positive) - self.unsignedPass(tile, negative)

    def unsignedPass(self, tile, half):
        peak = float(half.max())
        if peak == 0:
            return tile.localMatVec(half)
        return peak * tile.localMatVec(half / peak)


INPUT_ENCODINGS = {"minmax": MinMaxInput, "signed": SignedInput}


def inputEncodingNamed(name):
    if name not in INPUT_ENCODINGS:
        raise Exception(f"ExperimentConfigFileError: inputEncoding must be one of {sorted(INPUT_ENCODINGS)}, got {name!r}")
    return INPUT_ENCODINGS[name]()
