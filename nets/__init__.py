"""Models used by the frozen revision experiments."""

from .mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from .mobilenetv3_fractal import mobilenetv3_fractal
from .rebuttal_models import (
    mobilenetv3_gap, mobilenetv3_gap_matched, mobilenetv3_gap_wide,
    mobilenetv3_gap_gmp, mobilenetv3_fractal_only, mobilenetv3_gated_fractal,
    mobilenetv3_fractal_penultimate,
)

get_model_from_name = {
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenetv3_fractal": mobilenetv3_fractal,
    "mobilenetv3_gap": mobilenetv3_gap,
    "mobilenetv3_gap_matched": mobilenetv3_gap_matched,
    "mobilenetv3_gap_wide": mobilenetv3_gap_wide,
    "mobilenetv3_gap_gmp": mobilenetv3_gap_gmp,
    "mobilenetv3_fractal_only": mobilenetv3_fractal_only,
    "mobilenetv3_gated_fractal": mobilenetv3_gated_fractal,
    "mobilenetv3_fractal_penultimate": mobilenetv3_fractal_penultimate,
}
