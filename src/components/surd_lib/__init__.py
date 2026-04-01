"""Official SURD algorithm from the ALD-Lab repository.

Method: Martínez-Sánchez, Á., Arranz, G., Lozano-Durán, A. (2024)
        'Decomposing causality into its synergistic, unique, and
        redundant components', Nature Communications, 15, 9296.
Code:   https://github.com/ALD-Lab/SURD (MIT License)

I copied the core algorithm and information theory tools from the
official repository. I stripped out matplotlib plotting and the pymp
parallel runner (not compatible with Windows). The decomposition
algorithm itself is unchanged.
"""

from src.components.surd_lib.surd_core import surd, nice_print
from src.components.surd_lib.it_tools import myhistogram, entropy, entropy_nvars
