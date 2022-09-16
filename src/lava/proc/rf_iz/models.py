from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.rf_iz.process import RF_IZ
from lava.proc.rf.models import AbstractPyRFModelFloat

@implements(proc=RF_IZ, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRF_IZModelFloat(AbstractPyRFModelFloat):

    def run_spk(self):
        self.sub_thresh_dynamics()
        s_out = self.imag >= self.vth 
        self.real = self.real * (1 - s_out)  # reset dynamics
        self.imag = s_out * (self.vth -1e-5) + (1 - s_out) * self.imag  # the 1e-5 insures we don't spike again
        self.s_out.send(s_out) 
