# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC

import numpy as np

from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.magma.core.learning.learning_rule import Loihi3FLearningRule
from lava.proc.clp.novelty_detector.process import NoveltyDetector
from lava.proc.clp.nsm.process import Allocator, Readout
from lava.proc.clp.prototype_lif.process import PrototypeLIF
from lava.proc.dense.process import LearningDense, Dense, DelayDense
from lava.proc.io.source import RingBuffer
from lava.proc.monitor.process import Monitor
from lava.utils.weightutils import SignMode


class CLP(ABC):
    """
    CLP class that encapsulates current Lava implementation of the CLP
    algorithm.

    Parameters
        ----------
        supervised : boolean
            If True, CLP is configured to learn in supervised manner, i.e. the
            mismatch between its prediction and the true label provided by user
            will trigger allocation of a new prototype neuron
        n_protos : int
            Number of Prototype LIF neurons that this process need to read from.
        n_features: int
            The length of the feature vector which is the input to CLP.
        n_steps_per_sample : int
            How many time steps we allocate for each input pattern processing
             before injecting the next pattern.
        b_fraction : int
            Number of bits for fractional part in the fixed point representation
        du : int
            This is the du parameter of the PrototypeLIF neurons of the CLP
        dv : int
            This is the dv parameter of the PrototypeLIF neurons of the CLP
        vth: int
            This is the vth parameter of the PrototypeLIF neurons of the CLP
        t_wait : int
            The amount of time the process will wait after receiving
            signal about input injection to the system before sending out
            novelty detection signal. If in this time window the system (the
            Prototype neurons) generates an output, then the process will be
            reset and NO novelty detection signal will be sent out.

    """

    def __init__(self,
                 supervised=True,
                 learn_novels=True,
                 weights_proto=None,
                 proto_labels=None,
                 n_protos=2,
                 n_features=2,
                 n_steps_per_sample=10,
                 b_fraction=7,
                 du=4095,
                 dv=4095,
                 vth=1,
                 t_wait=10,
                 debug=False):

        self.supervised = supervised
        self.learn_novels = learn_novels
        self.n_protos = n_protos
        self.n_features = n_features
        self.n_steps_per_sample = n_steps_per_sample
        self.du = du
        self.dv = dv
        self.vth = vth
        self.t_wait = t_wait
        self.weights_proto = weights_proto
        self.debug = debug

        self.num_steps = 0

        self.prototypes = None
        self.monitors = None
        self.nvl_det = None
        self.readout_layer = None
        self.dense_proto = None
        self.data_input = None
        self.n_alloc_protos = 0
        self.s_pattern_inp = None
        self.s_user_label = None

        if self.weights_proto is None:
            # No pattern is stored yet. None of the prototypes are allocated
            self.weights_proto = np.zeros(shape=(n_protos, n_features),
                                          dtype=np.int32)

        self.n_alloc_protos = np.count_nonzero(
            np.count_nonzero(self.weights_proto, axis=1))

        self.proto_labels = proto_labels

        # Create a custom LearningRule. Define dw as a string
        dw = "2^-3*y1*x1*y0"

        # Learning rule parameters
        # Pre-trace decay constant. The high values means no decay
        x1_tau = 65535
        t_epoch = 1  # Epoch length

        self.learning_rule = Loihi3FLearningRule(dw=dw,
                                                 x1_tau=x1_tau,
                                                 t_epoch=t_epoch)

    def generate_input_spikes(self, x, y=None):

        n_total_samples = x.shape[0]
        # Number of time steps that the processes will run
        self.num_steps = n_total_samples * self.n_steps_per_sample

        # The graded spike array for input
        self.s_pattern_inp = np.zeros((self.n_features, self.num_steps))

        # Create input spike pattern that inject these inputs every
        # n_steps_per_sample time step
        self.s_pattern_inp[:, 1:-1:self.n_steps_per_sample] = x.T

        # The graded spike array for the user-provided label
        self.s_user_label = np.zeros((1, self.num_steps))
        if y is not None:
            n_train_samples = y.shape[0]
            train_time = n_train_samples * self.n_steps_per_sample
            self.s_user_label[
                0,
                self.t_wait + 5:train_time:self.n_steps_per_sample] = y.T
            self.s_user_label[
                0,
                self.t_wait + 7:train_time:self.n_steps_per_sample] = y.T

        return self.s_pattern_inp, self.s_user_label

    def setup_procs_and_conns(self):
        if self.prototypes is not None:
            self.proto_labels = self.readout_layer.proto_labels.get()
            self.weights_proto = self.dense_proto.weights.get()
            self.n_alloc_protos = np.count_nonzero(
                np.count_nonzero(self.weights_proto, axis=1))
        # Config for writing graded payload of the input spike to x1-trace
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        # Novelty detection input connection weights  (all-to-one connections)
        weights_in_aval = np.ones(shape=(1, self.n_features))
        weights_out_aval = np.ones(shape=(1, self.n_protos))

        # Processes
        data_input = RingBuffer(data=self.s_pattern_inp)

        user_label_input = RingBuffer(data=self.s_user_label)

        nvl_det = NoveltyDetector(t_wait=self.t_wait)

        allocator = Allocator(n_protos=self.n_protos,
                              next_alloc_id=self.n_alloc_protos + 1)

        readout_layer = Readout(n_protos=self.n_protos,
                                proto_labels=self.proto_labels)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=self.du,
                                  dv=self.dv,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=self.vth,
                                  shape=(self.n_protos,),
                                  name='lif_prototypes',
                                  sign_mode=SignMode.EXCITATORY,
                                  learning_rule=self.learning_rule)

        dense_proto = LearningDense(weights=self.weights_proto,
                                    learning_rule=self.learning_rule,
                                    name="proto_weights",
                                    num_message_bits=8,
                                    graded_spike_cfg=graded_spike_cfg)

        # Incoming (Dense) connection processes for NoveltyDetector
        dense_in_aval = Dense(weights=weights_in_aval)
        dense_out_aval = Dense(weights=weights_out_aval)

        # NoveltyDetector --> Allocator connection
        dense_nvl_alloc = Dense(weights=np.ones(shape=(1, 1)))

        # Allocator --> 3rd-factor channel of the PrototypeLIF
        dense_alloc_3rd_factor = Dense(
            weights=np.ones(shape=(self.n_protos, 1)),
            num_message_bits=8)

        # Readout --> Allocator connection
        dense_readout_alloc = Dense(weights=np.ones(shape=(1, 1)))

        # WTA weights and Dense proc to be put on Prototype population
        dense_wta = Dense(weights=np.ones(shape=(self.n_protos, self.n_protos)))

        # Default reset for PrototypeLIF population some time after input
        dense_reset = DelayDense(
            weights=np.ones(shape=(self.n_protos, self.n_features)),
            delays=self.n_steps_per_sample - 2)

        # Monitor processes
        monitor_nvl = Monitor()
        monitor_protos = Monitor()
        monitor_preds = Monitor()
        monitor_error = Monitor()

        # Connections

        # Data input -> PrototypeLIF connection
        data_input.s_out.connect(dense_proto.s_in)
        dense_proto.a_out.connect(prototypes.a_in)

        # Data input -> NoveltyDetector connection for input_available signal
        data_input.s_out.connect(dense_in_aval.s_in)
        dense_in_aval.a_out.connect(nvl_det.input_aval_in)

        # PrototypeLIF -> NoveltyDetector connection for output_available signal
        prototypes.s_out.connect(dense_out_aval.s_in)
        dense_out_aval.a_out.connect(nvl_det.output_aval_in)

        # WTA of prototypes
        prototypes.s_out.connect(dense_wta.s_in)
        dense_wta.a_out.connect(prototypes.reset_in)

        # Default delayed reset of PrototypeLIF after input
        data_input.s_out.connect(dense_reset.s_in)
        dense_reset.a_out.connect(prototypes.reset_in)

        # If we want novelty detection to trigger allocation
        if self.learn_novels:
            # Novelty detector -> Dense -> Allocator connection
            nvl_det.novelty_detected_out.connect(dense_nvl_alloc.s_in)
            dense_nvl_alloc.a_out.connect(allocator.trigger_in)

        # Allocator  -> Dense -> PrototypeLIF connection
        allocator.allocate_out.connect(dense_alloc_3rd_factor.s_in)
        dense_alloc_3rd_factor.a_out.connect(prototypes.a_third_factor_in)

        prototypes.s_out_bap.connect(dense_proto.s_in_bap)

        # Sending y1 spike
        prototypes.s_out_y1.connect(dense_proto.s_in_y1)

        # Prototype Neurons' outputs connect to the inference input of  the
        # Readout process
        prototypes.s_out.connect(readout_layer.inference_in)
        user_label_input.s_out.connect(readout_layer.label_in)

        # If we want to do supervised learning, we can connect Readout
        # layer's error signal to the allocator
        if self.supervised:
            # Readout trigger to the Allocator
            readout_layer.trigger_alloc.connect(dense_readout_alloc.s_in)
            dense_readout_alloc.a_out.connect(allocator.trigger_in)

        # Probe novelty detector and prototypes
        monitor_nvl.probe(target=nvl_det.novelty_detected_out,
                          num_steps=self.num_steps)
        monitor_error.probe(target=readout_layer.trigger_alloc,
                            num_steps=self.num_steps)
        monitor_protos.probe(target=prototypes.s_out, num_steps=self.num_steps)
        monitor_preds.probe(target=readout_layer.user_output,
                            num_steps=self.num_steps)

        self.prototypes = prototypes
        self.monitors = [monitor_nvl, monitor_error, monitor_protos,
                         monitor_preds]
        self.nvl_det = nvl_det
        self.readout_layer = readout_layer
        self.dense_proto = dense_proto
        self.data_input = data_input

        if self.debug:
            monitor_u = Monitor()
            monitor_u.probe(target=prototypes.u,
                            num_steps=self.num_steps)
            self.monitors.append(monitor_u)

        return self

    def get_results(self):
        monitor_nvl, monitor_error, monitor_protos, monitor_preds = \
            self.monitors[:4]

        novelty_spikes, proto_spikes, error_spikes, preds, currs = [None] * 5
        # Get results
        novelty_spikes = monitor_nvl.get_data()
        novelty_spikes = novelty_spikes[self.nvl_det.name][
            self.nvl_det.novelty_detected_out.name]

        proto_spikes = monitor_protos.get_data()
        proto_spikes = proto_spikes[self.prototypes.name][
            self.prototypes.s_out.name]

        preds = monitor_preds.get_data()[self.readout_layer.name][
            self.readout_layer.user_output.name]

        error_spikes = monitor_error.get_data()[self.readout_layer.name][
            self.readout_layer.trigger_alloc.name]

        if self.debug:
            monitor_u = self.monitors[4]
            currs = monitor_u.get_data()[self.prototypes.name][
                self.prototypes.u.name]

        return novelty_spikes, proto_spikes, error_spikes, preds, currs
