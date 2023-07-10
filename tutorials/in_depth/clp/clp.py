class CLP (self):
    def __init__(self):
        # Weights also should be in the fixed point representation if starting with some non-zero weights.
        weights_proto = weights_proto * 2 ** b_fraction

        # Novelty detection input connection weights  (all-to-one connections)
        weights_in_aval = np.ones(shape=(1, n_features))
        weights_out_aval = np.ones(shape=(1, n_protos))

        inp_pattern_fixed = np.vstack((X_train, X_test))

        # The graded spike array for input
        s_pattern_inp = np.zeros((n_features, num_steps))
        
        # Create input spike pattter that inject these inputs every n_steps_per_sample time step
        s_pattern_inp[:, 1:-1:n_steps_per_sample] = inp_pattern_fixed.T

        # The graded spike array for the user-provided label 
        s_user_label = np.zeros((1, num_steps))
        s_user_label[0, 17:n_train_samples*n_steps_per_sample:n_steps_per_sample] = y_train.T
        s_user_label[0, 18:n_train_samples*n_steps_per_sample:n_steps_per_sample] = y_train.T

        # Create a custom LearningRule. Define dw as a string
        dw = "2^-3*y1*x1*y0"
        
        learning_rule = Loihi3FLearningRule(dw=dw,
                                            x1_tau=x1_tau,
                                            t_epoch=t_epoch)

        # Processes
        data_input = RingBuffer(data=s_pattern_inp)
        
        user_label_input = RingBuffer(data=s_user_label)
        
        nvl_det = NoveltyDetector(t_wait=t_wait)

        allocator = Allocator(n_protos=n_protos)
        
        readout_layer = Readout(n_protos=n_protos)
        
        # Prototype Lif Process
        prototypes = PrototypeLIF(du=du,
                                  dv=dv,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=vth,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  sign_mode=SignMode.EXCITATORY,
                                  learning_rule=learning_rule)
        
        dense_proto = LearningDense(weights=weights_proto,
                                    learning_rule=learning_rule,
                                    name="proto_weights",
                                    num_message_bits=8,
                                    graded_spike_cfg=graded_spike_cfg)
        
        # Incoming (Dense) connection processes for NoveltyDetector
        dense_in_aval = Dense(weights=weights_in_aval)
        dense_out_aval = Dense(weights=weights_out_aval)
        
        # Allocator input Dense process to avoid deadlocking in the recurrent network
        dense_alloc_weight = Dense(weights=np.ones(shape=(1, 1)))
        
        # WTA weights and Dense proc to be put on Prototype population
        dense_wta = Dense(weights=np.ones(shape=(n_protos, n_protos)))
        
        # Monitor processes
        monitor_nvl = Monitor()
        monitor_protos = Monitor()
        monitor_v = Monitor()
        monitor_u = Monitor()
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
        
        # Novelty detector -> Allocator -> PrototypeLIF connection
        nvl_det.novelty_detected_out.connect(allocator.trigger_in)
        allocator.allocate_out.connect(prototypes.a_third_factor_in)
        
        # lif_prototypes.s_out.connect(proto_weights_dense.s_in_bap)
        prototypes.s_out_bap.connect(dense_proto.s_in_bap)
        
        # Sending y1 spike
        prototypes.s_out_y1.connect(dense_proto.s_in_y1)
        
        # Prototype Neurons' outputs connect to the inference input of the Readout process
        prototypes.s_out.connect(readout_layer.inference_in)
        user_label_input.s_out.connect(readout_layer.label_in)
        
        # Readout trigger to the Allocator
        readout_layer.trigger_alloc.connect(dense_alloc_weight.s_in)
        dense_alloc_weight.a_out.connect(allocator.trigger_in)

        # Probe novelty detector and prototypes
        monitor_nvl.probe(target=nvl_det.novelty_detected_out, num_steps=num_steps)
        monitor_error.probe(target=readout_layer.trigger_alloc, num_steps=num_steps)
        monitor_protos.probe(target=prototypes.s_out, num_steps=num_steps)
        monitor_preds.probe(target=readout_layer.user_output, num_steps=num_steps)
        monitor_v.probe(target=prototypes.v, num_steps=num_steps)
        monitor_u.probe(target=prototypes.u, num_steps=num_steps)
        # monitor_weights.probe(target=dense_proto.weights, num_steps=num_steps)