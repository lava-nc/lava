class R_STDPLoihi(LoihiLearningRule):
    def __init__(
            self,
            learning_rate: float,
            tau_plus: float,
            tau_minus: float,
            *args,
            **kwargs
    ):
        """
        UPDATE: 

        R-STDP learning rule.

        de = STDP(pre, post) - e/ tau_e

        dw = R * de
        
        Reference: https://www.frontiersin.org/articles/10.3389/fncir.2015.00085/full

        """

        #assuming STDP(pre,post) absolves these variables
        self.learning_rate = learning_rate
        self.A_plus = str(A_plus) if A_plus > 0 else f"({str(A_plus)})"             
        self.A_minus = str(A_minus) if A_minus > 0 else f"({str(A_minus)})"
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        #string learning rule for dt : ELIGIBILITY TRACE represented as tag_1 #Implement the decay! 
        dt = f"{self.learning_rate} * {self.A_plus} * x0 * y1 +" \
             f"{self.learning_rate} * {self.A_minus} * y0 * x1 - tag_1 * tau_e"

        # String learning rule for dw
        #the weights are updated at every-timestep and the magnitude is a product of y2 (R) and de (tag_1)
        dw = " u0 * tag_1 * y2 "

        # Other learning-related parameters
        # Trace impulse values
        x1_impulse = 16
        y1_impulse = 16
        y2_impulse = 0  #Reward : R 

        # Trace decay constants
        x1_tau = tau_plus
        y1_tau = tau_minus
        y2_tau = 2 ** 32-1

        super().__init__(
            dw=dw,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            y1_impulse=y1_impulse,
            y1_tau=y1_tau,
            y2_impulse=y2_impulse,
            y2_tau=y2_tau,
            *args,
            **kwargs
        )