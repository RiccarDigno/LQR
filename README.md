
# LQR
 Linear Quadratic Regulator and Iterative Learning

 Linear quadratic regulator (LQR) is one of the most popular frameworks to tackle control problems in general and Markov Decision Processes (MDPs) in particular.  The LQR case can be considered as the simplest scenario in the hard problems set of RL and it does admit elegant closed form formulae that are helpful for understanding many issues of this problem formulation. This peek is a tentative visit of the first step of the RL ladder.


 ## LQR problem formulation

 The problem framework here is MDPs in continuous state-space where $$ x_t \in \mathcal{X} \subset \mathcal{R}^n $$, $$u_t \in \mathcal{U} \in \mathcal{R}^m$$, $$w_t \in \mathcal{W} \in \mathcal{R}^N$$, which are usually referred to as states, control actions and some random disturbance at time $$t$$, respectively.
 Further, a stage-cost function is defined as $$ g : \mathcal{X} \times \mathcal{U} \times \mathcal{X} \rightarrow\mathcal{R}$$ together with a transition dynamic $$ f : \mathcal{X} \times \mathcal{U} \times \mathcal{W} \rightarrow \mathcal{X}$$ for a control action $$ u_t $$ in the state $$ x_t $$ with some disturbance $$ w_t $$, or often $$ w_{t+1} $$, since this last term is fully revealed at time $$ t+1$$.

 The goal is then to find an optimal (stationary) control policy $$ \pi :  \mathcal{X} \rightarrow \mathcal{U} $$.  A definition of a “stationary distribution”  can be found here in this post about [complex stuff with no bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/), just to refresh some concepts. The optimal policy is expected to solve:

 $$
 \min \sum_{t=0}^{\infty} \gamma^t \mathbf{E}[g(x_t , u_t , x_{t+1})] \\  \text{s.t.} \; x_{t+1} = f(x_t , u_t , w_t), x_0 \sim \mathcal{D}
 $$

 Infinite horizon LQR is then a subclass of the previous problem, where the dynamic is linear and the stage-cost is quadratic. This case is particularly nice due to some interesting noise-related properties, but Ben Recht does an extremely clear example talking about [The Linear Quadratic Regulator](https://www.argmin.net/2018/02/08/lqr/).
 In the best scenario without any disturbance ( we will later discuss how the LQR behaves with some disturbances in the control or state axes), the system becomes

 $$
 \min \mathbf{E}[\sum_{t=0}^{\infty} x_t Q x_t^{T} + u_t R u_t^{T}] \\  \text{s.t.} \; x_{t+1} = A x_t+ B u_t,\\
 u_t = \pi (x_t), \: x_0 \sim \mathcal{D}
 $$

 where $$ A \in \mathcal{R}^{n \times n}$$, $$ B \in \mathcal{R}^{n \times m}$$, $$ Q \succeq 0 $$ and $$ R \succ 0 $$.
 In such case, the optimal control policy results to be a static state feedback and value function are known to be linear and convex quadratic on state respectively

 $$
 \pi^{\star} (x) = K x, \; V^{\star}(x) = x^{T}P x
 $$

 where

 $$

 P = A^T P A + Q - A^T P B (B^T P B + R)^{-1} B^T P A, \\
 K = -(B^T P B + R)^{-1} B^T P A

 $$

 The first is known as the discrete time Algebraic Riccati Equation (DARE). In case of $$(A, B)$$-controllability and $$(Q, A)$$-detectability, the solution is unique and can be efficiently computed via the Riccati recursion or other alternatives.

 In the continous time case

 $$
 \dot{x}(t) = A x(t) + B u(t)
 $$

 the same steps would lead to continuous algebraic Riccatti equation (CARE) for a matrix P

 $$
 A^TP+PA−PBR^{-1}B^TP+Q=0, \\
 K=R^{-1}B^TP
 $$


 The LQR solution can be derived via adjoints method as well, which is now called back-propagation by the cool-kids, but I am no cool. Ben Recht did it as well talking about [The Linear Quadratic Regulator](https://www.argmin.net/2018/02/08/lqr/), but it is just too elegant not to include some short passages here.
 First, after defining the Lagrangian of the system

 $$
 \mathcal{L}(x, u, p) = \sum_{t=0}^N [x_t^T Q x_t + u_t^T R u_t - p_{t+1}{T} (x_{t+1} - A x_t - b u_t) ]
 $$

 We compute the gradients

 $$
 \nabla_{x_t}\mathcal{L} = Q x_t - p_t + A^T p_{t+1} \\
 \nabla_{u_t} \mathcal{L} = Ru_t + B^T p_{t+1} \\
 \nabla_{p_t} \mathcal{L} = -x_t + A x_{t-1} + B u_{t-1}
 $$

 We now need to find settings for all of the variables to make these gradients vanish. The simplest way to solve for the costates $$p_t$$ and control actions $$u_t$$ is to work backwards, leading to the (DARE) equation. How cool is this?

 ## iterative-LQR

 The linear dynamic can be manipulated to better interpret the role of a LQR control

 $$
 A x_t + B u_t = F_t \begin{bmatrix} x_t \\ u_t \end{bmatrix} + f_t
 $$

 where $$ F_t, f_t $$ represent the portions of the affine transformation of the dynamics. The stage-cost as well can be manipulated as

 $$
 g(x_t, u_t) = \frac{1}{2} \begin{bmatrix} x_t \\ u_t \end{bmatrix}^T G_t \begin{bmatrix} x_t \\ u_t \end{bmatrix} + \begin{bmatrix} x_t \\ u_t \end{bmatrix}^T g_t, \\
 G_t = \begin{bmatrix} G_{xx} & G_{xu} \\ G_{ux} & G_{uu} \end{bmatrix}
 $$

 it is then possible to solve for $$ u_t $$ only since it affects merely the last term

 $$
 Q(x_t, u_t) = const + g(x_t, u_t) \\
 \nabla_{u_{t}} Q(x_t, u_t) = G_{ux} x_t + G_{uu} u_t + g_{u}^T = 0
 $$

 leading to

 $$
 u_t = -G_{uu}^{-1}(G_{ux}x_t + g_u) \\
 u_t = K_t x_t + k_t
 $$

 by eliminating u_t via sostitution we get

 $$
 V(x_t) = Q(x_t, K_t x_t + k_t) = const + \frac{1}{2} x_t^T V_t x_t + v_t
 $$

 it is then possible to go on with the substitution since $$u_{t-1}$$ affects $$x_t$$ through the dynamics and so on deriving a backward recursion after some trivial initialization:

 for $$ t = T$$ to $$ 1 $$:
 1.  $$ Q_t = G_t + F_t^T V_{t+1} F_t$$
 2.  $$ q_t = c_t + F_t^TV_{t+1}f_t + F_t^TV^{t+1} $$
 3.  $$ Q(x_t, u_t) = const + \frac{1}{2} \begin{bmatrix} x_t \\ u_t \end{bmatrix}^T Q_t \begin{bmatrix} x_t \\ u_t \end{bmatrix} + \begin{bmatrix} x_t \\ u_t \end{bmatrix}^T q_t $$
 4. $$ u_t \longleftarrow argmin_{u_t} Q (x_t, u_t) = K_t x_t + k_t, \; K_t = -Q_{uu}^{-1}Q_{ux}, k_t = -Q_{uu}^{-1}q_u $$
 5. $$V_t = Q_{xx} + Q_{xu}K_t + K_t^T Q_{ux} + K_t^T Q_{uu} k_t$$
 6. $$v_t = q_x + Q_{xu}k_t + K_t^Tq_u + K_t^T Q_{uu}k_t $$
 7. $$V(x_t) = const + \frac{1}{2} x_t^T V_t x_t + x_t^Tv_t$$

 subsequently the forward recursion of the controlled system is performed

 for $$ t = 1$$ to $$ T $$:
 1. $$ u_t = K_t x_t + k_t$$
 2. $$ x_{t+1} = f(x_t, u_t) $$

 It is possible to generalize to stationary gaussian processes as well, as the one tackled by Kalman filtering, leading to the actual i-LQR formulation. In such cases

 $$
 x_{t+1} \sim p(x_{t+1} | x_t, u_t), \\
 p(x_{t+1} | x_t, u_t) = \mathcal(N)( F_t  \begin{bmatrix} x_t \\ u_t \end{bmatrix} + f_t, \; \Sigma)
 $$



 ## LQR with Structured Policy Iteration

 It is possible to induce a specific structure on the policy as well, and this might be extremely useful while talking about sparse and decentralized feedback control. From the standard LQR formulation, a regularizer is added on the policy to induce such structure. The regularized LQR problem can be then stated in the discrete time case as

 $$
 \min_K \mathbf{E}[\sum_{t=0}^{\infty} x_t Q x_t^{T} + u_t R u_t^{T}] + \lambda r(K) \\  \text{s.t.} \; x_{t+1} = A x_t+ B u_t,\\
 u_t = \pi (x_t), \: x_0 \sim \mathcal{D}
 $$

 for a nonnegative parameter $$\lambda \geq 0$$. The regularizer $$r:  \mathcal{R}^{n \times m} \rightarrow \mathcal{R} $$ is a nonnegative convex regularizer inducing the structure of the policy K.

 Different regularizers induce different types of structure on the policy K. It is possible to consider:

 1. Lasso reg.: $$ r(K) = \left\lVert K \right\lVert_1 = \sum_{i,j}  \| K_{i,j} \| $$, which leads to a sparse gain matrix K.
 2. Group lasso reg.: $$ r(K) = \left\lVert K \right\lVert_{\mathcal{G}, 2} = \sum_{g \in \mathcal{G}} \left\lVert K_{g} \right\lVert_2 $$, which induces a block-sparse gain matrix K.
 3. Nuclear-form reg.: $$ r(K) = \left\lVert K \right\lVert_{\star} = \sum_{i} \sigma_i(K) $$, being $$\sigma_i$$ the $$i-th$$ largest singular value of K. This last induces a low-rank gain matrix K.

 The same families of metrics can be extended to track reference policies with $$K^{ref} \in \mathcal{R}^{n \times m}$$.

 The whole stage-cost function is then formulated as

 $$
 F(K) := f(K) + \lambda r(K)
 $$

 Here $$ f(K) = \mathbf{Tr}(\Sigma_0 P)$$ where $$\Sigma_0 = \mathbf{E}[x_0 x_0^T]$$ is the covariance matrix of the initial state and $$ P$$ is the quadratic value matrix.

 It is possible to solve the whole problem using the so called Structured Policy Iteration (S-PI) algorithm by iteratively evaluating the policy via the $$ P^i $$ and $$\Sigma^i$$ matrices satisfying the Lyapunov equations:

 $$
 (A + B K^i )^T P^i(A + BK^i) - P^i + Q + (K^i)^TRK^i = 0 \\
 (A + BK^i)\Sigma^i (A + BK^i)^T - \Sigma^i + \Sigma_0 = 0
 $$

 and by defining a policy improvement accordingly using the gradient of K

 $$
 \nabla_K f(K^i) = 2 (( R + B^T P^i B)K^i + B^T P^i A)\Sigma^i
 $$

 and plugging it in a proximal gradient method adapted to the regularizer.

 The major cost incurs when solving the Lyapunov equations in the policy (and covariance) evaluation step.  Yet, a sequence of Lyapunov equations can be solved with less cost by using iterative methods with adopting the previous one (warm-start) or approximated one. For the whole algorithm structure and the model-free implementation via smoothing procedures you can check this recent article about [Structured Policy Iteration for Linear Quadratic Regulator](https://proceedings.icml.cc/static/paper_files/icml/2020/3607-Paper.pdf).

 ## LQR robustness

 Now, what if you want to solve some optimal control problem and you spend decades modelling all the portions of your problem, the dynamics, the control effects, the objectives and constraints. Robustness aims to quantify the effects of oversight on your systems behavior. What if some parts of your model was not accurate, or you did set too restrictive constraints. What are the consequences?

 For example, consider a continuous time system

 $$
 \dot{x}_t = A x_t + B_\star u_t
 $$

 where $B_\star$ represents a mismatch between the modeled dynamics and reality, in terms of the effect of the control action. It would be the case for the control of a motor in an electric bike, when the gear ratio is not well known.

 An attractive feature of LQR is that we can quantify precisely how much we will move far from the original (DARE) or (CARE) solution. In fact, a Lyapunov function can be used to guarantee stability of the system.
 A Lyapunov function is a function V that maps the states to real numbers, is nonnegative everywhere, is equal to $$0$$ only when $$x = 0$$ (or any variable transformation of it) and whose value is strictly decreasing along any trajectory of a dynamical system, guaranteeing for a Lyapunov-style stability, but you can check on the Wiki for [Lyapunov function]( https://en.wikipedia.org/wiki/Lyapunov_function).
 In equations:

 $$
 V(x) \geq 0, \; V(x) = 0 \iff x=0, \; \dot{V} < 0
 $$

 The last equation is particularly significant for the stability portion, because it actually contains the dynamics itself ( you can check it easily with some chain rule).
 If you have a Lyapunov function, then all trajectories must converge to $$x = 0$$ or whatever unique point where $$V=0$$: if you are at any nonzero state, the value of V will decrease. If you are at $$ 0 $$, then you will be at a global minimum of V and hence you will not move to any other trajectory. (Attention: this does not mean any other state in general! If you are not convinced about it, please check this [Cool video lecture about Lyapunov control](https://www.youtube.com/watch?v=oxzvAK394t0), which has some additional lemma from Mukherjee and Chen).

 Now, let P be the solution of the CARE and let’s set that $$ V(x) = x^T P x$$ is a Lyapunov function. This choice is usually the most straightforward in these cases: since P is positive definite, we have $$ V(x) \geq 0 $$ and $$ V(x) = 0$$  if and only if $$ x = 0$$ . To prove that the derivative of the Lyapunov function is negative, we can first compute the derivative

 $$
 \frac{d}{dt} x_t^T P x_t = x_t^T { (A - B_\star K )^T P + P (A - B_\star K)} x_t
 $$

 Note that it is sufficient to show that $$ (A - B_\star K )^T P + P (A - B_\star K) $$ is a negative definite matrix. To prove this,  a bit of algebra can lead to some sufficient conditions, using the definition of K and the fact that P solves the CARE

 $$
 (A - B_\star K )^T P + P (A - B_\star K) = P(B - B_\star) R^{-1}(B - B_\star )^T P - PB_\star R^{-1} B_\star^T P - Q
 $$

 With this final expression, a huge number of conditions under which we get “robustness for free” pops out.

 Firstly, for the base case where $$B = B_\star$$, since R is positive definite and Q is positive semidefinite, the entire expression is negative definite, and hence we have proven the system is stable ( as we hopefully expected)

 Secondly, there is a famous result that LQR has “large gain margins.”
 The gain margin of a control system is an interval $$(\alpha_0, \alpha_1)$$ such that for all $$\alpha$$ in this interval, our control system is stable with the controller $$\alpha K$$. Another way of thinking about the gain margin is to assume that $$B_\star = \alpha B$$, and to find the largest interval such that the system $$(A, B_\star)$$ is stabilized by a control policy K. For LQR, there are very large margins: if we plug in the identity $$ B_\star=\alpha B $$, we find that $$ x^T P x $$ is a Lyapunov function provided that $$\alpha \in (\frac{1}{2},\infty)$$. LQR control turns out to be robust to a wide range of perturbations to the matrix B. Intuitively, it makes sense that having a stronger control action our policy will still drive the system to zero. This is the range of  $$\alpha \in [1,\infty)$$. The other part of the interval is perhaps more interesting: for the LQR problem, even if we only have half of the control we had planned for, we still will successfully stabilize our system from any initial condition.

 In discrete time, unfortunately, the expressions are not as nice. Also, note that you cannot expect infinite gain margins in discrete time. In continuous time a differential equation $$\dot{x}_t = Mx_t$$  is stable if all of the eigenvalues of M have negative real parts. In discrete time, you need all of the eigenvalues to have magnitude less than 1. For almost any random set triple $$(A,B,K)$$, $$ A− \alpha BK$$ is going to have large eigenvalues for $$\alpha$$ large enough.

 Most generally, the control system will be stable provided that

 $$
 (B - B_\star)R^{-1}(B-B_\star)^T \prec B_\star R^{-1}B_\star^T
 $$

 The LQR gain margins fall out naturally from this expression when we assume $$B_\star = BM$$. However, we can guarantee much more general robustness using this inequality. For example, if we assume that $$B_\star = BM$$ for some square matrix M, then K stabilizes the pair $$(A,B_\star)$$ if all of the eigenvalues of $$ M+M^T$$ are greater than 1.

 Perhaps more in line with what we do in machine learning, suppose we are able to collect a lot of data, do some uncertainty quantification, and guarantee a bound $$\vert B−B_\star \vert_2  < \epsilon $$. Then as long as

 $$
 \epsilon \leq \lambda_\min (R) \lambda_{\min} (P^{-1}QP^{−1})
 $$

 we will be guaranteed stable execution. This expression depends on the matrices P, Q, and R, so it has a different flavor of the infinite gain margin conditions which held irrespective of the dynamics or the cost. Moreover, if P has large eigenvalues, then we are only able to guarantee safe execution for small perturbations to B. This foreshadows issues I’ll dive into in later posts. I want to flag here that these calculations reveal some fragilities of LQR: While the controller is always robust to perturbations along the direction of the matrix B, you can construct examples where the system is highly sensitive to tiny perturbations orthogonal to B.


 ## References

 [1] Kun J. [complex stuff with no bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)

 [2] Recht B.  [The Linear Quadratic Regulator](https://www.argmin.net/2018/02/08/lqr/)

 [3] Park Y. Rossi R.A. et alia [Structured Policy Iteration for Linear Quadratic Regulator](https://proceedings.icml.cc/static/paper_files/icml/2020/3607-Paper.pdf)

 [4] [Lyapunov function]( https://en.wikipedia.org/wiki/Lyapunov_function) Wikipedia page

 [5] [Cool video lecture about Lyapunov control](https://www.youtube.com/watch?v=oxzvAK394t0) by Bob Trenwith on YouTube
