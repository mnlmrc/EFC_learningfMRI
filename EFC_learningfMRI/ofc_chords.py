"""
ofc_chords.py
=============

Optimal feedback control (OFC) simulation of isometric extension-flexion
chords, produced by a controller that does NOT know the hand's enslaving
matrix and has to learn an internal model of it from experience.

The model is the OFC architecture of Diedrichsen, Shadmehr & Ivry (2010,
TICS), Box 1 / Figure I. Each box of that figure is one piece of this module:

    Figure I box          equation (paper)                 here
    -------------------   ------------------------------   ----------------------------------
    Plant                 x_{t+1} = A x_t + B u_t          Plant.step               (section 3)
    Sensory information   y_t = H x_t (delayed, noisy)     Plant.sense, augment_delay (3, 5)
    Forward model         x*_{t+1} = Â x̂_t + B̂ u_t         InternalModel (Â, B̂)      (4)
                                                           + 'predict' in simulate_trial (7)
    Sensory integration   x̂_t = x*_t + K (y_t - H x*_t)    kalman_gain              (6)
                                                           + 'correct' in simulate_trial (7)
    Control policy        u_t = -L x̂_t                     policy_gain: u_t = G (x̄ - Â x̂_t) (6)
                                                           + 'act' in simulate_trial (7)
    Cost function         L = argmin [q(x) + r(u)]         cost_matrices (Q -> q, R -> r),
                                                           minimised one step ahead in
                                                           policy_gain (6)

Three things differ from the paper's scheme:

1. Holding a force. The paper writes the policy as u = -L x̂, i.e. drive the
   state to zero. Here the goal is to hold the chord's target force x̄_c, and a
   held force needs a sustained command. So the policy aims at the target:
       u_t = G (x̄_c - Â x̂_t)
   Â x̂_t is where the force would drift without a command; G turns the gap to
   the target into the cheapest command that closes it. The same rule holds
   the chord and corrects errors (no separate feedforward command).

2. Learning. OFC says what the optimal policy is for a given internal model,
   not how the internal model is acquired (paper, "Current limitations").
   Here the forward model B̂ starts wrong (it knows nothing about enslaving)
   and is learned from the forward model's prediction errors (update_model).
   Because G and K are both computed from B̂, a wrong B̂ makes the whole
   controller wrong in a specific way, and learning fixes it.

3. One-step cost. The paper's L minimises q(x) + r(u) summed over the whole
   movement. Here G only minimises the error at the next step plus the effort
   of the command sent now (policy_gain): simpler, no Riccati equation, but
   without foresight. Because holding costs effort, the held force settles
   slightly short of the target (less so for smaller R).

Variables
---------
x       finger forces (n_fingers,), the state of the plant.
u       motor commands, one signed command per finger (n_fingers,).
        Positive part goes through E_flex, negative part through E_ext.
y       sensed forces: x delayed by `delay` steps plus sensory noise.
A, B    true plant dynamics. A = force decay, B = gain * enslaving matrix.
Â, B̂    the brain's internal (forward) model of A and B. Â = A is assumed
        known; B̂ has to be learned.
x*      forward-model prediction of the next state.
x̂       state estimate (prediction combined with feedback).
K       Kalman gain: how much x̂ trusts feedback y over the prediction x*.
G       policy gain: u = G (x̄ - Â x̂). Plays the role of the paper's L.
x̄_c     target force pattern of chord c.
Q_c, R  cost weights. Q_c weights force errors (task term q(x)),
        R weights motor commands (effort term r(u)).
E_flex  enslaving matrix for flexion:  E_flex[i, j] = force on finger i per
        unit command to finger j (diagonal = 1).
E_ext   same for extension. If E_ext == E_flex the plant is linear (one B for
        every chord). If they differ, the plant is piecewise linear and the
        best linear model B̂ depends on which chord you practise.

Plant dynamics
--------------
Force is a first-order low-pass filter of the drive:
    x_{t+1} = (1 - dt/tau) x_t + (dt/tau) * drive(u_t)
i.e. A = (1 - dt/tau) I and B = (dt/tau) E. At steady state x = drive(u) = E u,
so commands are in force units, and holding a force requires a SUSTAINED
command (dF/dt = 0 during a perfect hold even though u ≠ 0).

Noise
-----
Signal-dependent motor noise in command space (σ ∝ |u|, paper Box 2) plus a
small constant term, optional additive force noise, sensory noise and a
feedback delay. The controller treats the noise as if it did not depend on
u when designing G and K, so it is an approximation to the true optimum
(Todorov 2005, ref. 19 of the paper derives the exact one). That is fine for
the qualitative questions this module is built for.

Learning
--------
After each trial the internal model B̂ is updated by a normalized LMS step on
the forward-model prediction error, using the efference copy of u and the
(noisy, delay-aligned) sensed forces:
    B̂ ← B̂ + η · (x_{t+1} - Â x_t - B̂ u_t) u_tᵀ / ⟨‖u‖²⟩
The update only changes B̂ in the subspace spanned by the commands actually
sent, so practice teaches you about the command directions you use
("excitation"). This is the mechanism by which a SINGLE true enslaving matrix
can still yield chord-specific learning.

Alternative (run_experiment(learning='perturb'), update_model_perturb):
trial and error instead of a gradient. B̂ is changed at random after each
trial and the change is kept only if it predicts that trial's sensed forces
better (smaller sensory prediction error).

Two model-bank modes:
    'global'     one B̂ shared by all chords (learning can transfer)
    'per_chord'  a separate B̂ per chord (no transfer; local model)

Module layout (all functions are independent and replaceable)
----------------------------------------------------------------
1. Chords and targets       make_chords, chords_to_targets
2. Enslaving matrices       make_enslaving
3. Plant                    Plant
4. Internal model           InternalModel, initial_model, ModelBank
5. Delay augmentation       augment_delay
6. Controller design        cost_matrices, policy_gain, kalman_gain,
                            design_controller
7. Simulation               simulate_trial
8. Learning                 align_feedback, prediction_error, update_model,
                            update_model_perturb
9. Experiment               Session, default_schedule, run_experiment
10. Analysis                measure_forces, force_derivative, window_indices,
                            derivative_patterns, rdm, mean_distance,
                            finger_force_variance, finger_change_correlation,
                            lag1_autocorr, model_error, directional_model_error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

import EFC_learningfMRI.globals as gl


# ---------------------------------------------------------------------------
# 1. Chords and targets
#    Defines the task goal: the force pattern x̄_c each chord asks for. It
#    enters the cost function as the point the task term q(x) is centred on.
# ---------------------------------------------------------------------------

def make_chords(chord_ids: Sequence[int] = gl.chordID) -> dict[int, np.ndarray]:
    """Chords of the experiment (default gl.chordID), one digit per finger.

    Returns a dictionary mapping chord IDs to arrays (n_fingers,) with entries in {1, 2, 9}
    (1 extend, 2 flex, 9 neutral), in the order of chord_ids.
    """
    return {chord_id: np.array([int(d) for d in str(chord_id)]) for chord_id in chord_ids}
    # return np.array([[int(d) for d in str(c)] for c in chord_ids])


def chords_to_targets(chords: dict[int, np.ndarray], flex_force: float = 3.0,
                      ext_force: float | None = None) -> dict[int, np.ndarray]:
    """Target force patterns x̄_c, one per chord.

    Takes the output of make_chords ({chordID: (n_fingers,) array in {1, 2, 9}})
    and returns {chordID: (n_fingers,) target forces}, same keys and order.
    Flexed fingers (2) get -flex_force, extended fingers (1) get +ext_force,
    neutral fingers (9) get 0.
    """
    ext_force = flex_force if ext_force is None else ext_force
    return {chord_id: np.where(c == 2, -flex_force, np.where(c == 1, ext_force, 0.0))
            for chord_id, c in chords.items()}


# ---------------------------------------------------------------------------
# 2. Enslaving matrices
#    The physical coupling between fingers. This is what B (the plant's input
#    matrix) is made of, and what the internal model B̂ has to learn.
# ---------------------------------------------------------------------------

def make_enslaving(n_fingers: int = 5, spread: float = 0.35, decay: float = 1.0,
                   jitter: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Enslaving matrix E with E[i, j] = force on finger i per unit command to j.

    Diagonal = 1. Off-diagonal falls off with finger distance:
        E[i, j] = spread * exp(-(|i - j| - 1) / decay)
    so neighbours get `spread`, and fingers further away get less.
    `jitter` adds multiplicative random variation to off-diagonals (a "subject").
    """
    rng = np.random.default_rng() if rng is None else rng
    i, j = np.meshgrid(np.arange(n_fingers), np.arange(n_fingers), indexing="ij")
    dist = np.abs(i - j)
    E = spread * np.exp(-(dist - 1) / decay)
    if jitter > 0:
        E = E * (1 + jitter * rng.standard_normal(E.shape))
        E = np.clip(E, 0, None)
    np.fill_diagonal(E, 1.0)
    return E


# ---------------------------------------------------------------------------
# 3. Plant (the real hand)
#    Figure I, bottom ("World"): the box "Plant  x_{t+1} = A x_t + B u_t" and
#    the arrow "Sensory information y" coming out of it.
#    The brain never reads this object directly: the controller only sees
#    sensed forces y (Plant.sense) and its own commands u (efference copy).
# ---------------------------------------------------------------------------

@dataclass
class Plant:
    E_flex: np.ndarray
    E_ext: np.ndarray | None = None      # None -> same as E_flex (linear plant)
    dt: float = 0.01                     # s per step
    tau: float = 0.06                    # force time constant (s)
    motor_sdn: float = 0.08              # signal-dependent command noise (σ = sdn·|u|)
    motor_const: float = 0.03            # constant command noise σ
    force_sd: float = 0.0                # additive force noise σ
    sensory_sd: float = 0.05             # sensory noise σ
    delay: int = 5                       # feedback delay in steps

    def __post_init__(self):
        if self.E_ext is None:
            self.E_ext = self.E_flex.copy()

    @property
    def n(self) -> int:
        """Number of fingers (size of the state x)."""
        return self.E_flex.shape[0]

    @property
    def gain(self) -> float:
        """dt / tau: fraction of the way force moves toward the drive per step."""
        return self.dt / self.tau

    @property
    def A(self) -> np.ndarray:
        """Paper's A: force decay per step, (1 - dt/tau) I."""
        return (1 - self.gain) * np.eye(self.n)

    @property
    def is_linear(self) -> bool:
        return np.allclose(self.E_flex, self.E_ext)

    def drive(self, u: np.ndarray) -> np.ndarray:
        """Steady-state force produced by command u (piecewise linear):
        positive commands act through E_flex, negative ones through E_ext."""
        return self.E_flex @ np.maximum(u, 0) + self.E_ext @ np.minimum(u, 0)

    def B_effective(self, u_ref: np.ndarray | None = None) -> np.ndarray:
        """Paper's B, the true input matrix (gain * E). Exact for a linear plant;
        for an asymmetric plant, the local B around command u_ref (columns from
        E_flex where u_ref ≥ 0, else E_ext). Used only to evaluate B̂ (analysis)."""
        if self.is_linear or u_ref is None:
            return self.gain * self.E_flex
        cols = np.where(u_ref >= 0, 1, 0)
        E = np.where(cols[None, :] == 1, self.E_flex, self.E_ext)
        return self.gain * E

    def step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Plant box of Figure I: x_{t+1} = A x_t + B (u_t + ε_t).

        ε_t is signal-dependent motor noise as in paper Box 2, Eq. V:
        σ = motor_sdn·|u| + motor_const, ε = σ · N(0, 1) per finger. It is added
        to the command before the enslaving matrix, so it spreads across fingers.
        """
        noise = (self.motor_sdn * np.abs(u) + self.motor_const) * rng.standard_normal(self.n)
        u_noisy = u + noise
        x_next = self.A @ x + self.gain * self.drive(u_noisy)
        return x_next

    def sense(self, x_delayed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sensory information y of Figure I: the (already delayed) force plus
        Gaussian sensory noise. The caller passes x_{t-d} to implement the delay."""
        return x_delayed + self.sensory_sd * rng.standard_normal(self.n)


# ---------------------------------------------------------------------------
# 4. Internal model
#    Figure I, "Forward model  x*_{t+1} = Â x̂_t + B̂ u_t". (Â, B̂) are the
#    brain's beliefs about the plant's (A, B). Everything the controller does
#    (prediction, Kalman gain, policy gain) is computed
#    from these, never from the true plant.
# ---------------------------------------------------------------------------

@dataclass
class InternalModel:
    A_hat: np.ndarray    # Â: believed force decay (n × n)
    B_hat: np.ndarray    # B̂: believed effect of commands on force (n × n); learned

    def copy(self) -> "InternalModel":
        return InternalModel(self.A_hat.copy(), self.B_hat.copy())


def initial_model(plant: Plant, enslaving_guess: np.ndarray | None = None) -> InternalModel:
    """Naive internal model at the start of learning.

    Default guess for B̂ = gain * identity: the brain believes each command only
    moves its own finger (no knowledge of enslaving). Â is assumed known (the
    brain knows how force decays)."""
    guess = np.eye(plant.n) if enslaving_guess is None else enslaving_guess
    return InternalModel(plant.A.copy(), plant.gain * guess)


class ModelBank:
    """Holds the internal model(s). mode='global' shares one B̂ across chords
    (what is learned on one chord helps all others); mode='per_chord' keeps an
    independent copy per chord (no transfer)."""

    def __init__(self, base: InternalModel, n_chords: int, mode: str = "global"):
        if mode not in ("global", "per_chord"):
            raise ValueError("mode must be 'global' or 'per_chord'")
        self.mode = mode
        if mode == "global":
            self._models = [base.copy()]
        else:
            self._models = [base.copy() for _ in range(n_chords)]

    def get(self, c: int) -> InternalModel:
        return self._models[0 if self.mode == "global" else c]

    def set(self, c: int, model: InternalModel) -> None:
        self._models[0 if self.mode == "global" else c] = model


# ---------------------------------------------------------------------------
# 5. Delay augmentation
#    Figure I says sensory information is "delayed in time" (Box 1 text). The
#    standard trick to fit a delay into the equations of Figure I is to make
#    the internal state remember the last d forces. The feedback y_t = x_{t-d}
#    is then an ordinary (non-delayed) observation y_t = H z_t of part of that
#    longer state, and the Kalman filter and feedback gain can be used unchanged.
# ---------------------------------------------------------------------------

def augment_delay(A: np.ndarray, B: np.ndarray, d: int):
    """Rewrite the forward model so that a d-step sensory delay fits Figure I.

    The problem
    -----------
    Figure I assumes the senses report the CURRENT state: y_t = H x_t. With a
    delay they report an OLD one: y_t = x_{t-d}. The Kalman filter compares
    y_t with a prediction of y_t, so it needs a prediction of x_{t-d}, but the
    forward model only predicts the current force x_t. And the controller needs
    x_t, which the senses will not report for another d steps.

    The trick
    ---------
    Make the state longer: keep the current force AND the last d forces in it,
    like a shift register / conveyor belt with d+1 slots:

        z_t = [ x_t, x_{t-1}, ..., x_{t-d} ]          (n·(d+1),)
               slot 0  slot 1        slot d

    Now the old force the senses report is simply the last slot of z_t, so
    y_t = Ha z_t is an ordinary "current-state" observation again, and the
    standard Kalman filter / feedback gain of Figure I work unchanged on z. Estimating
    z_t means: estimate where the fingers are now (slot 0) and keep a record
    of where they were, so the delayed feedback can be compared with it.

    One step of the augmented system, z_{t+1} = Aa z_t + Ba u_t, does two things:
      slot 0       x_{t+1} = A x_t + B u_t    (the forward model, as before)
      slots 1..d   every old force moves one slot down the belt
                   (slot k at t+1 = slot k-1 at t); the oldest one drops off.

    Example, d = 2 (each entry is an n × n block, I = identity, 0 = zeros):

             [ A  0  0 ]          [ B ]
        Aa = [ I  0  0 ]     Ba = [ 0 ]     Ha = [ 0  0  I ]     S = [ I  0  0 ]
             [ 0  I  0 ]          [ 0 ]

        Aa z_t + Ba u_t = [ A x_t + B u_t,  x_t,  x_{t-1} ] = z_{t+1}   ✓
        Ha z_t = x_{t-2}  (what the senses report)
        S  z_t = x_t      (what the controller needs)

    Returns
    -------
    Aa  (N, N)  augmented dynamics, N = n·(d+1)
    Ba  (N, m)  augmented input matrix: commands only act on slot 0
    Ha  (n, N)  paper's H: reads the last slot (the delayed force the senses see)
    S   (n, N)  reads slot 0 (the current force) out of z
    """
    n, m = B.shape
    N = n * (d + 1)

    # Aa: forward model in slot 0, shift-by-one-slot for the history
    Aa = np.zeros((N, N))
    Aa[:n, :n] = A                                            # slot 0: x_{t+1} = A x_t (+ B u_t via Ba)
    for k in range(1, d + 1):
        Aa[k * n:(k + 1) * n, (k - 1) * n:k * n] = np.eye(n)  # slot k ← slot k-1

    # Ba: commands only change the current force
    Ba = np.zeros((N, m))
    Ba[:n] = B

    # Ha: the senses see the oldest slot, x_{t-d}
    Ha = np.zeros((n, N))
    Ha[:, d * n:] = np.eye(n)

    # S: picks the current force x_t out of z
    S = np.zeros((n, N))
    S[:, :n] = np.eye(n)
    return Aa, Ba, Ha, S


# ---------------------------------------------------------------------------
# 6. Controller design
#    Figure I, top: "Cost function J, L = argmin(q(x) + r(u))" and "Control
#    policy". Also the Kalman gain K of "Sensory integration".
#    All of it is computed from the INTERNAL model (Â, B̂).
# ---------------------------------------------------------------------------


def policy_gain(model: InternalModel, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Gain G of the "Control policy" box: u = argmin [q(x) + r(u)], one step ahead.

    Inputs, in the paper's notation:
        model   Â, B̂  (the INTERNAL model; the true A, B are never used)
        Q       paper's q(x) as a matrix: error cost  = (x - x̄)ᵀ Q (x - x̄)
        R       paper's r(u) as a matrix: effort cost = uᵀ R u

    One-step cost
    -------------
    At every step the controller picks the command u that minimises the force
    error at the NEXT step plus the effort of that command:

        cost(u) = (x_{t+1} - x̄)ᵀ Q (x_{t+1} - x̄)  +  uᵀ R u,     x_{t+1} = Â x̂_t + B̂ u
                  ───────────────────────────────    ───────
                  error per finger at the next step  effort

    Consequences further in the future are ignored. Setting the derivative
    with respect to u to zero:

        2 B̂ᵀ Q (Â x̂_t + B̂ u - x̄) + 2 R u = 0
        (R + B̂ᵀQB̂) u = B̂ᵀQ (x̄ - Â x̂_t)
        u = G (x̄ - Â x̂_t),     G = (R + B̂ᵀQB̂)⁻¹ B̂ᵀQ

    How to read it
    --------------
    Â x̂_t        where the force would drift in one step if no command were sent.
    x̄ - Â x̂_t    the gap the command has to close.
    G            turns that gap into the cheapest command that closes it:
                 B̂ᵀQ = how much each command moves the forces toward the
                 target; (R + B̂ᵀQB̂)⁻¹ = what a unit of command costs (effort,
                 plus the error it would create if overdone).

    The same rule holds the chord and corrects errors: there is no separate
    feedforward command (paper Box 1: "there is no conceptual difference
    between feed-forward and feedback control; the control policy governs
    both"). Because holding a force costs effort, the force settles slightly
    short of x̄; for one finger the shortfall is a fraction r / (r + q·dt/tau).

    Returns G, (n_commands × n_fingers).
    """
    B = model.B_hat
    cost_per_command = R + B.T @ Q @ B          # effort + error the command itself causes
    benefit = B.T @ Q                           # how much each command moves forces toward the target
    return np.linalg.solve(cost_per_command, benefit)


def kalman_gain(Aa: np.ndarray, Ha: np.ndarray, W: np.ndarray, V: np.ndarray,
                max_iter: int = 5000, tol: float = 1e-10) -> np.ndarray:
    """Kalman gain K of the "Sensory integration" box: x̂ = x* + K (y - H x*).

    W  how unreliable the forward-model prediction x* is: motor noise makes
       the plant drift from what was predicted (process noise covariance).
    V  how unreliable the feedback y is (sensory noise covariance).
    K weights the two sources by their inverse variance (paper Box 1): noisy
    senses (large V) -> small K, trust the prediction; noisy motor output
    (large W) -> large K, trust the senses.

    K depends on how uncertain the estimate x̂ is (P, a covariance), which in
    turn depends on K. So run the filter's uncertainty bookkeeping, step by
    step, until it settles:

        P_pred = A P Aᵀ + W                  uncertainty of the prediction x*
        K      = P_pred Hᵀ (H P_pred Hᵀ + V)⁻¹   weight given to the feedback
        P      = (I - K H) P_pred            uncertainty after using feedback

    This is about uncertainty in the estimate, not about future costs, and
    the controller does not use it.
    """
    N = Aa.shape[0]
    P = W.copy()
    K = np.zeros((N, Ha.shape[0]))
    for _ in range(max_iter):
        P_pred = Aa @ P @ Aa.T + W
        K_new = P_pred @ Ha.T @ np.linalg.inv(Ha @ P_pred @ Ha.T + V)
        P = (np.eye(N) - K_new @ Ha) @ P_pred
        P = 0.5 * (P + P.T)                     # keep symmetric against round-off
        if np.max(np.abs(K_new - K)) < tol:
            return K_new
        K = K_new
    return K


@dataclass
class Controller:
    """Everything the brain needs to perform one chord (Figure I, top half)."""
    G: np.ndarray        # control policy: u = G (x̄ - Â x̂)
    A_hat: np.ndarray    # Â, used by the policy to predict the drift
    target: np.ndarray   # x̄_c, target force of the chord
    K: np.ndarray        # sensory integration: Kalman gain
    Aa: np.ndarray       # forward model Â, B̂ (augmented with the delay)
    Ba: np.ndarray
    Ha: np.ndarray       # H: which part of z the senses report
    S: np.ndarray        # reads the current force out of z


def design_controller(model: InternalModel, plant: Plant, target: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Controller:
    """Build the full OFC controller for one chord from the INTERNAL model.

    1. forward model    (Â, B̂) augmented with the sensory delay
    2. control policy   G from the cost function (Q, R), see policy_gain
    3. integration      K from the assumed motor and sensory noise
    Plant noise parameters are assumed known to the brain.
    """
    d, n = plant.delay, plant.n

    # 1. forward model, augmented so that delayed feedback is a plain observation
    Aa, Ba, Ha, S = augment_delay(model.A_hat, model.B_hat, d)
    N = Aa.shape[0]

    # 2. control policy u = G (x̄ - Â x̂). It acts on the current force only, so
    #    it needs the plain (non-augmented) Â, B̂.
    G = policy_gain(model, Q, R)

    # 3. Kalman gain K.
    #    W: how much the plant is expected to drift from the prediction each
    #    step = motor noise (Box 2) at the command used to hold the chord,
    #    mapped to force through B̂, plus force noise. The hold command is what
    #    the policy sends when the force is at the target: G (x̄ - Â x̄).
    #    Delayed copies are exact, so only a tiny W there.
    u_hold = G @ (target - model.A_hat @ target)
    sd_u = plant.motor_sdn * np.abs(u_hold) + plant.motor_const
    Wx = model.B_hat @ np.diag(sd_u ** 2) @ model.B_hat.T + (plant.force_sd ** 2 + 1e-8) * np.eye(n)
    W = 1e-6 * np.eye(N)
    W[:n, :n] += Wx
    #    V: sensory noise
    V = plant.sensory_sd ** 2 * np.eye(n)
    K = kalman_gain(Aa, Ha, W, V)

    return Controller(G, model.A_hat.copy(), np.asarray(target, float), K, Aa, Ba, Ha, S)


# ---------------------------------------------------------------------------
# 7. Simulation
#    One pass around the loop of Figure I per time step:
#    control policy -> plant -> sensory information -> forward model ->
#    sensory integration -> control policy ...
# ---------------------------------------------------------------------------

def simulate_trial(plant: Plant, ctrl: Controller, n_steps: int,
                   rng: np.random.Generator, x0: np.ndarray | None = None) -> dict:
    """Run one trial (press from x0 and hold).

    Returns dict with:
      x      (n_steps+1, n) true forces
      y      (n_steps+1, n) sensed forces (y[t] ≈ x[t-d])
      u      (n_steps, n)   commands sent (efference copy, before noise)
      x_hat  (n_steps+1, n) estimate of the current force
    """
    n, d = plant.n, plant.delay
    x0 = np.zeros(n) if x0 is None else x0
    x = np.zeros((n_steps + 1, n)); x[0] = x0
    y = np.zeros((n_steps + 1, n))
    u = np.zeros((n_steps, n))
    x_hat = np.zeros((n_steps + 1, n))

    # the brain starts with a correct estimate: forces at x0, and were at x0
    z_hat = np.tile(x0, d + 1)
    y[0] = plant.sense(x0, rng)
    x_hat[0] = ctrl.S @ z_hat
    for t in range(n_steps):

        # Control policy: u_t = G (x̄ - Â x̂_t)   (x̂_t = current force, slot 0 of ẑ)
        u[t] = ctrl.G @ (ctrl.target - ctrl.A_hat @ (ctrl.S @ z_hat))

        # Plant: x_{t+1} = A x_t + B (u_t + noise)   (true, unknown to the brain)
        x[t + 1] = plant.step(x[t], u[t], rng)

        # Sensory information: y_{t+1} = x_{t+1-d} + noise
        y[t + 1] = plant.sense(x[max(t + 1 - d, 0)], rng)

        # Forward model: x*_{t+1} = Â x̂_t + B̂ u_t   (uses the efference copy u_t)
        z_pred = ctrl.Aa @ z_hat + ctrl.Ba @ u[t]

        # Sensory integration: x̂_{t+1} = x*_{t+1} + K (y_{t+1} - H x*_{t+1})
        z_hat = z_pred + ctrl.K @ (y[t + 1] - ctrl.Ha @ z_pred)

        x_hat[t + 1] = ctrl.S @ z_hat
        
    return {"x": x, "y": y, "u": u, "x_hat": x_hat}


# ---------------------------------------------------------------------------
# 8. Learning the internal model
#    Not part of Figure I: OFC assumes the forward model is already correct.
#    Here B̂ is improved after each trial from how wrong the forward model's
#    predictions were.
# ---------------------------------------------------------------------------

def align_feedback(trial: dict, plant: Plant):
    """The data the brain can learn from after a trial: for every step t, the
    sensed force before the command (x_t), the command (u_t, efference copy)
    and the sensed force after it (x_{t+1}).

    The senses report the past: y[t] = x[t-d] + noise. Shifting y by d steps
    gives the brain's (noisy) record of the force at time t: xs[t] ≈ x[t].
    The last d commands are dropped: their effect had not been sensed yet when
    the trial ended.

    Returns X0 (x_t), X1 (x_{t+1}), U (u_t), each (T, n), one row per step.
    """
    d = plant.delay
    y, u = trial["y"], trial["u"]      # y: (n_steps+1, n) sensed,  u: (n_steps, n) sent
    xs = y[d:]
    T = min(len(xs) - 1, len(u))
    return xs[:T], xs[1:T + 1], u[:T]


def prediction_error(model: InternalModel, X0: np.ndarray, X1: np.ndarray,
                     U: np.ndarray) -> float:
    """Mean squared sensory prediction error of the forward model on a trial:
        mean_t ‖x_{t+1} - (Â x_t + B̂ u_t)‖²
    (X0, X1, U from align_feedback). 0 = the model explains the hand perfectly."""
    err = X1 - X0 @ model.A_hat.T - U @ model.B_hat.T
    return float(np.mean(np.sum(err ** 2, axis=1)))


# def update_model(model: InternalModel, trial: dict, plant: Plant, lr: float = 0.05,
#                  eps: float = 1e-6) -> InternalModel:
#     """Update the forward model B̂ after one trial, from its prediction errors.

#     What is learned from
#     --------------------
#     The SENSORY PREDICTION ERROR of the forward model, not the error in
#     reaching the chord. At every step t the brain knows the command it sent
#     (efference copy u_t) and, once the delayed feedback arrives, the force
#     before (x_t) and after (x_{t+1}) that command. It asks: "if my model were
#     right, what would x_{t+1} have been?"
#         predicted_t = Â x_t + B̂ u_t
#         e_t         = x_{t+1} - predicted_t           (n_fingers,)
#     e_t = 0 at every step means B̂ explains the hand perfectly.

#     How B̂ changes
#     -------------
#     Gradient descent on the mean squared prediction error
#         ℓ(B̂) = 1/(2T) Σ_t ‖e_t‖²,    ∂ℓ/∂B̂ = -(1/T) Σ_t e_t u_tᵀ
#     one step per trial:
#         B̂ ← B̂ + lr · mean_t(e_t u_tᵀ) / mean_t(‖u_t‖²)

#     Element by element, B̂[i, j] (believed force on finger i per unit command
#     to finger j) changes by  lr · mean(e_t[i] · u_t[j]) / mean(‖u‖²):
#       finger i pressed MORE than predicted while finger j was commanded
#         -> increase the believed influence of j on i (learn enslaving);
#       finger i pressed LESS than predicted -> decrease it;
#       finger j never commanded (u[j] = 0) -> column j does not change.
#     So the model is only corrected along command directions that were
#     actually used: practising some chords teaches mostly about those.

#     Dividing by mean ‖u‖² (normalized LMS) makes the step size independent of
#     how hard the fingers press, so lr ≈ fraction of the error fixed per trial.

#     Parameters
#     ----------
#     model   current internal model (Â, B̂); Â is kept fixed
#     trial   output of simulate_trial; uses the sensed forces 'y' and the
#             efference copy 'u' (the true forces 'x' are never used)
#     plant   only used for the sensory delay d
#     lr      learning rate
#     eps     avoids division by zero if no command was sent
#     returns a new InternalModel with the updated B̂ (the input is not modified)
#     """
#     # --- align feedback with commands (see align_feedback) -----------------
#     X0, X1, U = align_feedback(trial, plant)
#     T = len(U)

#     # --- prediction errors -------------------------------------------------
#     # Rows are time steps, so 'X0 @ Âᵀ' computes Â x_t for every t at once.
#     pred = X0 @ model.A_hat.T + U @ model.B_hat.T    # forward-model prediction of x_{t+1}
#     err = X1 - pred                                  # e_t, one row per step  (T, n)

#     # --- gradient step -----------------------------------------------------
#     # err.T @ U = Σ_t e_t u_tᵀ: entry [i, j] pairs the error on finger i with
#     # the command to finger j. Divided by T -> mean over the trial. This is the
#     # NEGATIVE gradient of ℓ, hence it is added to B̂.
#     dB = err.T @ U / T                               # (n, n)
#     norm = np.mean(np.sum(U ** 2, axis=1)) + eps     # mean ‖u_t‖² over the trial

#     new = model.copy()
#     new.B_hat = model.B_hat + lr * dB / norm
#     return new


def update_model(model: InternalModel, trial: dict, plant: Plant,
                         sigma: float = 0.02,
                         rng: np.random.Generator | None = None) -> InternalModel:
    """Trial-and-error alternative to update_model: perturb B̂, keep the
    perturbation only if it predicts the trial better.

    Same goal as update_model (a smaller sensory prediction error), but
    WITHOUT computing a gradient: the learner does not know in which direction
    to change B̂, it only tries a random change and checks whether it helped.

        1. try    B̂_try = B̂ + noise           noise ~ N(0, (sigma·dt/tau)²) per entry
        2. score  both B̂ and B̂_try on this trial's data:
                  SPE(B̂) = mean_t ‖x_{t+1} - (Â x_t + B̂ u_t)‖²   (prediction_error)
        3. keep   B̂_try if SPE(B̂_try) < SPE(B̂), otherwise keep B̂

    Both models are scored on the SAME trial (same commands, same sensed
    forces), so the comparison is not confounded by trial-to-trial noise. The
    trial itself was performed with the current B̂; the tried model only has
    to "explain" it better.

    As with update_model, only command directions used in the trial affect the
    score, so B̂ improves mostly along those directions. Learning is much
    slower than gradient descent (most random changes in 25 dimensions do not
    help); sigma sets the size of each try.

    Parameters
    ----------
    model   current internal model (Â, B̂); Â is kept fixed
    trial   output of simulate_trial (uses 'y' and 'u' only)
    plant   used for the delay d and for dt/tau
    sigma   size of the random change, in enslaving-matrix units
            (sigma = 0.02 -> each E-entry changes by about ±0.02)
    rng     random generator for the perturbation
    returns the kept InternalModel (a new object either way)
    """
    rng = np.random.default_rng() if rng is None else rng
    X0, X1, U = align_feedback(trial, plant)

    # 1. random change of every entry of B̂. B̂ = (dt/tau)·Ê, so scaling by
    #    plant.gain makes sigma a change in Ê, the believed enslaving matrix.
    tried = model.copy()
    tried.B_hat = model.B_hat + sigma * plant.gain * rng.standard_normal(model.B_hat.shape)

    # 2. score current and tried model on the same data
    spe_now = prediction_error(model, X0, X1, U)
    spe_try = prediction_error(tried, X0, X1, U)

    # 3. keep whichever predicts the trial better
    return tried if spe_try < spe_now else model.copy()


# ---------------------------------------------------------------------------
# 9. Experiment
#    Repeats design_controller -> simulate_trial -> update_model over the
#    days of the training protocol.
# ---------------------------------------------------------------------------

@dataclass
class Session:
    day: int
    chords: Sequence[int]       # indices of chords performed
    n_reps: int                 # repetitions per chord
    learn: bool = True          # update the internal model?
    record: bool = False        # store trials for analysis?
    label: str = ""


def default_schedule(n_chords: int = 8, trained: Sequence[int] = (0, 1, 2, 3),
                     n_reps_train: int = 10, n_reps_test: int = 20,
                     pretrain_days: Sequence[int] = (1, 2, 3, 4),
                     train_days: Sequence[int] = tuple(range(5, 24)),
                     test_days: Sequence[int] = (4, 10, 24),
                     learn_during_test: bool = False) -> list[Session]:
    """Schedule mirroring the chord-learning experiment:
    pretraining on all chords, then training on a subset, with test sessions
    (all chords) recorded at the end of `test_days`."""
    all_c = list(range(n_chords))
    sessions = [Session(0, all_c, n_reps_test, learn=False, record=True, label="baseline")]
    for day in range(1, max(list(train_days) + list(test_days)) + 1):
        if day in pretrain_days:
            sessions.append(Session(day, all_c, n_reps_train, learn=True, label="pretrain"))
        elif day in train_days:
            sessions.append(Session(day, list(trained), n_reps_train, learn=True, label="train"))
        if day in test_days:
            sessions.append(Session(day, all_c, n_reps_test, learn=learn_during_test,
                                    record=True, label="test"))
    return sessions


# ---------------------------------------------------------------------------
# 10. Analysis
#     Measures computed from simulated trials, as you would from real data.
# ---------------------------------------------------------------------------

def measure_forces(trial: dict, measure_sd: float = 0.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Forces as recorded by the experimenter (true force + transducer noise)."""
    x = trial["x"]
    if measure_sd <= 0:
        return x
    rng = np.random.default_rng() if rng is None else rng
    return x + measure_sd * rng.standard_normal(x.shape)


def force_derivative(F: np.ndarray, dt: float, absolute: bool = True,
                     smooth_steps: int = 0) -> np.ndarray:
    """dF/dt by finite differences (optionally moving-average smoothed first).
    Returns array of length len(F) - 1."""
    if smooth_steps > 1:
        k = np.ones(smooth_steps) / smooth_steps
        F = np.apply_along_axis(lambda s: np.convolve(s, k, mode="same"), 0, F)
    dF = np.diff(F, axis=0) / dt
    return np.abs(dF) if absolute else dF


def window_indices(dt: float, start_s: float, end_s: float) -> slice:
    """Time window in seconds -> slice of time steps."""
    return slice(int(round(start_s / dt)), int(round(end_s / dt)))


def derivative_patterns(trials_by_chord: list, dt: float, window: slice,
                        measure_sd: float = 0.0, smooth_steps: int = 0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Mean |dF/dt| per finger in `window`, averaged over trials.
    Returns (n_chords, n_fingers)."""
    out = []
    for trials in trials_by_chord:
        pats = [force_derivative(measure_forces(t, measure_sd, rng), dt,
                                 smooth_steps=smooth_steps)[window].mean(0)
                for t in trials]
        out.append(np.mean(pats, axis=0))
    return np.array(out)


def rdm(patterns: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Representational dissimilarity matrix between rows of `patterns`."""
    P = np.asarray(patterns, float)
    if metric == "euclidean":
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    if metric == "correlation":
        return 1 - np.corrcoef(P)
    raise ValueError(metric)


def mean_distance(D: np.ndarray, idx: Sequence[int]) -> float:
    """Mean of the off-diagonal distances among the chords in idx."""
    idx = np.asarray(idx)
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    return float(sub[iu].mean())


def finger_force_variance(trials_by_chord: list, window: slice) -> np.ndarray:
    """Within-trial force variance per finger in window, averaged over trials.
    Returns (n_chords, n_fingers)."""
    return np.array([np.mean([t["x"][window].var(0) for t in trials], axis=0)
                     for trials in trials_by_chord])


def finger_change_correlation(trials: list, dt: float, window: slice) -> np.ndarray:
    """Correlation matrix (n_fingers × n_fingers) of signed force changes in
    window, pooled over the given trials of ONE chord. This is the "structure
    of movement variability" of the paper: correlations between effectors
    produced by enslaving plus feedback control."""
    dF = np.concatenate([force_derivative(t["x"], dt, absolute=False)[window] for t in trials])
    return np.corrcoef(dF.T)


def lag1_autocorr(trials: list, dt: float, window: slice) -> np.ndarray:
    """Lag-1 autocorrelation of signed force changes per finger (ONE chord).
    Negative values indicate overshoot / oscillatory corrections."""
    vals = []
    for t in trials:
        dF = force_derivative(t["x"], dt, absolute=False)[window]
        dF = dF - dF.mean(0)
        num = (dF[1:] * dF[:-1]).sum(0)
        den = (dF ** 2).sum(0) + 1e-12
        vals.append(num / den)
    return np.mean(vals, axis=0)


def model_error(B_hat: np.ndarray, plant: Plant, u_ref: np.ndarray | None = None) -> float:
    """How wrong the forward model is overall: ‖B̂ - B‖ / ‖B‖ (Frobenius),
    B taken locally around u_ref."""
    B = plant.B_effective(u_ref)
    return float(np.linalg.norm(B_hat - B) / np.linalg.norm(B))


def directional_model_error(B_hat: np.ndarray, plant: Plant, u_dir: np.ndarray) -> float:
    """Relative prediction error of B̂ along one command direction u_dir,
    i.e. how well the model predicts the effect of the commands a chord uses."""
    u_dir = np.asarray(u_dir, float)
    true = plant.gain * plant.drive(u_dir)
    return float(np.linalg.norm(B_hat @ u_dir - true) / (np.linalg.norm(true) + 1e-12))
