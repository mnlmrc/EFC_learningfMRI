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
    Control policy        u_t = -L x̂_t                     policy: u_t = G (x̄ - Â x̂_t) (6)
                                                           + 'act' in simulate_trial (7)
    Cost function         L = argmin [q(x) + r(u)]         cost_matrices (Q -> q, R -> r),
                                                           minimised one step ahead in
                                                           policy (6)

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
   of the command sent now (policy): simpler, no Riccati equation, but
   without foresight. Because holding costs effort, the held force settles
   slightly short of the target (less so for smaller R).

Variables
---------
x       finger forces (n_fingers,), the state of the plant.
u       motor commands (n_commands,). One per finger by default; more than
        the fingers for a redundant hand. Negative = flexion, positive =
        extension (same sign as the forces).
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
E       (n_fingers × n_commands): E[i, j] = force on finger i per unit
        command j, in either direction. Square with unit diagonal = enslaving
        matrix; wider = redundant commands.

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
6. Controller design        cost_matrices, policy, kalman_gain,
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

def make_enslaving(n_fingers: int = 5, spread: float = 0.35, decay: float = 1.0, jitter: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
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
    """The hand: n_fingers forces driven by n_commands signed motor commands.

    Sign convention (same as chords_to_targets): negative = flexion,
    positive = extension, for both commands u and forces x.

    E is (n_fingers × n_commands): column j is the force pattern that command j
    produces on the five fingers (negative u_j -> flexion pattern, positive
    u_j -> the same pattern in extension). With n_commands = n_fingers and a
    unit diagonal this is the usual enslaving matrix. With more commands than
    fingers (e.g. np.hstack of two matrices) the hand is redundant: many
    command patterns give the same forces, and the effort/noise cost decides.
    One matrix means enslaving is the same in flexion and extension.

    Motor noise is signal-dependent (paper Box 2), with its own constant per
    command: motor_sdn is a vector (n_commands,), the same in both directions.
    A scalar is expanded to the same value for every command.
    """
    E: np.ndarray
    dt: float = 0.01                     # s per step
    tau: float = 0.06                    # force time constant (s)
    motor_sdn: float | np.ndarray = 0.08 # (m,) σ = sdn·|u| per command; scalar -> same for all
    motor_const: float = 0.03            # constant command noise σ
    force_sd: float = 0.0                # additive force noise σ
    sensory_sd: float = 0.05             # sensory noise σ
    delay: int = 5                       # feedback delay in steps

    def __post_init__(self):
        self.E = np.asarray(self.E, float)
        # one noise constant per command: scalar -> same value for all, vector must have length m
        v = np.asarray(self.motor_sdn, float)
        if v.ndim == 0:
            v = np.full(self.m, float(v))
        v = v.ravel()
        if v.size != self.m:
            raise ValueError(f"motor_sdn must be a scalar or have one value per command ({self.m}), got {v.size}")
        self.motor_sdn = v.copy()

    @property
    def n(self) -> int:
        """Number of fingers (size of the state x)."""
        return self.E.shape[0]

    @property
    def m(self) -> int:
        """Number of motor commands (size of u)."""
        return self.E.shape[1]

    def noise_sd(self, u: np.ndarray) -> np.ndarray:
        """Standard deviation of the motor noise on each command:
        σ = c·|u| + motor_const (paper Box 2, Eq. V). (m,)"""
        return self.motor_sdn * np.abs(u) + self.motor_const

    @property
    def gain(self) -> float:
        """dt / tau: fraction of the way force moves toward the drive per step."""
        return self.dt / self.tau

    @property
    def A(self) -> np.ndarray:
        """Paper's A: force decay per step, (1 - dt/tau) I."""
        return (1 - self.gain) * np.eye(self.n)

    @property
    def B(self) -> np.ndarray:
        """Paper's B: effect of the commands on force per step, (dt/tau) E.
        Used only to evaluate B̂ (analysis); the controller never sees it."""
        return self.gain * self.E

    def drive(self, u: np.ndarray) -> np.ndarray:
        """Steady-state force produced by command u: E u."""
        return self.E @ u

    def step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Plant box of Figure I: x_{t+1} = A x_t + B (u_t + ε_t).

        ε_t is signal-dependent motor noise as in paper Box 2, Eq. V:
        σ = c·|u| + motor_const (noise_sd), ε = σ · N(0, 1) per command. It is
        added to the command before E, so it spreads across fingers.
        """
        noise = self.noise_sd(u) * rng.standard_normal(self.m)
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
    B_hat: np.ndarray    # B̂: believed effect of commands on force (n × m); learned

    def copy(self) -> "InternalModel":
        return InternalModel(self.A_hat.copy(), self.B_hat.copy())


def initial_model(plant: Plant, enslaving_guess: np.ndarray | None = None) -> InternalModel:
    """Naive internal model at the start of learning.

    Default guess for B̂ = gain * identity: the brain believes each command only
    moves its own finger (no knowledge of enslaving). Â is assumed known (the
    brain knows how force decays). With redundant commands (m > n) there is no
    obvious naive guess, so enslaving_guess (n × m) must be given."""
    if enslaving_guess is None:
        if plant.m != plant.n:
            raise ValueError("with n_commands != n_fingers, pass enslaving_guess (n × m)")
        enslaving_guess = np.eye(plant.n)
    guess = np.asarray(enslaving_guess, float)
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


def policy(model: InternalModel, Q: np.ndarray, R: np.ndarray,
                #noise_sdn: np.ndarray | None = None
                ) -> np.ndarray:
    """Gain G of the "Control policy" box: u = argmin [q(x) + r(u)], one step ahead.

    Inputs, in the paper's notation:
        model      Â, B̂  (the INTERNAL model; the true A, B are never used)
        Q          paper's q(x) as a matrix: error cost  = (x - x̄)ᵀ Q (x - x̄)
        R          paper's r(u) as a matrix: effort cost = uᵀ R u  (m × m)
        noise_sdn  optional: signal-dependent noise constant c of each command
                   (m,), i.e. Plant.motor_sdn. If given, the cost also includes
                   the extra error the command's own noise will cause (below).

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

    Redundant commands
    ------------------
    With more commands than fingers (m > n) many u give the same forces. The
    effort term then decides how the work is shared across commands (paper,
    "Distribution of work across multiple effectors"): with R = r·I it is
    spread as evenly as possible.

    Noise-aware cost (noise_sdn given)
    ----------------------------------
    Command j carries noise with sd c_j·|u_j| (Box 2), which moves the next
    force by B̂[:, j]·noise. Its expected contribution to the error cost is
        E[(B̂ ε)ᵀ Q (B̂ ε)] = Σ_j c_j² (B̂ᵀQB̂)_jj · u_j²  =  uᵀ D u,
        D = diag(c_j² (B̂ᵀQB̂)_jj)
    i.e. the paper's "u² as variability" reading of the effort term: noisy
    commands are expensive in proportion to how noisy they are. It simply
    adds D to R. With redundant commands this steers the work away from noisy
    commands; with m = n it only weakens noisy commands slightly.

    Returns G, (n_commands × n_fingers).
    """
    B = model.B_hat
    cost_per_command = R + B.T @ Q @ B          # effort + error the command itself causes
    # if noise_sdn is not None:                   # + expected error from the command's own noise
    #     cost_per_command = cost_per_command + np.diag(noise_sdn ** 2 * np.diag(B.T @ Q @ B))
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


def design_controller(model: InternalModel, plant: Plant, target: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                      # noise_aware: bool = False
                      ) -> Controller:
    """Build the full OFC controller for one chord from the INTERNAL model.

    1. forward model    (Â, B̂) augmented with the sensory delay
    2. control policy   G from the cost function (Q, R), see policy
    3. integration      K from the assumed motor and sensory noise
    Plant noise parameters are assumed known to the brain.

    R is (n_commands × n_commands). noise_aware=True adds the expected cost of
    signal-dependent noise (plant.motor_sdn) to the policy (policy).
    """
    d, n = plant.delay, plant.n

    # 1. forward model, augmented so that delayed feedback is a plain observation
    Aa, Ba, Ha, S = augment_delay(model.A_hat, model.B_hat, d)
    N = Aa.shape[0]

    # 2. control policy u = G (x̄ - Â x̂). It acts on the current force only, so
    #    it needs the plain (non-augmented) Â, B̂.
    G = policy(model, Q, R,) # noise_sdn=plant.motor_sdn if noise_aware else None)

    # 3. Kalman gain K.
    #    W: how much the plant is expected to drift from the prediction each
    #    step = motor noise (Box 2) at the command used to hold the chord,
    #    mapped to force through B̂, plus force noise. The hold command is what
    #    the policy sends when the force is at the target: G (x̄ - Â x̄).
    #    Delayed copies are exact, so only a tiny W there.
    u_hold = G @ (target - model.A_hat @ target)
    sd_u = plant.noise_sd(u_hold)
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
      u      (n_steps, m)   commands sent (efference copy, before noise)
      x_hat  (n_steps+1, n) estimate of the current force
    n = fingers, m = commands.
    """
    n, d = plant.n, plant.delay
    x0 = np.zeros(n) if x0 is None else x0
    x = np.zeros((n_steps + 1, n)); x[0] = x0
    y = np.zeros((n_steps + 1, n))
    u = np.zeros((n_steps, plant.m))
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

    Returns X0 (x_t) and X1 (x_{t+1}), each (T, n_fingers), and U (u_t),
    (T, n_commands), one row per step.
    """
    d = plant.delay
    y, u = trial["y"], trial["u"]      # y: (n_steps+1, n) sensed,  u: (n_steps, m) sent
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


