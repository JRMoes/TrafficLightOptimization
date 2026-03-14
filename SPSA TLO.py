import random
import math
from collections import deque
import heapq
import numpy as np
from scipy.stats import t
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter


# Problem definition 

streets = ["Ba1", "Ba2", "Be1", "Be2"]  # Bakeri1, Bakeri2, Besat1, Besat2
lanes = [1, 2, 3]
left_straight_lanes = [1, 2]
right_lane = 3
EPS = 1e-6

WARMUP = 1200.0  # 20 minutes
HORIZON_DEFAULT = 3600.0  # one hour


# theta representation conversions

# Stable order for parameter vectors
APPROACHES = ["Ba1", "Ba2", "Be1", "Be2"]


def theta_to_dict(theta):

    if isinstance(theta, dict):
        return {a: float(theta[a]) for a in APPROACHES}
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 0:
        return {a: float(arr.item()) for a in APPROACHES}
    if arr.size != len(APPROACHES):
        raise ValueError(f"theta vector must have length {len(APPROACHES)}")
    return {a: float(arr[i]) for i, a in enumerate(APPROACHES)}


def theta_to_vec(theta_dict):
    """dict approach->float to np.array in APPROACHES order."""
    return np.array([theta_dict[a] for a in APPROACHES], dtype=float)



# Table 1: distribution functions fitted to empirical data


IAT_beta = {
    "Ba1": {"shift": -0.5, "scale": 22.0, "alpha": 0.971, "beta": 2.04},
    "Ba2": {"shift": -0.5, "scale": 41.0, "alpha": 0.968, "beta": 3.44},
    "Be1": {"shift": -0.5, "scale": 23.0, "alpha": 0.634, "beta": 1.61},
    "Be2": {"shift": -0.5, "scale": 24.0, "alpha": 0.963, "beta": 1.99},
}

crossing_time = {
    "Ba1": {"dist": "poisson", "lambda": 8.21},
    "Ba2": {"dist": "normal", "mean": 5.66, "sd": 2.08},
    "Be1": {"dist": "shift_lognormal", "shift": 2.5, "mean": 7.65, "sd": 6.42},
    "Be2": {"dist": "shift_erlang", "shift": 2.5, "mean": 2.82, "k": 3},
}


#Table 2: estimated average time to move one row (secs)


row_move_time = {
    "Ba1": 2.675,
    "Ba2": 2.4666,
    "Be1": 2.6818,
    "Be2": 2.3243,
}


# Figure 3: turning probabilities 

turn_prob = {
    "Ba1": {"left": 0.40, "straight": 0.50},  # right = 0.10
    "Ba2": {"left": 0.33, "straight": 0.57},  # right = 0.10
    "Be1": {"left": 0.20, "straight": 0.70},  # right = 0.10
    "Be2": {"left": 0.23, "straight": 0.67},  # right = 0.10
}

WARMUP = 1200.0  #
HORIZON_DEFAULT = 3600

#Fixed all-red 
USE_ALL_RED = True
ALL_RED_TIME_FIXED = 5

#start-up lost time (reaction + initial acceleration) 
EPS = 1e-6
USE_STARTUP_LOST = False
STARTUP_LOST_TIME = 1.5

#Lane 3 diversion rule for straight vehicles
ALLOW_STRAIGHT_TO_LANE3 = True
P_STRAIGHT_TO_LANE3 = 0.10

# Sampling 

def sample_iat(rng, approach):
    p = IAT_beta[approach]
    x = p["shift"] + p["scale"] * rng.betavariate(p["alpha"], p["beta"])
    return x if x > 0 else 0.0


def sample_turn(rng, approach):
    pL = turn_prob[approach]["left"]
    pS = turn_prob[approach]["straight"]
    r = rng.random()
    if r < pL:
        return "left"
    if r < pL + pS:
        return "straight"
    return "right"


def sample_crossing_time(rng, approach):
    spec = crossing_time[approach]
    dist = spec["dist"]

    if dist == "poisson":
        lam = spec["lambda"]
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return max(1.0, float(k - 1))

    if dist == "normal":
        mean = spec["mean"]
        sd = spec["sd"]
        x = rng.gauss(mean, sd)
        while x <= 0:
            x = rng.gauss(mean, sd)
        return x

    if dist == "shift_lognormal":
        shift = spec["shift"]
        mean = spec["mean"]
        sd = spec["sd"]
        sigma2 = math.log(1.0 + (sd * sd) / (mean * mean))
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - 0.5 * sigma2
        return shift + rng.lognormvariate(mu, sigma)

    if dist == "shift_erlang":
        shift = spec["shift"]
        mean = spec["mean"]
        k = int(spec["k"])
        rate = k / mean
        s = 0.0
        for _ in range(k):
            s += rng.expovariate(rate)
        return shift + s

    raise ValueError("Unknown crossing distribution")



# Lane choice 

def choose_lane_for_left_or_straight(turn, q1_len, q2_len):
    if q1_len < q2_len:
        return 1
    if q2_len < q1_len:
        return 2
    return 1 if turn == "left" else 2



# Event-driven simulation object with advance_to(t)

class TrafficSim:
    def __init__(self, seed=1, horizon=HORIZON_DEFAULT, warmup=WARMUP, theta=15.0, phi_min=10.0, all_red_duration=10.0):
        self.horizon = float(horizon)

        # RNG stream 
        self.rng = random.Random(seed)
        self.all_red_duration = float(all_red_duration)
        # all-red phase state
        self.all_red = False
        self.all_red_until = None
        self.pending_green = None

        self.stopline = {s: {lane: None for lane in lanes} for s in streets}
        self.behind = {s: {lane: deque() for lane in lanes} for s in streets}

        self.vehicles = {}
        self.next_vid = 0

        self.ev = []
        self.seq = 0
        self.now = 0.0

        # release log for animation (t_release, street, lane, turn)
        self.release_log = deque()

        # schedule first arrival for each street
        for s in streets:
            t0 = sample_iat(self.rng, s)
            self._push(t0, "arrival", s)

        # SOTL parameters (start with one theta for all approaches)
        self.theta = theta_to_dict(theta)  
        self.phi_min = phi_min  

        # SOTL dynamic signal state
        self.current_green = "Ba1"  # initial green approach 
        self.green_start_time = 0.0  # when current green began

        # integral counters κ_i (car-seconds), one per approach
        self.kappa = {s: 0.0 for s in streets}

        # for event-driven integration
        self.last_kappa_update = 0.0

        # for scheduling the next SOTL threshold-check event
        self._sotl_token = 0
        self._next_sotl_time = None
        self._schedule_next_sotl_check(0.0)

        # Start-up lost time state (per approach, per lane)
        self.startup_needed = {s: {1: False, 2: False, 3: False} for s in streets}
        self.startup_pending = {s: {1: False, 2: False, 3: False} for s in streets}

    def _arm_startup_on_green_begin(self, approach, t=None):

        if t is None:
            t = self.now

        if not USE_STARTUP_LOST:
            return

        s = approach

        lane1_has_queue = (self.stopline[s][1] is not None) or (len(self.behind[s][1]) > 0)
        lane2_has_queue = (self.stopline[s][2] is not None) or (len(self.behind[s][2]) > 0)

        self.startup_needed[s][1] = lane1_has_queue
        self.startup_needed[s][2] = lane2_has_queue
        self.startup_pending[s][1] = False
        self.startup_pending[s][2] = False

        vid3 = self.stopline[s][3]
        if vid3 is not None and self.vehicles[vid3]["turn"] != "right":
            self.startup_needed[s][3] = True
            self.startup_pending[s][3] = False
        else:
            self.startup_needed[s][3] = False
            self.startup_pending[s][3] = False

    def _handle_delayed_release(self, t, street, lane, vid):
        # Only release if the same vehicle is still at the stop line
        if self.stopline[street][lane] != vid:
            self.startup_pending[street][lane] = False
            return

        v = self.vehicles[vid]

        # If this vehicle requires green, ensure green still holds
        requires_green = (lane in (1, 2)) or (lane == 3 and v["turn"] != "right")
        if requires_green and (not self.is_green(t, street)):
            self.startup_pending[street][lane] = False
            return

        # Execute release
        v["release_time"] = t
        v["exit_time"] = t + v["cross_time"]
        self._log_release(t, street, lane, vid)

        self.stopline[street][lane] = None
        self.startup_pending[street][lane] = False

        self._schedule_move_up_if_possible(t, street, lane)

        # Allow additional releases at same timestamp
        self._attempt_releases(t)

    def is_green(self, t, approach):
        if self.all_red:
            return False
        return approach == self.current_green

    def _controlled_queue_len(self, street):
        q = 0
        # lanes 1/2 always controlled
        for lane in (1, 2):
            q += (self.stopline[street][lane] is not None) + len(self.behind[street][lane])

        # lane 3: only count vehicles that require green (non-right)
        vid = self.stopline[street][3]
        if vid is not None and self.vehicles[vid]["turn"] != "right":
            q += 1
        q += sum(1 for vid in self.behind[street][3] if self.vehicles[vid]["turn"] != "right")
        return q

    def _update_kappa_to(self, t):
        dt = t - self.last_kappa_update
        if dt <= 0:
            return

        if self.all_red:
            # all approaches are red
            for s in streets:
                n = self._controlled_queue_len(s)
                self.kappa[s] += n * dt
        else:
            # integrate only for red approaches 
            for s in streets:
                if s == self.current_green:
                    continue
                n = self._controlled_queue_len(s)
                self.kappa[s] += n * dt

        self.last_kappa_update = t

    def _start_all_red(self, t, next_green):
        # Enter clearance
        self.all_red = True
        self.pending_green = next_green
        self.all_red_until = t + self.all_red_duration

        # No approach is green during clearance (all red phase)
        old_green = self.current_green
        self.current_green = None

        # Reset the old green's kappa 
        if old_green is not None:
            self.kappa[old_green] = 0.0

        # Schedule event that ends all-red and activates the pending green
        self._push(self.all_red_until, "end_all_red", None)

    def _end_all_red(self, t):
        self.all_red = False
        new_green = self.pending_green
        self.pending_green = None
        self.all_red_until = None

        self.current_green = new_green
        self.green_start_time = t  # mark green start time

        # reset kappa for new green approach
        self.kappa[new_green] = 0.0

        if USE_STARTUP_LOST:
            self._arm_startup_on_green_begin(new_green, t)

        # now vehicles can release on lanes 1/2 for this approach
        self._attempt_releases(t)

    def _schedule_next_sotl_check(self, now):
        EPS = 1e-6

        if self.all_red:
            # schedule a check just after the end
            t_next = self.all_red_until
            if t_next is not None and t_next > now + EPS:

                if self._next_sotl_time is None or t_next < self._next_sotl_time - 1e-12:
                    self._sotl_token += 1
                    self._next_sotl_time = t_next
                    self._push(t_next, "sotl_check", self._sotl_token)
            return

        # Compute earliest possible switch time due to min green
        earliest = max(now, self.green_start_time + self.phi_min)

        best_time = None

        for s in streets:
            if s == self.current_green:
                continue
            n = self._controlled_queue_len(s)
            if n <= 0:
                continue

            remaining = self.theta[s] - self.kappa[s]

            if remaining <= 0:
                t_hit = earliest
            else:
                t_hit = now + (remaining / n)
                if t_hit < earliest:
                    t_hit = earliest

            if best_time is None or t_hit < best_time:
                best_time = t_hit

        if best_time is None:
            self._next_sotl_time = None
            return

        # Force strictly future scheduling 
        if best_time <= now + EPS:
            best_time = now + EPS


        # Only schedule if
        # -nothing is scheduled 
        # - or this is earlier than the currently scheduled check
        if self._next_sotl_time is not None and best_time >= self._next_sotl_time - 1e-12:
            return

        # Schedule the improved earlier check
        self._sotl_token += 1
        token = self._sotl_token
        self._next_sotl_time = best_time
        self._push(best_time, "sotl_check", token)

    def _attempt_sotl_switch(self, t):
        if self.all_red:
            return False

        # must satisfy minimum green time
        if (t - self.green_start_time) < self.phi_min:
            return False

        # find any RED approach at/over threshold
        candidates = []
        for s in streets:
            if s == self.current_green:
                continue
            if self.kappa[s] >= self.theta[s]:
                candidates.append(s)

        if not candidates:
            return False

        new_green = max(candidates, key=lambda s: self.kappa[s])

        # start clearance instead of immediate green
        self._start_all_red(t, new_green)
        return True

    def _push(self, time, kind, payload):
        if time > self.horizon:
            return
        self.seq += 1
        heapq.heappush(self.ev, (time, self.seq, kind, payload))

    def _schedule_move_up_if_possible(self, t, street, lane):
        if self.stopline[street][lane] is not None:
            return
        if len(self.behind[street][lane]) == 0:
            return
        dt = row_move_time[street]
        self._push(t + dt, "move_up", (street, lane))

    def _right_is_blocked(self, street):
        return False

    def _log_release(self, t, street, lane, vid):
        self.release_log.append((t, street, lane, self.vehicles[vid]["turn"]))

    def _attempt_releases(self, t):
        made_progress = True
        while made_progress:
            made_progress = False

            # Lanes 1 and 2 require green
            for s in streets:
                if not self.is_green(t, s):
                    continue

                for lane in (1, 2):
                    vid = self.stopline[s][lane]
                    if vid is None:
                        continue

                    if USE_STARTUP_LOST and self.startup_pending[s][lane]:
                        continue

                    if USE_STARTUP_LOST and self.startup_needed[s][lane]:
                        self.startup_needed[s][lane] = False
                        self.startup_pending[s][lane] = True
                        self._push(t + STARTUP_LOST_TIME, "delayed_release", (s, lane, vid))
                        continue

                    v = self.vehicles[vid]
                    v["release_time"] = t
                    v["exit_time"] = t + v["cross_time"]
                    self._log_release(t, s, lane, vid)

                    self.stopline[s][lane] = None
                    made_progress = True
                    self._schedule_move_up_if_possible(t, s, lane)

            # Lane 3 discharge rules
            # - Right turns are allowed regardless of signal, subject to blocking
            # - Non-right in lane 3 only on green and they remain FIFO constrained
            for s in streets:
                lane = 3
                vid = self.stopline[s][lane]
                if vid is None:
                    continue

                if USE_STARTUP_LOST and self.startup_pending[s][lane]:
                    continue

                v = self.vehicles[vid]

                if v["turn"] == "right":
                    if self._right_is_blocked(s):
                        continue
                else:
                    if not self.is_green(t, s):
                        continue

                    if USE_STARTUP_LOST and self.startup_needed[s][lane]:
                        self.startup_needed[s][lane] = False
                        self.startup_pending[s][lane] = True
                        self._push(t + STARTUP_LOST_TIME, "delayed_release", (s, lane, vid))
                        continue

                v["release_time"] = t
                v["exit_time"] = t + v["cross_time"]
                self._log_release(t, s, lane, vid)

                self.stopline[s][lane] = None
                made_progress = True
                self._schedule_move_up_if_possible(t, s, lane)

    def _handle_arrival(self, t, street):
        turn = sample_turn(self.rng, street)
        cross = sample_crossing_time(self.rng, street)

        # queue lengths include stopline and behind
        q1 = (1 if self.stopline[street][1] is not None else 0) + len(self.behind[street][1])
        q2 = (1 if self.stopline[street][2] is not None else 0) + len(self.behind[street][2])
        q3 = (1 if self.stopline[street][3] is not None else 0) + len(self.behind[street][3])

        if turn == "right":
            lane = 3
        else:
            lane = choose_lane_for_left_or_straight(turn, q1, q2)

            # diversion rule: straight can choose lane 3 with prob 0.10 if lane 3 is strictly shortest
            if ALLOW_STRAIGHT_TO_LANE3 and turn == "straight" and q3 < min(q1, q2):
                if self.rng.random() < P_STRAIGHT_TO_LANE3:
                    lane = 3

        vid = self.next_vid
        self.next_vid += 1

        self.vehicles[vid] = {
            "id": vid,
            "street": street,
            "lane": lane,
            "turn": turn,
            "arrival_time": t,
            "cross_time": cross,
            "release_time": None,
            "exit_time": None,
        }

        if self.stopline[street][lane] is None:
            self.stopline[street][lane] = vid
        else:
            self.behind[street][lane].append(vid)

        self._attempt_releases(t)

        t_next = t + sample_iat(self.rng, street)
        self._push(t_next, "arrival", street)

    def _handle_move_up(self, t, street, lane):
        if self.stopline[street][lane] is not None:
            return
        if len(self.behind[street][lane]) == 0:
            return
        self.stopline[street][lane] = self.behind[street][lane].popleft()
        self._attempt_releases(t)

    def advance_to(self, t_target):
        t_target = min(float(t_target), self.horizon)

        last_t = None
        same_time_count = 0

        while self.ev and self.ev[0][0] <= t_target:
            t, _, kind, payload = heapq.heappop(self.ev)

            if last_t is not None and abs(t - last_t) < 1e-12:
                same_time_count += 1
                if same_time_count > 100000:
                    raise RuntimeError(
                        "Too many events at the same simulation time "
                        "(likely immediate re-scheduling of sotl_check)"
                    )
            else:
                same_time_count = 0
            last_t = t

            self._update_kappa_to(t)
            self.now = t

            if kind == "arrival":
                self._handle_arrival(t, payload)
            elif kind == "move_up":
                s, lane = payload
                self._handle_move_up(t, s, lane)
            elif kind == "sotl_check":
                token = payload
                if token == self._sotl_token:
                    self._next_sotl_time = None
                    self._attempt_sotl_switch(t)
            elif kind == "end_all_red":
                self._end_all_red(t)
            elif kind == "delayed_release":
                s, lane, vid = payload
                self._handle_delayed_release(t, s, lane, vid)

            self._schedule_next_sotl_check(self.now)

        self._update_kappa_to(t_target)
        self.now = t_target

    def queue_lengths(self):
        """Return q_len[street][lane] including stopline + behind."""
        q = {s: {} for s in streets}
        for s in streets:
            for lane in lanes:
                q[s][lane] = (1 if self.stopline[s][lane] is not None else 0) + len(self.behind[s][lane])
        return q


# batch simulation 

def run_one_hour(seed=1, horizon=HORIZON_DEFAULT, warmup=WARMUP, theta=15.0, phi_min=10.0, all_red_duration=10.0):

    sim = TrafficSim(seed=seed, horizon=horizon, warmup=warmup, theta=theta, phi_min=phi_min,
                     all_red_duration=all_red_duration)
    sim.advance_to(horizon)

    # stats: time in system -> exit_time - arrival_time 
    tis = []
    tis_by_street = {s: [] for s in streets}

    for v in sim.vehicles.values():
        if v["exit_time"] is None:
            continue
        if v["exit_time"] > horizon:
            continue
        if v["arrival_time"] < warmup:
            continue

        dt = v["exit_time"] - v["arrival_time"]
        tis.append(dt)
        tis_by_street[v["street"]].append(dt)

    mean_tis = sum(tis) / len(tis) if tis else float("nan")
    mean_tis_by_street = {
        s: (sum(lst) / len(lst) if lst else float("nan"))
        for s, lst in tis_by_street.items()
    }

    # queue lengths at end (include stop line + behind)
    q_end = sim.queue_lengths()

    return {
        "vehicles_created": len(sim.vehicles),
        "departures_within_horizon": len(tis),
        "mean_time_in_system": mean_tis,
        "mean_time_in_system_by_street": mean_tis_by_street,
        "queue_end": q_end,
    }


# Confidence interval calculation

def mean_confidence_interval(samples, alpha=0.05):

    samples = np.asarray(samples, dtype=float)
    n = samples.size

    mean = samples.mean()
    std = samples.std(ddof=1)
    se = std / np.sqrt(n)

    tcrit = t.ppf(1.0 - alpha / 2.0, df=n - 1)

    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    return mean, ci_low, ci_high


# Matplotlib animation

def animate_debug(
        seed=2,
        horizon=300.0,
        warmup=WARMUP,
        theta=15.0,
        phi_min=10.0,
        all_red_duration=10.0,
        fps=10,
        time_scale=2.0,
        max_dots_per_lane=50,
        junction_window=1.2,
        save_mp4=False,  # NEW
        mp4_filename="traffic_debug.mp4"
):

    sim = TrafficSim(seed=seed, horizon=horizon, warmup=warmup, theta=theta, phi_min=phi_min,
                     all_red_duration=all_red_duration)

    state = {
        "paused": False,
        "frame": 0,
        "fps": float(fps),
        "time_scale": float(time_scale),
        "dt": float(time_scale) / float(fps),  # CRITICAL CHANGE
    }

    def _reset_sim():
        nonlocal sim
        sim = TrafficSim(
            seed=seed,
            horizon=horizon,
            warmup=warmup,
            theta=theta,
            phi_min=phi_min,
            all_red_duration=all_red_duration,
        )
        state["frame"] = 0

    # layout parameters 
    approach_x = {s: i for i, s in enumerate(streets)}
    lane_y_base = {1: 0.0, 2: 1.2, 3: 2.4}
    dot_spacing = 0.09
    stopline_offset = 0.15
    lane_band_height = 0.9

    y_junction_base = lane_y_base[3] + lane_band_height + 0.05
    y_signal_base = y_junction_base + 0.45

    turn_marker = {"left": "<", "straight": "o", "right": ">"}
    turn_order = ["left", "straight", "right"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(f"Traffic Debug Animation")
    ax.set_xlim(-0.6, len(streets) - 0.4)
    ax.set_ylim(-0.4, y_signal_base + 0.65)
    ax.set_xticks(range(len(streets)))
    ax.set_xticklabels(streets)
    ax.set_yticks([lane_y_base[1], lane_y_base[2], lane_y_base[3], y_junction_base, y_signal_base])
    ax.set_yticklabels(["Lane 1 (L/S)", "Lane 2 (L/S)", "Lane 3 (R)", "Junction", "Signal"])
    ax.grid(True, axis="x", alpha=0.2)
    ax.grid(True, axis="y", alpha=0.2)

    for y in [lane_y_base[1] + lane_band_height, lane_y_base[2] + lane_band_height]:
        ax.axhline(y=y, linewidth=1)

    # Signal panels 
    signal_patches = {}
    signal_text = {}
    for s in streets:
        x = approach_x[s]
        rect = Rectangle((x - 0.35, y_signal_base), 0.7, 0.32, alpha=0.6)
        ax.add_patch(rect)
        signal_patches[s] = rect
        signal_text[s] = ax.text(x, y_signal_base + 0.16, "", ha="center", va="center", fontsize=9)

    # Blocked right-turn hatch overlay
    blocked_patches = {}
    for s in streets:
        x = approach_x[s]
        rect = Rectangle(
            (x - 0.35, lane_y_base[3]),
            0.7,
            lane_band_height,
            fill=False,
            hatch="///",
            alpha=0.6,
            visible=False,
        )
        ax.add_patch(rect)
        blocked_patches[s] = rect

    # 3 scatters per lane 
    scat_behind = {}
    scat_stopline = {}
    for s in streets:
        for lane in lanes:
            for trn in turn_order:
                scat_behind[(s, lane, trn)] = ax.scatter([], [], s=16, marker=turn_marker[trn])
                scat_stopline[(s, lane, trn)] = ax.scatter([], [], s=55, marker=turn_marker[trn])

    # Junction: one per turn
    scat_junction = {trn: ax.scatter([], [], s=26, marker=turn_marker[trn]) for trn in turn_order}

    # Overflow labels
    overflow_text = {}
    for s in streets:
        for lane in lanes:
            x = approach_x[s]
            y = lane_y_base[lane] + lane_band_height - 0.05
            overflow_text[(s, lane)] = ax.text(x + 0.18, y, "", ha="left", va="top", fontsize=8)

    # HUD
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")
    help_text = ax.text(
        0.02, 0.05,
        "Space: pause/resume | Right: step | R: reset",
        transform=ax.transAxes, ha="left", va="top", fontsize=9
    )

    parameter_text = ax.text(
        0.02, 0.10,
        f"θ={theta} cs | φ_min={phi_min} s, | AR={all_red_duration} s",
        transform=ax.transAxes, ha="left", va="top", fontsize=9
    )

    empty = np.empty((0, 2))

    def _stopline_xy(street, lane):
        x = approach_x[street]
        y = lane_y_base[lane] + stopline_offset
        return x, y

    def _signal_status(now, approach):
        if sim.all_red:
            rem = max(0.0, sim.all_red_until - now)
            return "AR", rem
        if approach == sim.current_green:
            return "G", max(0.0, sim.phi_min - (now - sim.green_start_time))
        else:
            return "R", sim.kappa[approach]

    # Pause/step key controls
    def on_key(event):
        key = event.key
        if key == " ":
            state["paused"] = not state["paused"]
        elif key == "right":
            if state["paused"]:
                state["frame"] += 1
                _render_frame(state["frame"])
                fig.canvas.draw_idle()
        elif key in ("r", "R"):
            _reset_sim()
        elif key == "+":
            state["time_scale"] *= 2.0
            state["dt"] = state["time_scale"] / state["fps"]
        elif key == "-":
            state["time_scale"] /= 2.0
            state["dt"] = state["time_scale"] / state["fps"]

    fig.canvas.mpl_connect("key_press_event", on_key)

 
    #Rendering
    def _render_frame(frame_idx):
        t = warmup + frame_idx * state["dt"]
        sim.advance_to(t)

        # prune old releases
        while sim.release_log and (sim.now - sim.release_log[0][0]) > junction_window:
            sim.release_log.popleft()

        # signals
        for s, rect in signal_patches.items():
            st, rem = _signal_status(sim.now, s)
            rect.set_facecolor("green" if st == "G" else "red")
            signal_text[s].set_text(f"{st} {rem:0.1f}s")

        # blocked right-turn hatch
        for s in streets:
            lane3_vid = sim.stopline[s][3]
            is_blocked = (lane3_vid is not None) and sim._right_is_blocked(s)
            blocked_patches[s].set_visible(bool(is_blocked))

        # queue lengths for overflow label
        q = sim.queue_lengths()

        # per lane: update stopline + behind scatters
        for s in streets:
            for lane in lanes:
                x0, y0 = _stopline_xy(s, lane)

                extra = int(q[s][lane]) - int(max_dots_per_lane)
                overflow_text[(s, lane)].set_text(f"+{extra}" if extra > 0 else "")

                # stopline: clear 3, set 1 if present
                scat_stopline[(s, lane, "left")].set_offsets(empty)
                scat_stopline[(s, lane, "straight")].set_offsets(empty)
                scat_stopline[(s, lane, "right")].set_offsets(empty)

                stop_vid = sim.stopline[s][lane]
                if stop_vid is not None:
                    trn = sim.vehicles[stop_vid]["turn"]
                    scat_stopline[(s, lane, trn)].set_offsets(np.asarray([[x0, y0]], dtype=float))

                # behind: build coords lists
                coords_left = []
                coords_straight = []
                coords_right = []


                idx = 0
                for vid in sim.behind[s][lane]:
                    if idx >= max_dots_per_lane:
                        break
                    trn = sim.vehicles[vid]["turn"]
                    yy = y0 + (idx + 1) * dot_spacing
                    if trn == "left":
                        coords_left.append([x0, yy])
                    elif trn == "straight":
                        coords_straight.append([x0, yy])
                    else:
                        coords_right.append([x0, yy])
                    idx += 1

                scat_behind[(s, lane, "left")].set_offsets(
                    np.asarray(coords_left, dtype=float) if coords_left else empty)
                scat_behind[(s, lane, "straight")].set_offsets(
                    np.asarray(coords_straight, dtype=float) if coords_straight else empty)
                scat_behind[(s, lane, "right")].set_offsets(
                    np.asarray(coords_right, dtype=float) if coords_right else empty)

        # junction releases
        coords_junc_left = []
        coords_junc_straight = []
        coords_junc_right = []

        for (t_rel, street, lane, trn) in sim.release_log:
            age = sim.now - t_rel
            y = y_junction_base + 0.30 * (age / junction_window)
            x = approach_x[street] + (lane - 2) * 0.06
            if trn == "left":
                coords_junc_left.append([x, y])
            elif trn == "straight":
                coords_junc_straight.append([x, y])
            else:
                coords_junc_right.append([x, y])

        scat_junction["left"].set_offsets(np.asarray(coords_junc_left, dtype=float) if coords_junc_left else empty)
        scat_junction["straight"].set_offsets(
            np.asarray(coords_junc_straight, dtype=float) if coords_junc_straight else empty)
        scat_junction["right"].set_offsets(np.asarray(coords_junc_right, dtype=float) if coords_junc_right else empty)

        time_text.set_text(
            f"t = {sim.now:7.2f}s   "
            f"fps={state['fps']:.0f}   "
            f"speed={state['time_scale']:.1f}×   "
            f"{'PAUSED' if state['paused'] else ''}"
        )

    # Animation
    visible = max(0.0, horizon - warmup)
    frames = int(math.ceil(visible * state["fps"])) + 1

    def init():
        for s in streets:
            signal_patches[s].set_facecolor("red")
            signal_text[s].set_text("")
            blocked_patches[s].set_visible(False)

        time_text.set_text("")
        for txt in overflow_text.values():
            txt.set_text("")

        for sc in list(scat_behind.values()) + list(scat_stopline.values()) + list(scat_junction.values()):
            sc.set_offsets(empty)

        return []

    def update(_):
        if not state["paused"]:
            state["frame"] += 1
        _render_frame(state["frame"])
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=int(1000 / max(1.0, state["fps"])),
        blit=False,
        repeat=False,
    )

    if save_mp4:
        writer = FFMpegWriter(
            fps=state["fps"],
            metadata=dict(
                title="Traffic Signal Debug Animation",
                artist="TrafficSim",
                comment=f"speed={time_scale}x",
            ),
            bitrate=1800,
        )

        print(f"Saving MP4 to '{mp4_filename}' ...")
        ani.save(mp4_filename, writer=writer)
        print("MP4 export complete.")

    else:
        plt.show()

    return ani


# SPSA (over theta, the same for all approaches)

def evaluate(theta_vec, seed=None, horizon=None, warmup=None, phi_min=None, all_red_duration=None):
    if seed is None:
        seed = 1
    if horizon is None:
        horizon = HORIZON_DEFAULT
    if warmup is None:
        warmup = WARMUP
    if phi_min is None:
        phi_min = 5.0
    if all_red_duration is None:
        all_red_duration = 5.0

    sim = run_one_hour(
        seed=seed,
        horizon=horizon,
        warmup=warmup,
        theta=theta_vec,
        phi_min=phi_min,
        all_red_duration=all_red_duration,
    )
    return sim["mean_time_in_system"]


def project(theta, lo, hi):
    return np.minimum(np.maximum(theta, lo), hi)


def spsa_optimize_projection( # This version projects the wrong way
        theta0,
        evaluate_fn,
        n_iter=200,
        epsilon=1.0,
        eta=1.0,
        lo=0.0,
        hi=np.inf,
        batch=1,
        seed0=12345,
):
    theta = np.array(theta0, dtype=float)
    d = theta.size

    history = []

    for k in range(n_iter):
        # Rademacher perturbation
        delta = np.where(np.random.rand(d) < 0.5, -1.0, +1.0)

        theta_plus = project(theta + eta * delta, lo, hi)
        theta_minus = project(theta - eta * delta, lo, hi)

        # Common random numbers: same seeds for + and -
        j_plus = 0.0
        j_minus = 0.0
        for r in range(batch):
            seed = seed0 + 10_000 * k + r
            j_plus += evaluate_fn(theta_plus, seed)
            j_minus += evaluate_fn(theta_minus, seed)
        j_plus /= batch
        j_minus /= batch

        # SPSA gradient estimate
        ghat = (j_plus - j_minus) / (2.0 * eta) * (1.0 / delta)

        # update
        theta_new = project(theta - epsilon * ghat, lo, hi)

        history.append({
            "k": k,
            "theta": theta.copy(),
            "theta_plus": theta_plus,
            "theta_minus": theta_minus,
            "J_plus": j_plus,
            "J_minus": j_minus,
            "ghat": ghat,
            "epsilon": epsilon,
            "eta": eta,
        })
        theta = theta_new

    return theta, history

def spsa_optimize(
        theta0,
        evaluate_fn,
        n_iter=200,
        epsilon=1.0,
        eta=1.0,
        lo=0.0,
        hi=np.inf,
        batch=1,
        seed0=12345,
        rng_seed=12345,
        safety=1e-8,
):
    theta = np.array(theta0, dtype=float)
    d = theta.size if theta.ndim > 0 else 1
    theta = np.atleast_1d(theta)

    lo = np.full(d, lo) if np.isscalar(lo) else np.array(lo, dtype=float)
    hi = np.full(d, hi) if np.isscalar(hi) else np.array(hi, dtype=float)

    rng = np.random.default_rng(rng_seed)
    history = []

    for k in range(n_iter):
        delta = rng.choice([-1.0, +1.0], size=d)

        # choose feasible symmetric eta
        dist_to_lo = theta - lo
        dist_to_hi = hi - theta
        eta_max = np.min(np.minimum(dist_to_lo, dist_to_hi))
        eta_eff = min(eta, max(0.0, eta_max - safety))

        if eta_eff <= 0.0:
            history.append({
                "k": k,
                "theta": theta.copy(),
                "status": "stopped_no_feasible_symmetric_perturbation",
                "eta": eta,
                "eta_eff": eta_eff,
            })
            break

        theta_plus = theta + eta_eff * delta
        theta_minus = theta - eta_eff * delta

        j_plus = 0.0
        j_minus = 0.0
        for r in range(batch):
            seed = seed0 + 10_000 * k + r
            j_plus += evaluate_fn(theta_plus, seed)
            j_minus += evaluate_fn(theta_minus, seed)
        j_plus /= batch
        j_minus /= batch

        ghat = (j_plus - j_minus) / (2.0 * eta_eff) * (1.0 / delta)

        theta_new = np.minimum(np.maximum(theta - epsilon * ghat, lo), hi)

        history.append({
            "k": k,
            "theta": theta.copy(),
            "theta_plus": theta_plus.copy(),
            "theta_minus": theta_minus.copy(),
            "J_plus": j_plus,
            "J_minus": j_minus,
            "ghat": ghat.copy(),
            "epsilon": epsilon,
            "eta": eta,
            "eta_eff": eta_eff,
            "status": "ok",
        })
        theta = theta_new

    return theta if theta.size > 1 else theta.item(), history


def scalar(x):
    a = np.asarray(x)
    if a.ndim == 0:
        return a.item()
    if a.size == 1:
        return a.reshape(-1)[0].item()
    raise ValueError(f"Expected scalar objective, got shape {a.shape}")


start = time.time()


# Run simulation / animation / SPSA via CONFIG only

def run_from_config(cfg):
    mode = cfg["MODE"]

    if mode == "spsa":
        start = time.time()

        theta_star, hist = spsa_optimize(
            theta0=np.array(cfg["SPSA_THETA0"], dtype=float),
            evaluate_fn=lambda th, seed: evaluate(
                th,
                seed=seed,
                warmup=cfg["WARMUP"],
                horizon=cfg["HORIZON"],
                phi_min=cfg["PHI_MIN"],
                all_red_duration=cfg["ALL_RED_DURATION"],
            ),
            lo=cfg["SPSA_LO"],
            hi=cfg["SPSA_HI"],
            epsilon=cfg["SPSA_EPSILON"],
            eta=cfg["SPSA_ETA"],
            n_iter=cfg["SPSA_N_ITER"],
            batch=cfg["SPSA_BATCH"],
            seed0=cfg["SEED"],
        )

        print("Optimised theta (Ba1,Ba2,Be1,Be2):", theta_star)
        print("Final estimated objective:", scalar((hist[-1]["J_plus"] + hist[-1]["J_minus"]) / 2.0))

        if cfg.get("SPSA_PLOT", True):
            xs = [h["k"] for h in hist]
            thetas = np.array([h["theta"] for h in hist])

            plt.figure()
            for i in range(thetas.shape[1]):
                plt.plot(xs, thetas[:, i], label=f"theta[{APPROACHES[i]}]")
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Theta")
            plt.title("SPSA Optimisation Path")
            plt.grid(True)
            plt.show()

            Js = [scalar((h["J_plus"] + h["J_minus"]) / 2.0) for h in hist]
            plt.figure()
            plt.plot(xs, Js)
            plt.xlabel("Iteration")
            plt.ylabel("Objective (estimated)")
            plt.title("SPSA Objective Over Iterations")
            plt.grid(True)
            plt.show()

        end = time.time()
        print(f"SPSA optimisation took {end - start:0.2f} seconds.")
        return

    if mode == "batch":
        run = run_one_hour(
            seed=cfg["SEED"],
            horizon=cfg["HORIZON"],
            warmup=cfg["WARMUP"],
            theta=np.array(cfg["THETA"], dtype=float),
            phi_min=cfg["PHI_MIN"],
            all_red_duration=cfg["ALL_RED_DURATION"],
        )
        print("mean_time_in_system:", run["mean_time_in_system"])
        print("mean_time_in_system_by_street:", run["mean_time_in_system_by_street"])
        print("queue_end:", run["queue_end"])
        return

    if mode == "animate":
        animate_debug(
            seed=cfg["SEED"],
            horizon=cfg["HORIZON"],
            warmup=cfg["WARMUP"],
            theta=np.array(cfg["THETA"], dtype=float),
            phi_min=cfg["PHI_MIN"],
            all_red_duration=cfg["ALL_RED_DURATION"],
            fps=cfg["FPS"],
            time_scale=cfg["TIME_SCALE"],
            save_mp4=cfg["SAVE_MP4"],
            mp4_filename=cfg["FILENAME"],
        )
        return

    raise ValueError(f"Unknown MODE='{mode}'. Use 'batch', 'animate', or 'spsa'.")


if __name__ == "__main__":
    CONFIG = {
        # One of: "batch", "animate", "spsa"
        "MODE": "spsa",

        # Shared simulation settings
        "SEED": 1,
        "HORIZON": 360.0,
        "WARMUP": 0.0,
        "PHI_MIN": 5,
        "ALL_RED_DURATION": 5.0,

        # Threshold vector for batch/animate (Ba1, Ba2, Be1, Be2)
        "THETA": [50.0, 110.0, 50.0, 100.0],

        # Animation settings
        "FPS": 10,
        "TIME_SCALE": 1.5,
        "SAVE_MP4": False,
        "FILENAME": "traffic_debug.mp4",

        # SPSA settings
        "SPSA_THETA0": [60.0, 60.0, 60.0, 60.0],
        "SPSA_EPSILON": 5.0,
        "SPSA_ETA": 30.0,
        "SPSA_N_ITER": 200,
        "SPSA_BATCH": 5,
        "SPSA_LO": 0.0,
        "SPSA_HI": 200.0,
        "SPSA_PLOT": True,
    }

    run_from_config(CONFIG)