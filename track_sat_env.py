import os
import jdk4py
import gymnasium as gym
import numpy as np

import KalmanFilter
from multi_target_env import MultiTargetEnv

# Set JAVA_HOME BEFORE starting JVM
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)

import orekit_jpype as orekit
from jpype.types import JDouble, JArray

# Start JVM FIRST
orekit.initVM()

# import Java-dependent helpers
orekit.pyhelpers.setup_orekit_data(from_pip_library=True)

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.propagation.analytical.tle import TLE,TLEPropagator
from org.orekit.utils import Constants
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation import StateCovarianceMatrixProvider, StateCovariance
from org.hipparchus.linear import DiagonalMatrix, Array2DRowRealMatrix
from org.hipparchus.geometry.euclidean.threed import Vector3D

class TrackSatEnv(gym.Env):
    def __init__(self, d_state=6, max_steps=100):
        self.max_targets = 3
        self.max_steps = max_steps
        self.fov = np.deg2rad(2)        # diameter of FOV
        self.threshold_fov = 0.5 
        self.frame = FramesFactory.getEME2000()
        self.initialDate = AbsoluteDate(2025, 3, 23, 0, 0, 0, TimeScalesFactory.getUTC())

        # initial covariance diagonal
        diag_initial_values = [1e6, 1e6, 1e6, 1., 1., 1.]

        self.tles = {
            "TDRS05": {
                "tle": TLE(
                    "1 21639U 91054B   25081.51181291 -.00000056  00000-0  00000+0 0  9995",
                    "2 21639  14.1260 357.0831 0003813   3.2787 258.6419  0.99948788123177"
                )
            },
            "TDRS06": {
                "tle": TLE(
                    "1 22314U 93003B   25081.20863616 -.00000286  00000-0  00000+0 0  9991",
                    "2 22314  14.1718   0.3832 0006404 172.6992  36.1023  1.00270003117841"
                )
            },
            "TDRS12": {
                "tle": TLE(
                    "1 39504U 14004A   25080.81834953 -.00000263  00000-0  00000-0 0  9998",
                    "2 39504   3.6316  10.0282 0002090 260.4678 162.8071  1.00272110 39741"
                )
            }
        }

        self.satellites = {}

        # propagate to initial state
        for name, sat_data in self.tles.items():

            tle = sat_data["tle"]

            # Create propagator
            propagator = TLEPropagator.selectExtrapolator(tle)

            # Propagate to common initial epoch
            spacecraftState = propagator.propagate(self.initialDate)

            # Transform state to J2000
            pv_j2000 = spacecraftState.getPVCoordinates(self.frame)

            # Create fresh covariance matrix 
            diag_values = JArray(JDouble)(diag_initial_values)
            covariance = DiagonalMatrix(diag_values)

            self.satellites[name] = {
                "tle": tle,
                "propagator": propagator,
                "state": pv_j2000,
                "cov": np.array(covariance.getData()),
                "is_known": True
            }
            
        # Observation space
        self.obs_dim_per_target = 1
        obs_len = self.max_targets * self.obs_dim_per_target
        self.observation_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32
        )

        # Action space
        self.n_actions = self.max_targets
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # known mask
        """ self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.max_targets] = True """

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.lost_counter = 0

        # Set all satellites to known
        for sat in self.satellites.values():
            sat["is_known"] = True

        self.obs = self._get_obs()
        info = {}

        return self.obs, info
    
    def _get_obs(self, target_id=None):
        obs_list = []
        for sat in self.satellites.values():
            trace = np.trace(sat["cov"])
            known = 1.0 if sat["is_known"] else 0.0

            obs_list.append(trace * known)
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        lost_targets = []
        reward = 0
        for index, (name, sat) in enumerate(self.satellites.items()):

            # set up observation model
            observationFcn = KalmanFilter.extract_measurement

            sigma_theta = np.deg2rad(1.0 / 3600.0)
            sigma_r = 0.1

            R = np.diag([sigma_theta**2, sigma_theta**2, sigma_r**2])
            
            # propagate all targets
            currentDate = self.initialDate.shiftedBy(self.step_count)
            dt = 1.
            sat["state"], sat["cov"] = TrackSatEnv.propagate_sat_keplerian_force(sat["state"], sat["cov"], currentDate, dt, self.frame)

            # transform cov into measurement space
            

            # update state of target that is addressed in action
            if action == index:
                
                # check validity
                if not(sat["is_known"]):
                    info = {"invalid_action": True, "lost_target": lost_targets}
                    # Termination
                    done = self.step_count >= self.max_steps
                    obs = self._get_obs()
                    return obs, -1.0, done, False, info
                
                xUpdate, PUpdate = MultiTargetEnv.ekf_update(sat["state"], sat["cov"], R, observationFcn)
                Hk, Gk = observationFcn(xUpdate)
                S = Hk @ PUpdate @ Hk.T
                prob = MultiTargetEnv.compute_fov_prob_full(S, self.fov, self.fov)
                reward += prob

                # compute information gain
                iG = MultiTargetEnv.compute_kl_divergence(sat["state"], sat["cov"], xUpdate, PUpdate)
            else:
                # compare FOV-probability of neglected targets with FOV
                Hk, Gk = observationFcn(sat["state"])
                S = Hk @ sat["cov"] @ Hk.T
                prob = MultiTargetEnv.compute_fov_prob_full(S, self.fov, self.fov)
                reward += prob
                if prob<self.threshold_fov:
                    lost_targets.append(name)
                    sat["is_known"] = False     # satellite considered as lost

            obs = self._get_obs()
            self.step_count += 1

            # Termination
            done = self.step_count >= self.max_steps
            truncated = False

            # Info dict can include diagnostics
            info = {
                "sat_name": name,
                "iG": iG,
                "lost_target": lost_targets
            }
            self.obs = obs

            return obs, reward, done, truncated, info

    
    @staticmethod
    def propagate_sat_keplerian_force(x, P, initialDate, dt, frame):

        mu = Constants.WGS84_EARTH_MU

        P_matrix = Array2DRowRealMatrix(P.tolist())

        # Create initial orbit at initialDate
        orbit = CartesianOrbit(x, frame, initialDate, mu)
        covInit = StateCovariance(P_matrix, initialDate, frame, OrbitType.CARTESIAN, PositionAngleType.MEAN)

        # Keplerian propagator
        propagator = KeplerianPropagator(orbit)

        # Setup STM
        stmName = "stm"
        harvester = propagator.setupMatricesComputation(stmName, None, None)

        # Add covariance provider
        covProvider = StateCovarianceMatrixProvider(
            "covariance",
            stmName,
            harvester,
            covInit
        )

        propagator.addAdditionalDataProvider(covProvider)

        # Propagate
        targetDate = initialDate.shiftedBy(dt)
        state = propagator.propagate(targetDate)

        # Extract propagated covariance
        P_next = covProvider.getStateCovariance(state).getMatrix()

        # convert to numpy objects
        P_next = np.array(P_next.getData())
        x_next = state.getPVCoordinates(frame)
        x_next = np.array([
            x_next.getPosition().getX(),
            x_next.getPosition().getY(),
            x_next.getPosition().getZ(),
            x_next.getVelocity().getX(),
            x_next.getVelocity().getY(),
            x_next.getVelocity().getZ()
        ])

        return x_next, P_next

if __name__ == "__main__":
    env = TrackSatEnv(d_state=6)
    env.step(0)
