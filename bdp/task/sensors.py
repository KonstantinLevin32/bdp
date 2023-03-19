from collections import OrderedDict, defaultdict

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSparseReward, CompositeSuccess)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.rearrange_sensors import NumStepsMeasure
from habitat.tasks.rearrange.utils import (UsesRobotInterface,
                                           batch_transform_point,
                                           coll_name_matches)
from habitat.tasks.utils import cartesian_to_polar, get_angle


@registry.register_measure
class NoReward(Measure):
    cls_uuid: str = "no_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NoReward.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0.0

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_measure
class OccupancyMeasure(Measure):
    cls_uuid: str = "occupancy_measure"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OccupancyMeasure.cls_uuid

    def reset_metric(self, *args, task, observations, **kwargs):
        # all_preds = observations["all_predicates"]
        # if self._robo_preds is None:
        #     self._robo_preds = []
        #     preds = task.pddl_problem.get_possible_predicates()
        #     for robo_i in [0, 1]:
        #         self._robo_preds.append(
        #             [
        #                 pred_i
        #                 for pred_i, p in enumerate(preds)
        #                 if p.name == "robot_at" and f"ROBOT_{robo_i}" in str(p)
        #             ]
        #         )
        # self._pred_sums = np.zeros((2, len(self._robo_preds[0])))
        # self._num_steps = 0
        self._agent_pos = defaultdict(list)
        self.update_metric(
            *args,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent in ["AGENT_0", "AGENT_1"]:
            pos = observations[f"{agent}_localization_sensor"]
            self._agent_pos[agent].append([pos[0], pos[2]])
        self._metric = self._agent_pos

        # self._num_steps += 1
        # for robo_i, robo_preds in enumerate(self._robo_preds):
        #     truth_vals = np.array([all_preds[pred_i] for pred_i in robo_preds])
        #     self._pred_sums[robo_i] += truth_vals
        # self._metric = self._pred_sums / self._num_steps

    def __init__(self, sim, config, *args, **kwargs):
        # self._robo_preds = None
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_measure
class InvalidPddlActionPenalty(Measure):
    cls_uuid: str = "pddl_action_penalty"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return InvalidPddlActionPenalty.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if (
            self._config.PENALTY != 0.0
            and task.actions["PDDL_APPLY_ACTION"].was_prev_action_invalid
        ):
            self._metric = self._config.PENALTY
        else:
            self._metric = 0.0

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_sensor
class NavDidCollideSensor(UsesRobotInterface, Sensor):
    cls_uuid: str = "nav_did_collide"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavDidCollideSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        nav_ac = task.actions[f"AGENT_{self.robot_id}_BASE_VELOCITY"]
        return np.array([float(nav_ac.did_collide)]).astype(np.float32)


@registry.register_sensor
class ActionHistorySensor(UsesRobotInterface, Sensor):
    cls_uuid: str = "action_history"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        self._pddl_action = None
        self._cur_write_idx = 0
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return ActionHistorySensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    @property
    def pddl_action(self):
        if self._pddl_action is None:
            robot_id = self.robot_id if self.robot_id is not None else "0"
            self._pddl_action = self._task.actions[
                f"AGENT_{robot_id}_PDDL_APPLY_ACTION"
            ]
        return self._pddl_action

    def _get_observation_space(self, *args, config, **kwargs):
        self._action_ordering = self._task.pddl_problem.get_ordered_actions()
        entities_list = self._task.pddl_problem.get_ordered_entities_list()

        self._action_map = {}
        self._action_offsets = {}
        self._n_actions = 0
        for action in self._action_ordering:
            param = action._params[0]
            self._action_map[action.name] = [
                entity
                for entity in entities_list
                if entity.expr_type.is_subtype_of(param.expr_type)
            ]
            self._action_offsets[action.name] = self._n_actions
            self._n_actions += len(self._action_map[action.name])
        self._dat = np.zeros((self.config.WINDOW, self._n_actions), dtype=np.float32)

        return spaces.Box(
            shape=(self.config.WINDOW * self._n_actions + 1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, **kwargs):
        self._cur_write_idx = self._cur_write_idx % self.config.WINDOW
        if self._task._cur_episode_step == 0 or self._cur_write_idx == 0:
            self._cur_write_idx = 0
            self._dat *= 0.0

        ac = self.pddl_action._prev_action
        if ac is not None:
            if self.pddl_action.was_prev_action_invalid:
                set_idx = self._action_offsets["noop"]
            else:
                use_name = ac.name
                set_idx = self._action_offsets[use_name]
                param_value = ac.param_values[0]
                entities = self._action_map[use_name]
                set_idx += entities.index(param_value)
            self._dat[self._cur_write_idx, set_idx] = 1.0
            self._cur_write_idx += 1
        cur_step = self._task._cur_episode_step / self.config.STEPS

        return np.concatenate([[cur_step], self._dat.reshape(-1)])


@registry.register_sensor
class CollInfoSensor(Sensor):
    cls_uuid: str = "coll_info"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return CollInfoSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim.perform_discrete_collision_detection()
        contact_points = self._sim.get_physics_contact_points()
        found_contact = False

        robot_ids = [
            robot.get_robot_sim_id() for robot in self._sim.robots_mgr.robots_iter
        ]
        assert len(robot_ids) == 2

        for cp in contact_points:
            if coll_name_matches(cp, robot_ids[0]) and coll_name_matches(
                cp, robot_ids[1]
            ):
                found_contact = True

        robot_pos = [robot.base_pos for robot in self._sim.robots_mgr.robots_iter]
        dist = np.linalg.norm(np.array(robot_pos[0] - robot_pos[1]))

        return np.array([dist, float(found_contact)], dtype=np.float32)


@registry.register_sensor
class MultiRobotRelObjPosSensor(UsesRobotInterface, Sensor):
    """
    The current robot is returned first in the observation space.
    """

    cls_uuid: str = "multi_robot_rel_obj_pos"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return MultiRobotRelObjPosSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        self.n_targets = config.N_TARGETS
        # 2 robots, 2d polar coordinates
        self._polar_pos = np.zeros(
            (config.N_ROBOTS, self.n_targets, 2), dtype=np.float32
        )
        return spaces.Box(
            shape=(self.n_targets * config.N_ROBOTS * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        _, pos = self._sim.get_targets()
        pos = pos[: self.n_targets]

        other_robot_id = (self.robot_id + 1) % 2

        for robot_id in [self.robot_id, other_robot_id]:
            robot_T = self._sim.get_robot_data(robot_id).robot.base_transformation
            rel_pos = batch_transform_point(pos, robot_T.inverted(), np.float32)

            for i, rel_obj_pos in enumerate(rel_pos):
                rho, phi = cartesian_to_polar(rel_obj_pos[0], rel_obj_pos[2])
                self._polar_pos[robot_id, i] = [rho, -phi]

        return self._polar_pos.reshape(-1)


@registry.register_sensor
class RelativePositionsSensor(UsesRobotInterface, Sensor):
    cls_uuid: str = "rel_pos_sensor"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return RelativePositionsSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        this_pos = self._sim.get_robot_data(self.robot_id).robot.base_pos
        other_id = (self.robot_id + 1) % 2
        other_pos = self._sim.get_robot_data(other_id).robot.base_pos
        rel_pos = other_pos - this_pos
        return np.array(rel_pos, dtype=np.float32)


@registry.register_measure
class SumInvalidPddlAction(Measure):
    cls_uuid: str = "sum_pddl_invalid_action"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SumInvalidPddlAction.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._metric = 0.0
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent in ["AGENT_0", "AGENT_1"]:
            if task.actions[f"{agent}_PDDL_APPLY_ACTION"].was_prev_action_invalid:
                self._metric += 1.0

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_measure
class NumNavActions(Measure):
    cls_uuid: str = "num_nav_actions"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumNavActions.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._metric = 0.0
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent in ["AGENT_0", "AGENT_1"]:
            base_vel = task.actions[f"{agent}_BASE_VELOCITY"].prev_base_vel
            if base_vel is not None and np.linalg.norm(base_vel) > 0.001:
                self._metric += 1.0

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_measure
class DidRobotsCollide(Measure):
    cls_uuid: str = "did_robots_collide"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidRobotsCollide.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._sim = sim

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        self._sim.perform_discrete_collision_detection()
        contact_points = self._sim.get_physics_contact_points()
        found_contact = False

        robot_ids = [
            robot.get_robot_sim_id() for robot in self._sim.robots_mgr.robots_iter
        ]
        assert len(robot_ids) == 2

        for cp in contact_points:
            if coll_name_matches(cp, robot_ids[0]) and coll_name_matches(
                cp, robot_ids[1]
            ):
                found_contact = True

        self._metric = float(found_contact)


@registry.register_measure
class SumMeasure(Measure):
    def _get_sum_measure(self):
        return None

    def reset_metric(self, *args, **kwargs):
        self._metric = 0.0
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        met = task.measurements.measures[self._get_sum_measure()].get_metric()

        self._metric += float(met)


@registry.register_measure
class NumDidRobotsCollide(SumMeasure):
    cls_uuid: str = "num_did_robots_collide"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumDidRobotsCollide.cls_uuid

    def _get_sum_measure(self):
        return DidRobotsCollide.cls_uuid


@registry.register_measure
class DistinctCollision(Measure):
    cls_uuid: str = "distinct_collide"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistinctCollision.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._sim = sim

    def reset_metric(self, *args, **kwargs):
        self._prev_contact = False
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        if self._config.GEO_COLL:
            robot_pos = [robot.base_pos for robot in self._sim.robots_mgr.robots_iter]

            robots_dist = float(np.linalg.norm(np.array(robot_pos[0] - robot_pos[1])))
            found_contact = robots_dist < self._config.DIST_THRESH
        else:
            found_contact = task.measurements.measures[
                DidRobotsCollide.cls_uuid
            ].get_metric()

        self._metric = float(found_contact and (not self._prev_contact))
        self._prev_contact = found_contact


@registry.register_measure
class NumDistinctCollide(SumMeasure):
    cls_uuid: str = "num_distinct_collide"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumDistinctCollide.cls_uuid

    def _get_sum_measure(self):
        return DistinctCollision.cls_uuid


@registry.register_measure
class RobotsDistance(Measure):
    cls_uuid: str = "robots_dist"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotsDistance.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._sim = sim

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        robot_pos = [robot.base_pos for robot in self._sim.robots_mgr.robots_iter]

        self._metric = float(np.linalg.norm(np.array(robot_pos[0] - robot_pos[1])))


@registry.register_measure
class SuccessNoColl(Measure):
    cls_uuid: str = "success_no_coll"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SuccessNoColl.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        succ = task.measurements.measures[CompositeSuccess.cls_uuid].get_metric()
        any_coll = (
            task.measurements.measures[NumDistinctCollide.cls_uuid].get_metric() > 0
        )

        if any_coll:
            self._metric = False
        else:
            self._metric = succ


@registry.register_measure
class EfficiencyMetric(Measure):
    cls_uuid: str = "efficiency"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EfficiencyMetric.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def __init__(self, config, *args, **kwargs):
        self._config = config
        super().__init__(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        succ = task.measurements.measures[CompositeSuccess.cls_uuid].get_metric()
        n_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()

        max_steps = self._config.STEPS

        self._metric = succ * ((max_steps - n_steps) / max_steps)


@registry.register_measure
class AgentBlameMeasure(Measure):
    cls_uuid: str = "agent_blame"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AgentBlameMeasure.cls_uuid

    def _get_goal_values(self, task):
        return {
            goal_name: task.pddl_problem.is_expr_true(expr)
            for goal_name, expr in task.pddl_problem.stage_goals.items()
        }

    def reset_metric(self, *args, task, **kwargs):
        self._prev_goal_states = self._get_goal_values(task)
        self._agent_blames = {}
        for agent in range(2):
            for k in self._prev_goal_states:
                self._agent_blames[f"{agent}_{k}"] = False
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        cur_goal_states = self._get_goal_values(task)

        changed_goals = []
        for goal_name, is_sat in cur_goal_states.items():
            if is_sat and not self._prev_goal_states[goal_name]:
                changed_goals.append(goal_name)

        for k in changed_goals:
            any_true = False
            sub_goal = task.pddl_problem.stage_goals[k]
            for agent in range(2):
                pddl_action = task.actions[f"AGENT_{agent}_PDDL_APPLY_ACTION"]
                if pddl_action._prev_action is None:
                    continue
                if pddl_action.was_prev_action_invalid:
                    continue
                post_cond_in_sub_goal = (
                    len(
                        AgentBlameMeasure._logical_expr_contains(
                            sub_goal, pddl_action._prev_action._post_cond
                        )
                    )
                    > 0
                )
                self._agent_blames[f"{agent}_{k}"] = post_cond_in_sub_goal
                any_true = any_true or post_cond_in_sub_goal

            # If neither of the agents can be attributed (TODO fix this
            # situation so it doesn't happen), then attribute both agents.
            if not any_true:
                for agent in range(2):
                    pddl_action = task.actions[f"AGENT_{agent}_PDDL_APPLY_ACTION"]
                    if pddl_action._prev_action is None:
                        continue
                    if pddl_action.was_prev_action_invalid:
                        continue
                    self._agent_blames[f"{agent}_{k}"] = True

        self._prev_goal_states = cur_goal_states
        self._metric = self._agent_blames

    @staticmethod
    def _logical_expr_contains(expr, preds):
        ret = []
        for sub_expr in expr.sub_exprs:
            if isinstance(sub_expr, LogicalExpr):
                ret.extend(AgentBlameMeasure._logical_expr_contains(sub_expr, preds))
            elif isinstance(sub_expr, Predicate):
                for pred in preds:
                    if sub_expr.compact_str == pred.compact_str:
                        ret.append(sub_expr)
                        break
            else:
                raise ValueError()
        return ret


@registry.register_measure
class CookingReward(CompositeSparseReward):
    cls_uuid: str = "cooking_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CookingReward.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        if self._config.END_ON_COLLIDE:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    # InvalidPddlActionPenalty.cls_uuid,
                    DistinctCollision.cls_uuid,
                ],
            )
        self._order_idx = 0
        self._stage_succ = []
        super().reset_metric(*args, task=task, **kwargs)

    def _get_stage_reward(self, name):
        reward = self._config.STAGE_SPARSE_REWARD
        if isinstance(reward, Config):
            return reward[name]
        else:
            return reward

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(*args, task=task, **kwargs)
        # pen = task.measurements.measures[
        #     InvalidPddlActionPenalty.cls_uuid
        # ].get_metric()
        # self._metric -= pen

        self._metric = 0.0

        if self._config.END_ON_COLLIDE or self._config.COLLIDE_PENALTY != 0.0:
            did_robots_collide = task.measurements.measures[
                DistinctCollision.cls_uuid
            ].get_metric()
            if did_robots_collide:
                self._metric -= self._config.COLLIDE_PENALTY
                if self._config.END_ON_COLLIDE:
                    task.should_end = True

        if not self._config.GIVE_STAGE_REWARDS:
            return

        for stage_name, logical_expr in task.pddl_problem.stage_goals.items():
            if stage_name in self._stage_succ:
                continue

            if self._config.REQUIRE_STAGE_ORDER:
                expected = self._config.STAGE_ORDERING[self._order_idx]
                if stage_name != expected:
                    continue

            if task.pddl_problem.is_expr_true(logical_expr):
                self._metric += self._get_stage_reward(stage_name)
                self._stage_succ.append(stage_name)
                self._order_idx += 1
