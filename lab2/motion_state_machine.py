from answer_task1 import *
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from direct.gui.OnscreenText import OnscreenText

mix_time = 10


class MotionState:
    def __init__(self, bvh_file_path, is_loop) -> None:
        self.motion = BVHMotion(bvh_file_path)
        self.is_loop = is_loop
        if is_loop:
            self.motion = build_loop_motion(self.motion)
        self.start_frame = 0
        self.mix_motion = None
        self.is_mixing = False

    def start(self, cur_frame, pos, facing_axis):
        self.start_frame = cur_frame
        self.motion = self.motion.translation_and_rotation(0, pos, facing_axis)
        # last_state_play_frame = last_state.get_cur_play_frame(cur_frame)
        # self.mix_motion = concatenate_two_motions(
        #     last_state.motion,
        #     self.motion,
        #     last_state_play_frame,
        #     mix_time,
        # )
        # self.mix_motion = self.mix_motion.sub_sequence(
        #     last_state_play_frame, last_state_play_frame + self.motion.motion_length
        # )
        # print(self.mix_motion.motion_length)
        # self.is_mixing = self.mix_motion != None

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        return None

    def update(self, cur_frame):
        play_frame = cur_frame - self.start_frame
        if play_frame > self.motion.motion_length - 1:
            self.is_mixing = False
            if self.is_loop:
                self.start_frame = cur_frame
                play_frame = 0
                pos = self.motion.joint_position[-1, 0, [0, 2]]
                rot = self.motion.joint_rotation[-1, 0]
                facing_axis = (
                    R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                )
                self.motion = self.motion.translation_and_rotation(0, pos, facing_axis)
            else:
                play_frame = self.motion.motion_length - 1

        play_motion = (
            self.mix_motion
            if self.is_mixing and self.mix_motion != None
            else self.motion
        )
        joint_translation, joint_orientation = play_motion.batch_forward_kinematics()

        joint_translation = joint_translation[play_frame]
        joint_orientation = joint_orientation[play_frame]
        return self.motion.joint_name, joint_translation, joint_orientation

    def end(self, cur_frame):
        _, joint_translation, joint_orientation = self.update(cur_frame)
        pos = joint_translation[0, [0, 2]]
        rot = joint_orientation[0]
        facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
        return pos, facing_axis

    def frade_out_motion(self, cur_frame):
        return self.motion.sub_sequence(cur_frame, cur_frame + 10)

    def get_cur_play_frame(self, cur_frame):
        play_frame = cur_frame - self.start_frame
        return (
            play_frame
            if play_frame < self.motion.motion_length - 1
            else self.motion.motion_length - 1
        )


class IdleState(MotionState):
    def __init__(self) -> None:
        MotionState.__init__(self, "lab2/motion_material/idle.bvh", True)

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        angle0 = np.rad2deg(np.arccos(desired_rot_list[0][-1]) * 2)
        angle1 = np.rad2deg(np.arccos(desired_rot_list[-1][-1]) * 2)
        if angle1 - angle0 > 20:
            # turn right
            print(angle1, angle0)
            return WalkTurnRightState()
        elif angle1 - angle0 < -20:
            print(angle1, angle0)
            # turn left
            return WalkTurnLeftState()
        elif 0 < angle1 - angle0 < 20:
            return WalkForwardState()
        # print(desired_vel_list[1])
        # if np.linalg.norm(desired_vel_list[-1]) > 0.2:
        #     cosangle = np.dot(desired_vel_list[0], desired_vel_list[-1]) / (
        #         np.linalg.norm(desired_vel_list[0])
        #         * np.linalg.norm(desired_vel_list[-1])
        #     )
        #     angle = np.arccos(cosangle)
        #     print(cosangle, angle)
        #     if angle > 0.5:
        #         return WalkTurnRightState()
        #     elif angle < -0.5:
        #         return WalkTurnLeftState()
        #     return WalkForwardState()
        # else:
        #     return None
        return None


class WalkTurnRightState(MotionState):
    def __init__(self) -> None:
        MotionState.__init__(
            self, "lab2/motion_material/walk_and_ture_right.bvh", False
        )
        self.motion = self.motion.sub_sequence(80, 110)

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        play_frame = cur_frame - self.start_frame
        if play_frame > self.motion.motion_length - 1:
            return IdleState()
        else:
            return None


class WalkTurnLeftState(MotionState):
    def __init__(self) -> None:
        MotionState.__init__(self, "lab2/motion_material/walk_and_turn_left.bvh", False)
        self.motion = self.motion.sub_sequence(0, 50)

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        play_frame = cur_frame - self.start_frame
        if play_frame > self.motion.motion_length - 1:
            return IdleState()
        else:
            return None


class WalkForwardState(MotionState):
    def __init__(self) -> None:
        MotionState.__init__(self, "lab2/motion_material/walk_forward.bvh", True)

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        if np.linalg.norm(desired_vel_list[1]) < 0.2:
            return IdleState()
        else:
            return None


class BlendState(MotionState):
    def __init__(self, last_state, next_state) -> None:
        self.is_loop = False
        self.last_state = last_state
        self.next_state = next_state
        self.is_blend = False

    def start(self, cur_frame):
        super().start(cur_frame)
        last_state_play_frame = self.last_state.get_cur_play_frame(cur_frame)
        self.motion = concatenate_two_motions(
            self.last_state.motion,
            self.next_state.motion,
            last_state_play_frame,
            mix_time,
        )
        self.motion = self.motion.sub_sequence(
            last_state_play_frame, self.motion.motion_length
        )

    def check_trans(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):
        frame = cur_frame - self.start_frame
        if frame > self.motion.motion_length - 1:
            return self.next_state


test_text = OnscreenText(text="desired_vel_list", pos=(0, -0.9))


class MotionStatemachine:
    def __init__(self) -> None:
        self.cur_state = IdleState()

    def update_state(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
        current_gait,
        cur_frame,
    ):

        rot = desired_rot_list[0] - desired_rot_list[6]
        a = R.from_quat(desired_rot_list[0])
        b = R.from_quat(desired_rot_list[-1])

        # print(angle0,angle1)
        # cosangle = np.dot(desired_pos_list[0], desired_pos_list[-1]) / (
        #     np.linalg.norm(desired_pos_list[0]) * np.linalg.norm(desired_pos_list[-1])
        # )
        angle0 = np.rad2deg(np.arccos(desired_rot_list[0][-1]) * 2)
        angle1 = np.rad2deg(np.arccos(desired_rot_list[-1][-1]) * 2)
        # angle = np.rad2deg(np.arccos(cosangle))
        test_text.text = "angle0:{0:.3f} angle1:{1:.3f}".format(angle0, angle1)
        new_state = self.cur_state.check_trans(
            desired_pos_list,
            desired_rot_list,
            desired_vel_list,
            desired_avel_list,
            current_gait,
            cur_frame,
        )
        if new_state != None:
            pos, facing_axis = self.cur_state.end(cur_frame)
            # if self.cur_state.is_blend :
            #     self.cur_state = BlendState(self.cur_state, new_state)
            # else:
            new_state.start(cur_frame, pos, facing_axis)
            print(type(self.cur_state).__name__ + "->" + type(new_state).__name__)
            self.cur_state = new_state
        return self.cur_state.update(cur_frame)
