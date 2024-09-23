#!/usr/bin/env python3

import rospy
import numpy as np
import scipy.interpolate as interpolate

from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped, TwistStamped
from whisker_customed_msg.msg import MagneticFieldVector, EESensorState, FrankaCtrl
from collections import deque
from filterpy.kalman import KalmanFilter


# Filtering parameters
initial_state = [-0.058, 0.087] # !!! DYX !!! : To be defined, the fixed coordinate under our optimal deflection state
initial_variance = 10.0  
filter_window = 10 
scaling_factor_Q = 0.01 
tip_pos_s_deq = deque(maxlen=filter_window) 
tip_pos_w_filtered_que = [] 

# Excuting parameters
collision_threshold = 5500 # !!! DYX !!! : To be defined, detect the collision when pass this value
contacted = 0
touch_index = 0
stable_distance = 400

# Rotation decision parameters
keypoint_interval = 10
keypoint_length = 20
keypoints_deq = deque(maxlen=keypoint_length)
n_interior_knots = 5
spline_degree = 3
u_next = 1 + 1 / (keypoint_length - 1)
theta_last_measured = - 0.5 * np.pi
theta_last_desired = - 0.5 * np.pi
theta_next_desired = - 0.5 * np.pi
wrap_count = 0
rot_limit_scale = 0.002
initial_slope = 0.5 * np.pi # !!! DYX !!! : To be defiend

# Translation decision parameters
DEF_TARGET = 7400 # !!! DYX !!! : To be defined, correspond to the optimal cotact distance XX mm
KP = 0.0002
KI = 0.00003
KD = 0.000003
X_VEL = 0.002
Y_VEL = 0.008
TOTAL_VEL = 0.008
PID_scale_bound = TOTAL_VEL / X_VEL  


class EESensorStateData:
    def __init__(self) -> None:
        self.current_time = rospy.Time()
        self.deflection_moment = 0.0
        self.xpos_ee = 0.0
        self.ypos_ee = 0.0
        self.zrot_ee = 0.0


class FrankaRobotCtrl:
    def __init__(self) -> None:
        self.xvel = 0.0
        self.yvel = 0.0
        self.orientation = 0.0


class BayesianFilter:
    def __init__(self) -> None:
        self.f = KalmanFilter(dim_x=2, dim_z=2)
        self.f.x = np.array([initial_state[0], initial_state[1]])
        self.f.F = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.f.H = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.f.P *= initial_variance
        self.f.R = np.eye(self.f.dim_z)
        self.f.Q = np.eye(self.f.dim_x) * scaling_factor_Q


class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point=0) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.previous_error = 0
        self.integral = 0
        self.previous_time = rospy.get_time()

    def update(self, measured_value):
        current_time = rospy.get_time()
        dt = current_time - self.previous_time
        if dt <= 0.0:
            dt = 1e-6  # Avoid division by zero or negative time intervals

        error = self.set_point - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.previous_time = current_time

        return output


# Synchronized state information on end-effector and sensor
state = EESensorStateData()
# Calculated next ctrl command for franka robot
ctrl = FrankaRobotCtrl()
# Bayesian filter initialization
filter = BayesianFilter()
# PIDControll initialization
# controller = PIDController(KP, KI, KD, DEF_TARGET)


# Calculate tip position via acquired measuremnt based on a characterized model : Y
def deflection2fy(an):
    fY = (-1.637e-17 * an**5 + 3.271e-13 * an**4 - 1.127e-09 * an**3 -
         1.587e-05 * an**2 + 0.1498 * an - 366.6)
    return -fY / 1000


# Calculate tip position via acquired measuremnt based on a characterized model : X
def deflection2fx(an):
    fX = (-6.332e-17 * an**5 + 1.362e-12 * an**4 - 5.152e-09 * an**3 - 
          6.971e-05 * an**2 + 0.6287 * an - 1419)
    return (75 - fX) / 1000


# Iteratively update the measurement noise
def update_noise_matrices(measurements):
    # The measurement noise R - covariance matrix was determined and updated empicically by the most recently collected data points
    # sampled length : filter_window = N, including the value on current index
    zset = np.array(measurements)
    filter.f.R = np.cov(zset[:, 0], zset[:, 1])


# Prevent the rotaion from exterme disturbance and refine the orientation
def refineOrientation(r):
    global wrap_count
    global theta_last_desired
    global theta_last_measured
    #  Use last original measurement (-pi, pi) to detect a full turn
    # if r - theta_last_measured > 1.888 * np.pi:
    #     wrap_count += 1
    # elif r - theta_last_measured < -1.888 * np.pi:
    #     wrap_count -= 1
        
    # detect wrap 
    if r > 0 and theta_last_measured < 0 : # -pi -> pi & +0 -> -0 : clockwise : we were forwarding along this direction, so increase wrap in this block
        wrap_count += 1
    elif r < 0 and theta_last_measured > 0 : # pi -> -pi & -0 -> +0 : counter clockwise
        wrap_count -= 1
    
    r_refined = r - wrap_count * 2 * np.pi
    rospy.loginfo("%f. %f, %f", r, theta_last_measured, wrap_count)

    # rotary speed limitation
    global touch_index

    if touch_index >= stable_distance:
        # Multiply a servoing ratio to regulat our limit, lower ratio means larger direction change between every decision
        rot_speed_limit = rot_limit_scale * keypoint_interval # 0.00125
        # Use last desired of accumulated rotation to limit speed
        theta_err = r_refined - theta_last_desired
        if (theta_err > rot_speed_limit and theta_err < 0.5 * np.pi) or \
            (theta_err < - 1.5*np.pi and theta_err > - 2*np.pi + rot_speed_limit) or \
            (theta_err <= -0.5*np.pi and theta_err >= -np.pi) or \
            (theta_err >= np.pi and theta_err <= 1.5*np.pi):
            r_refined = theta_last_desired + rot_speed_limit * 0.1  
            # r_refined = theta_last_desired    
        elif (theta_err < -rot_speed_limit and theta_err > -0.5 * np.pi) or \
            (theta_err > 1.5*np.pi and theta_err < 2*np.pi - rot_speed_limit) or \
            (theta_err >= 0.5*np.pi and theta_err <= np.pi) or \
            (theta_err <= - np.pi and theta_err >= - 1.5*np.pi):
            r_refined = theta_last_desired - rot_speed_limit

    # r : (-pi, pi); r_refined : no upper bound accumlated rotation
    return r_refined


def callback(sensor_msg, frankaEE_msg):
    # !!! DYX !!! : transform the quanterian into radian
    
    # Give access of the synchronized states (Serial Node & Franka Robot Node) to local object
    state.current_time = rospy.Time.now()
    state.deflection_moment = sensor_msg.magnetic_y
    state.xpos_ee = frankaEE_msg.twist.linear.x
    state.ypos_ee = frankaEE_msg.twist.linear.y
    state.zrot_ee = frankaEE_msg.twist.angular.z
    
    rospy.loginfo("------")
    rospy.loginfo("Time: %f, Deflection: %f, Position: (%f, %f), Orientation: %f",
                  state.current_time.to_sec(), state.deflection_moment,
                  state.xpos_ee, state.ypos_ee, state.zrot_ee)

    # Publish the current robot & sensor state
    msg = EESensorState()
    msg.header.stamp = rospy.Time.now()
    msg.magnetic_y = state.deflection_moment
    msg.pose_ee.position.x = state.xpos_ee
    msg.pose_ee.position.y = state.ypos_ee
    msg.pose_ee.orientation.z = state.zrot_ee
    state_pub.publish(msg)

    # Collision detection
    global contacted
    if np.abs(state.deflection_moment) > collision_threshold:
        contacted = 1
        
    # Calculate the contact and next ctrl here
    global touch_index
    if contacted:  
        touch_index += 1
        # Produce direct estimate of tip position
        tip_X_s = deflection2fx(state.deflection_moment)
        tip_Y_s = deflection2fy(state.deflection_moment)
        tip_pos_s_deq.append([tip_X_s, tip_Y_s])
        if len(tip_pos_s_deq) == filter_window:
            # Update measurement noise every iteration and filter the result
            update_noise_matrices(tip_pos_s_deq)
            filter.f.predict() # !!! DYX !!! : better to transform state transition (static assumption) into world frame
            filter.f.update(tip_pos_s_deq[-1])
            tip_pos_s_filtered = filter.f.x.copy()
            # Transform the filtered tip point into world-fixed frame
            tip_pos_w_filtered = np.dot(
                np.array([
                    [
                        np.cos(state.zrot_ee), -np.sin(state.zrot_ee),
                        state.xpos_ee
                    ],
                    [
                        np.sin(state.zrot_ee),
                        np.cos(state.zrot_ee), state.ypos_ee
                    ],
                    [0, 0, 1],
                ]),
                np.array([[tip_pos_s_filtered[0]], [tip_pos_s_filtered[1]],
                          [1]]),
            )
            tip_pos_w_filtered_que.append(
                [tip_pos_w_filtered[0][0], tip_pos_w_filtered[1][0]])
        else:
            tip_pos_w_filtered = np.dot(
                np.array([
                    [
                        np.cos(state.zrot_ee), -np.sin(state.zrot_ee),
                        state.xpos_ee
                    ],
                    [
                        np.sin(state.zrot_ee),
                        np.cos(state.zrot_ee), state.ypos_ee
                    ],
                    [0, 0, 1],
                ]),
                np.array([[tip_X_s], [tip_Y_s], [1]]),
            )
            tip_pos_w_filtered_que.append(
                [tip_pos_w_filtered[0][0], tip_pos_w_filtered[1][0]])
            
        contact_msg = TwistStamped()
        contact_msg.header.stamp = rospy.Time.now()
        contact_msg.twist.linear.x = tip_pos_w_filtered_que[-1][0]
        contact_msg.twist.linear.y = tip_pos_w_filtered_que[-1][1]
        contact_pub.publish(contact_msg)

        # Rotary direction decision for next iteration
        global theta_next_desired
        global theta_last_desired
        global theta_last_measured
        if touch_index % keypoint_interval == 0:
            keypoints_deq.append(tip_pos_w_filtered_que[-1])

            # Only if we have collected enough keypoints and it is currently a keypoint, we make a decision
            if len(keypoints_deq) == keypoint_length:
                # Predict next contact position along BSpline curve
                qs = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
                knots = np.quantile(np.array(keypoints_deq)[:,1], qs)
                tck, u = interpolate.splprep(
                    [np.array(keypoints_deq)[:,0],
                    np.array(keypoints_deq)[:,1]],
                    t=knots,
                    k=spline_degree,
                )
                tip_pos_w_next = interpolate.splev(u_next, tck)
                # Transform a slope to next into angular measurement
                theta_next_measured = np.arctan2(
                    tip_pos_w_next[1] - np.array(keypoints_deq)[-1][1],
                    tip_pos_w_next[0] - np.array(keypoints_deq)[-1][0])
                # Refine the measurement as acceptable desired rotary ctrl
                theta_next_desired = refineOrientation(theta_next_measured)
                rospy.loginfo("%f, %f, %f, %f,",theta_next_desired, theta_next_measured, tip_pos_w_next[1] - np.array(keypoints_deq)[-1][1], tip_pos_w_next[0] - np.array(keypoints_deq)[-1][0])
                theta_last_desired = theta_next_desired
                theta_last_measured = theta_next_measured
            else:
                theta_next_measured = np.pi/2
                theta_next_desired = theta_next_measured

        # Linear translation decision for next iteration
        PID_scale = controller.update(state.deflection_moment)
        PID_scale = max(min(PID_scale, PID_scale_bound), -PID_scale_bound)
        xvel_s_next_desired = PID_scale * X_VEL
        # yvel_s_next_desired = np.sqrt(TOTAL_VEL**2 - xvel_s_next_desired**2)  
        yvel_s_next_desired = Y_VEL

        # Transform all the ctrls into world-fixed frame and usable format
        # theta_w_next_desired = theta_next_desired - initial_slope
        theta_w_next_desired = theta_next_desired + 1.5 * np.pi
        if theta_w_next_desired > np.pi:
            theta_w_next_desired = np.pi
        # !!!
        # When data.ctrl[2] = 0 in a start forward motion initial_slope = 0.5 * np.pi, the whisker base
        # frame has no rotation on the world-fixed frame. theta_next_desired is a value with no upper bound
        # accumulated via the theta_next_measured which produced from arctan2 increase periodically ranging
        # from (-pi, pi). Subtracted by a initial_slope will make the theta compatable with the actual rotary
        # change from sensor from to world frame, that is the rotation around the Z-axis of world coordinates.
        # So, all in all, it cames to a conclusion that theta_w_next_desired could be directly used as an
        # absolute target rotation radians on world-fixed frame ???
        # !!!
        xvel_w_next_desired = xvel_s_next_desired * \
            np.cos(theta_w_next_desired) - yvel_s_next_desired * \
            np.sin(theta_w_next_desired)
        yvel_w_next_desired = xvel_s_next_desired * \
            np.sin(theta_w_next_desired) + yvel_s_next_desired * \
            np.cos(theta_w_next_desired)

        # Publish and transport the ctrl commands
        if touch_index >= stable_distance:
            ctrl_msg = FrankaCtrl()
            ctrl_msg.xvel = xvel_w_next_desired
            ctrl_msg.yvel = yvel_w_next_desired
            ctrl_msg.orientation = theta_w_next_desired
            ctrl_pub.publish(ctrl_msg)
            
        rospy.loginfo("Contacted! Touch index: %f, xvel: %f, yvel: %f, new direction: %f",
                touch_index, xvel_w_next_desired,
                yvel_w_next_desired, theta_w_next_desired)


def main():
    rospy.init_node('Master_node', anonymous=True)

    global controller
    controller = PIDController(KP, KI, KD, DEF_TARGET)
    
    global contact_pub
    contact_pub = rospy.Publisher(
        '/TipPosotion', TwistStamped, queue_size=10
    )

    # Wait for the first message from both topics
    rospy.wait_for_message('/MagneticSensor', MagneticFieldVector)
    rospy.wait_for_message('/FrankaEE_State', TwistStamped)
    rospy.loginfo("Received Messages from  Sensor and Franka!")   

    global state_pub  # State publisher 
    state_pub = rospy.Publisher(
        '/EE_Sensor_state', EESensorState, queue_size=10)
    global ctrl_pub  # Ctrl publisher
    ctrl_pub = rospy.Publisher('/Franka_Ctrl', FrankaCtrl, queue_size=10)
    


    # Subscribe to the /Sensor_state and /FrankaEE_State topics
    sensor_sub = Subscriber('/MagneticSensor', MagneticFieldVector)
    frankaEE_sub = Subscriber('/FrankaEE_State', TwistStamped)

    # Define the synchronization policy and synchronizer
    ats = ApproximateTimeSynchronizer(
        [sensor_sub, frankaEE_sub], queue_size=10, slop=0.1)

    # Register the callback with the synchronizer
    ats.registerCallback(callback)

    # Spin to process incoming messages
    rospy.spin()


if __name__ == '__main__':
    main()
