#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"
#include <ros/ros.h>
#include <whisker_customed_msg/FrankaCtrl.h>
#include <geometry_msgs/TwistStamped.h>
#include <Eigen/Dense>

#define TOTAL_VEL 0.008 // Ensure consistency

#define TRANSLATION_VEL_LIMIT 1.7           // dp.Translation - (m/s)
#define TRANSLATION_ACC_LIMIT 13.0 * 0.1    // ddp.Translation - (m/s^2) : factor 10
#define TRANSLATION_JERK_LIMIT 6500.0 * 0.1 // dddp.Translation - (m/s^3) : factor 10
#define ROTATION_VEL_LIMIT 2.5              // dp.Rotation - (rad/s)
#define ROTATION_ACC_LIMIT 25.0 * 0.1       // ddp.Rotation - (rad/s^2) : factor 10
#define ROTATION_JERK_LIMIT 12500.0 * 0.1   // dddp.Rotation - (rad/s^3) : factor 10

struct ControlCommands
{
    double x_velocity = 0.0;
    double y_velocity = -TOTAL_VEL;
    double rotation = M_PI; // Target rotation position on z-axis
    std::mutex mtx;
};

ControlCommands control_commands;
std::array<double, 16> initial_pose;
double accumulated_val = control_commands.rotation;

void controlCommandCallback(const whisker_customed_msg::FrankaCtrl::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(control_commands.mtx);
    control_commands.x_velocity = msg->xvel;
    control_commands.y_velocity = msg->yvel;
    control_commands.rotation = static_cast<double>(msg->orientation); // DYX : float32 -> double
    ROS_INFO("!!! Received control command: x_velocity=%f, y_velocity=%f, rotation=%f", msg->xvel, msg->yvel, msg->orientation);
}

double ramp(double target, double previous, double max_rate, double dt)
{
    double diff = target - previous;
    double max_change = max_rate * dt;
    if (fabs(diff) > max_change)
    {
        target = previous + max_change * (diff > 0 ? 1 : -1);
    }
    return target;
}

Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d euler;
    euler[0] = atan2(R(2, 1), R(2, 2));                                  // Roll
    euler[1] = atan2(-R(2, 0), sqrt(pow(R(2, 1), 2) + pow(R(2, 2), 2))); // Pitch
    euler[2] = atan2(R(1, 0), R(0, 0));                                  // Yaw
    return euler;
}

geometry_msgs::TwistStamped matrixToTwistStamped(const std::array<double, 16> &O_T_EE)
{
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix << O_T_EE[0], O_T_EE[1], O_T_EE[2],
        O_T_EE[4], O_T_EE[5], O_T_EE[6],
        O_T_EE[8], O_T_EE[9], O_T_EE[10];

    Eigen::Vector3d euler_angles = rotationMatrixToEulerAngles(rotation_matrix);

    geometry_msgs::TwistStamped twist;
    twist.twist.linear.x = O_T_EE[12];
    twist.twist.linear.y = O_T_EE[13];
    twist.twist.linear.z = O_T_EE[14];

    twist.twist.angular.x = euler_angles[0];
    twist.twist.angular.y = euler_angles[1];
    twist.twist.angular.z = euler_angles[2];

    return twist;
}

void wrapDetection(double cur, double pre)
{
    if (!std::isnan(pre))
    {
        if (pre > 0 && cur < 0 && std::abs(pre - cur) > 3) // pi -> -pi
        {
            accumulated_val += (cur + 2 * M_PI) - pre;
        }
        else if (pre < 0 && cur > 0 && std::abs(pre - cur) > 3) // -pi -> pi
        {
            accumulated_val += (cur - 2 * M_PI) - pre;
        }
        else // 0 -> -0 and -0 -> 0 and all the other situations
        {
            accumulated_val += cur - pre;
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "robot_ctrl");
    ros::NodeHandle nh;

    if (argc != 2)
    {
        ROS_ERROR("Usage: %s <robot-hostname>", argv[0]);
        return -1;
    }

    ros::Subscriber control_command_sub = nh.subscribe("/Franka_Ctrl", 10, controlCommandCallback);
    ros::Publisher robot_state_pub = nh.advertise<geometry_msgs::TwistStamped>("/FrankaEE_State", 10);

    // Use AsyncSpinner for non-blocking callback handling, the spin loop and control loop were separate
    ros::AsyncSpinner spinner(1);
    spinner.start();

    try
    {
        franka::Robot robot(argv[1]);
        setDefaultBehavior(robot);
        setDefaultBehavior(robot);
        std::array<double, 7> q_default = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}}; // Franka Robot Default Pose
        // std::array<double, 7> q_initPosi = {{-0.7643998571446068, -0.22742027574051701, -7.572768736026133e-05, -2.4360524018554077, -0.000192071483101851, 2.2076728652802347, 0.021008460091220007}}; // Bottle
        std::array<double, 7> q_initPosi_0 = {{-0.5099099633502283, -0.12138736607230087, -0.0001052034688498743, -2.631685962597177, -0.00045342696415238124, 2.5083344097137448, 0.27578429961552614}}; // 160 circle
        std::array<double, 7> q_initPosi_MPI = {{0.24427948517757547, 0.4584773261202814, 2.518500587821277e-05, -1.8082812422869678, -0.0005250496148132926, 2.2667111081644187, -2.1107083238495714}};
        ROS_WARN("This example will move the robot! Please make sure to have the user stop button at hand!");
        ROS_INFO("Press Enter to continue...");
        std::cin.ignore();
        MotionGenerator motion_generator_default(0.5, q_default);
        robot.control(motion_generator_default);
        MotionGenerator motion_generator_initPosi(0.5, q_initPosi_MPI);
        robot.control(motion_generator_initPosi);
        ROS_INFO("Finished moving to initial joint configuration.");

        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

        std::array<double, 7> lower_torque_thresholds_nominal{{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0}};
        std::array<double, 7> upper_torque_thresholds_nominal{{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
        std::array<double, 7> lower_torque_thresholds_acceleration{{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0}};
        std::array<double, 7> upper_torque_thresholds_acceleration{{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
        std::array<double, 6> lower_force_thresholds_nominal{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
        std::array<double, 6> upper_force_thresholds_nominal{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
        std::array<double, 6> lower_force_thresholds_acceleration{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
        std::array<double, 6> upper_force_thresholds_acceleration{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
        robot.setCollisionBehavior(
            lower_torque_thresholds_acceleration, upper_torque_thresholds_acceleration,
            lower_torque_thresholds_nominal, upper_torque_thresholds_nominal,
            lower_force_thresholds_acceleration, upper_force_thresholds_acceleration,
            lower_force_thresholds_nominal, upper_force_thresholds_nominal);

        double time_max = 150.0;
        double time = 0.0;
        double previous_yaw = control_commands.rotation;
        robot.control([=, &time, &robot_state_pub, &previous_yaw](const franka::RobotState &robot_state, franka::Duration period) -> franka::CartesianVelocities
                      {
    time += period.toSec();
    double dt = period.toSec();
    if (time == 0.0) {
        initial_pose = robot_state.O_T_EE_c;
    }
    ROS_INFO("-----");
    ROS_INFO("At time : %f; Perception control commands : %f, %f, %f.", time, control_commands.x_velocity, control_commands.y_velocity, control_commands.rotation);

    geometry_msgs::TwistStamped ee_state = matrixToTwistStamped(robot_state.O_T_EE);
    ee_state.header.stamp = ros::Time::now();
    robot_state_pub.publish(ee_state);
    ROS_INFO("At time : %f; Robot cartesian pose : %f, %f, %f.", time, ee_state.twist.linear.x, ee_state.twist.linear.y, ee_state.twist.angular.z);

    // Retrieve linear velocity
    double target_v_x;
    double target_v_y;
    {
        std::lock_guard<std::mutex> lock(control_commands.mtx);
        target_v_x = control_commands.x_velocity;
        target_v_y = control_commands.y_velocity;
    }

    // Retrieve target rotation and transform into angular velocity 
    double current_yaw = atan2(robot_state.O_T_EE_c[1], robot_state.O_T_EE_c[0]);
    wrapDetection(current_yaw, previous_yaw);
    previous_yaw = current_yaw;
    double target_yaw;
    {
        std::lock_guard<std::mutex> lock(control_commands.mtx);
        target_yaw = control_commands.rotation;
    } 
    ROS_INFO("Current yaw: %f, accumulated yaw: %f", current_yaw, accumulated_val);
    double angular_error = target_yaw - accumulated_val;
    double target_w_z = angular_error * 1.0;

    // dP limit
    target_v_x = std::max(-TRANSLATION_VEL_LIMIT, std::min(target_v_x, TRANSLATION_VEL_LIMIT));
    target_v_y = std::max(-TRANSLATION_VEL_LIMIT, std::min(target_v_y, TRANSLATION_VEL_LIMIT));
    target_w_z = std::max(-ROTATION_VEL_LIMIT, std::min(target_w_z, ROTATION_VEL_LIMIT));

    // Compute target ddP
    double target_a_x = (target_v_x - robot_state.O_dP_EE_c[0]) / dt;
    double target_a_y = (target_v_y - robot_state.O_dP_EE_c[1]) / dt;
    double target_a_wz = (target_w_z - robot_state.O_dP_EE_c[5]) / dt;

    // ddP limit
    target_a_x = std::max(-TRANSLATION_ACC_LIMIT, std::min(target_a_x, TRANSLATION_ACC_LIMIT));
    target_a_y = std::max(-TRANSLATION_ACC_LIMIT, std::min(target_a_y, TRANSLATION_ACC_LIMIT));
    target_a_wz = std::max(-ROTATION_ACC_LIMIT, std::min(target_a_wz, ROTATION_ACC_LIMIT));

    // Ramp ddP based on dddP limits
    double a_x = ramp(target_a_x, robot_state.O_ddP_EE_c[0], TRANSLATION_JERK_LIMIT, dt);
    double a_y = ramp(target_a_y, robot_state.O_ddP_EE_c[1], TRANSLATION_JERK_LIMIT, dt);
    double a_wz = ramp(target_a_wz, robot_state.O_ddP_EE_c[5], ROTATION_JERK_LIMIT, dt);

    // Update dP based on ramped ddP
    double v_x = robot_state.O_dP_EE_c[0] + a_x * dt;
    double v_y = robot_state.O_dP_EE_c[1] + a_y * dt;
    double w_z = robot_state.O_dP_EE_c[5] + a_wz * dt;
    ROS_INFO("At time : %f; Final control commands input : %f, %f, %f.", time, v_x, v_y, w_z);

    franka::CartesianVelocities output = {{v_x, v_y, 0.0, 0.0, 0.0, w_z}};
    if (time >= time_max) {
        ROS_INFO("Finished motion, shutting down example");
        return franka::MotionFinished(output);
    }
    return output; });
    }
    catch (const franka::Exception &e)
    {
        ROS_ERROR("Franka exception: %s", e.what());
        try
        {
            franka::Robot robot(argv[1]);
            ROS_WARN("Attempting automatic error recovery...");
            robot.automaticErrorRecovery();
            ROS_INFO("Automatic error recovery successful.");
        }
        catch (const franka::Exception &recovery_exception)
        {
            ROS_ERROR("Automatic error recovery failed: %s", recovery_exception.what());
            return -1;
        }
    }

    spinner.stop();
    return 0;
}
