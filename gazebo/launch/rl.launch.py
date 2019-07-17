from launch import LaunchDescription
import launch.actions
import launch.substitutions
import launch_ros.actions

def generate_launch_description():
    gzserver_exe = launch.actions.ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so',
             launch.substitutions.LaunchConfiguration('world')],
        output='screen'
    )
    
    gzclient_exe = launch.actions.ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )
    
    # #実行プログラムのノード
    # node = launch_ros.actions.Node(
    #     package='PKG_program',
    #     node_executable='module',
    #     output='screen',
    #     #remappings=[
    #     #    ('cmd_vel', '/PROJECT/cmd_vel'), #トピックのパスはここで変える
    #     #   ('laser_scan', '/PROJECT/laser_scan')
    #     #]
    # )

    keyboard_pub = launch_ros.actions.Node(
        package='travel',
        node_executable='keyboard_publisher',
        output='screen',
    )

    obj_detect_pub = launch_ros.actions.Node(
        package='travel',
        node_executable='object_detection_publisher',
        output='screen',
    ) 

    rlagent = launch_ros.actions.Node(
        package='travel',
        node_executable='rlagent',
        output='screen',
    )

    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(
          'world',
          default_value=['empty.world', ''],
          description='Gazebo world file'),
        gzserver_exe,
        gzclient_exe,
        keyboard_pub,
        obj_detect_pub,
        rlagent
        #ローンチ時にプログラムを実行する
    ])