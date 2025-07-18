# Generate urdf for ur3e

### Download the description
from 
https://github.com/UniversalRobots/Universal_Robots_ROS2_Description

After that cd to urdf, and then
```
xacro ur.urdf.xacro ur_type:=ur3e name:=ur3e > ur3e.urdf
```

One problem occur is that, 

```
(base) kklab-ur-robot@kklab-ur-robot-MS-7D77:~/Universal_Robots_ROS2_Description/urdf$ xacro ur.urdf.xacro ur_type:=ur3e name:=ur3e > ur3e.urdf
Invalid parameter "force_abs_paths"
when instantiating macro: ur_robot (/opt/ros/humble/share/ur_description/urdf/ur_macro.xacro)
in file: ur.urdf.xacro
```


this problem is due to we may use the gazebo or other things. When using this we need this parameter, we can go the urdf and remove it.

```
force_abs_paths="$(arg force_abs_paths)"
```

And the just generate again from 

```
xacro ur.urdf.xacro ur_type:=ur3e name:=ur3e > ur3e.urdf
```

### Do cp to copy the robot urdf and meshes to simulation path
We can find the meshes in the original ur_ros2 path, and cp it also cp the urdf or ur3e to the simulation path. Also do the replace of urdf path

```
cd ~/Desktop/pybullet_test/ur5-bullet/UR5/ur_e_description/urdf
sed -i 's@package://ur_description/meshes@../meshes@g' ur3e.urdf
```

Change the package to the real path of your mesh in the simulation space.
