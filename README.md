# ComputerVisionForRobotics-2024-2
Software para el curso "Visión computacional aplicada a la robótica" semestre 2024-2, Maestría en Ingeniería Eléctrica, UNAM

## Requerimientos

* Ubuntu 20.04 https://releases.ubuntu.com/focal/ubuntu-20.04.6-desktop-amd64.iso
* ROS Noetic http://wiki.ros.org/noetic/Installation/Ubuntu

## Instalación

Nota: se asume que ya se tiene instalado Ubuntu y ROS.

* $ cd
* $ git clone https://github.com/mnegretev/ComputerVisionForRobotics-2024-2
* $ cd ComputerVisionForRobotics-2024-2
* $ ./Setup.sh
* $ cd catkin_ws
* $ catkin_make -j2 -l2
* $ echo "source ~/ComputerVisionForRobotics-2024-2/catkin_ws/devel/setup.bash" >> ~/.bashrc
* $ source ~/.bashrc

## Pruebas

Para probar que todo se instaló y compiló correctamente:

* $ cd 
* $ source ComputerVisionForRobotics-2024-2/catkin_ws/devel/setup.bash
* $ roslaunch surge_et_ambula vision_course.launch

Si todo se instaló y compiló correctamente, se debería ver un visualizador como el siguiente:
![rviz](https://github.com/mnegretev/ComputerVisionForRobotics-2024-2/blob/master/Media/rviz.png)

Un ambiente simulado como el siguiente:
![gazebo](https://github.com/mnegretev/ComputerVisionForRobotics-2024-2/blob/master/Media/gazebo.png)

Y una GUI como la siguiente:
![GUIExample](https://github.com/mnegretev/ComputerVisionForRobotics-2024-2/blob/master/Media/gui.png)


## Contacto
Dr. Marco Negrete<br>
Profesor Asociado C<br>
Departamento de Procesamiento de Señales<br>
Facultad de Ingeniería, UNAM <br>
marco.negrete@ingenieria.unam.edu<br>
