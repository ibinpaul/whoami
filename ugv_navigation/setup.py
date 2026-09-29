from glob import glob

from setuptools import setup

package_name = 'ugv_navigation'

setup(
    name=package_name,
    version='0.0.0',
    # costmap_core stays ROS-independent; costmap_ros holds the ROS layer.
    packages=['costmap_core', 'costmap_ros'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config/robots', glob('config/robots/*.yaml')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='TODO@todo.todo',
    description='Dev 3 costmap subsystem: ROS-independent core and ROS 2 layer.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'costmap_node = costmap_ros.costmap_node:main',
        ],
    },
)
