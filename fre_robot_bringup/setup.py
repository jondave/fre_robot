from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fre_robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Recursively include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.launch.py', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mark1',
    maintainer_email='mark1@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)