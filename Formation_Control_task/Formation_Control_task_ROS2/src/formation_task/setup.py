from setuptools import setup
from glob import glob

package_name = 'formation_task'
scripts = ['agent_i','agent_lett_i','visualizer']

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('resource/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessandro Cecconi, Marco Bugo, Roman Sudin',
    maintainer_email='alessandro.cecconi@studio.unibo.it, marco.bugo@studio.unibo.it, roman.sudin@studio.unibo.it',
    description='Package as delivery for DAS course at Unibo held by Professor Notarstefano',
    license='see github as a reference',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            '{1} = {0}.{1}:main'.format(package_name, script) for script in scripts
        ],
    },
)
