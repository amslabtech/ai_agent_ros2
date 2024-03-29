from setuptools import setup
 
package_name = 'travel'
 
setup(
    name=package_name,
    version='0.0.0',
    packages=['object_detection','yolo3', 'mrcnn'],
    package_dir={'yolo3': 'object_detection/yolo3', 'mrcnn': 'object_detection/mrcnn'},
    py_modules=[
        'demo',
        'demo_yolo',
        'text_publisher',
        'keyboard_publisher',
        'image_publisher',
        'object_detection_publisher',
        'action',
        'policy',
        'state',
        'agent',
        'rlagent',
        'spawn_agent'
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='user',
    author_email="user@todo.todo",
    maintainer='user',
    maintainer_email="user@todo.todo",
    keywords=['ROS', 'ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='TODO: Package description.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo = demo:main',
            'demo_yolo = demo_yolo:main',
            'text_publisher = text_publisher:main',
            'keyboard_publisher = keyboard_publisher:main',
            'image_publisher = image_publisher:main',
            'object_detection_publisher = object_detection_publisher:main',
            'agent = agent:main',
            'rlagent = rlagent:main',
            'spawn_agent = spawn_agent:main',
        ],
    },
)
