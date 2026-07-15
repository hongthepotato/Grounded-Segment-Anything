from setuptools import setup

package_name = "yolo_inference"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gstack",
    maintainer_email="noreply@example.com",
    description="YOLO inference ROS2 node for edge deployment",
    license="MIT",
    entry_points={
        "console_scripts": [
            "node = yolo_inference.node:main",
        ],
    },
)
