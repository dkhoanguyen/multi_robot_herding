import setuptools

packages = setuptools.find_packages(".")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_robot_herding",
    version="1.0.0",
    author="Khoa Nguyen",
    author_email="khoanguyendacdang2198@gmail.com",
    packages=packages,
    python_requires='>=3'
)