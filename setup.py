from setuptools import setup, find_packages

setup(
    name='tclm',
    version='0.0.0',
    author='Hangyeol Yu',
    author_email="hangyeol.yu@riiid.co",
    packages=find_packages(),
    include_package_data=True,
    description="A codebase for experiments on task compositional language model", 
    package_data={},
    zip_safe=False,
)
