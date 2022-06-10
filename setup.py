from setuptools import setup

VERSION = '0.1.0'

setup(
    name='HGF',
    url='https://github.com/mesoScopic-Computational-AuditioN-lab/HGF',
    author='Jorie van Haren',
    author_email='jjg.vanharen@maastrichtuniversity.nl',
    packages=['HGF'],
    install_requires=['numpy'],
    version=VERSION,
    license='MIT',
    description='A Hierarchical Gaussian Filter Toolbox for Python',
)
