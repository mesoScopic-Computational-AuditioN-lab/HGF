from setuptools import setup

VERSION = '0.1.0'

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='HGF',
    url='https://github.com/mesoScopic-Computational-AuditioN-lab/HGF',
    author='Jorie van Haren',
    author_email='jjg.vanharen@maastrichtuniversity.nl',
    # Needed to actually package something
    packages=['HGF'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version=VERSION,
    # The license can be anything you like
    license='MIT',
    description='A Hierarchical Gaussian Filter Toolbox for Python',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
