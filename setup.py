from setuptools import setup, find_packages

setup(name='abdg',
      version='0.1',
      description='ABDG',
      long_description='Attribute-based Decision Graph',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
      ],
      url='https://github.com/airysen/abdg',
      author='Arseniy Kustov',
      author_email='me@airysen.co',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'tqdm', 'pandas', 'networkx', 'imblearn', 'caimcaim', 'sklearn'],
      include_package_data=True,
      zip_safe=False)
