from distutils.core import setup
setup(
  name = 'fst-pso',
  packages = ['fstpso'], # this must be the same as the name above
  version = '1.2.2',
  description = 'Fuzzy Self-Tuning PSO global optimization library',
  author = 'Marco S. Nobile',
  author_email = 'nobile@disco.unimib.it',
  url = 'https://github.com/aresio/fst-pso', # use the URL to the github repo
  #download_url = 'https://github.com/peterldowns/mypackage/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['fuzzy logic', 'particle swarm optimization', 'optimization', 'pso'], # arbitrary keywords
  license='LICENSE.txt',
  long_description=open('README.txt').read(),
  install_requires=[
        "pyfuzzy >= 0.1.0",        
        "numpy >= 1.12.0"
    ],
  include_package_data=True,
  classifiers = ['Programming Language :: Python :: 2.7'],
  package_data={ 
    'fstpso': ['pso_1st_half_2.fcl', 'pso_2nd_half_2.fcl'],
},
)