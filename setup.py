from distutils.core import setup
setup(
  name = 'fst-pso',
  packages = ['fstpso'], 
  version = '1.8.1',
  description = 'Fuzzy Self-Tuning PSO global optimization library',
  author = 'Marco S. Nobile',
  author_email = 'nobile@disco.unimib.it',
  url = 'https://github.com/aresio/fst-pso', # use the URL to the github repo
  keywords = ['fuzzy logic', 'optimization', 'discrete optimization', 'continuous optimization', 'particle swarm optimization',  'pso', 'fst-pso', 'fuzzy self-tuning pso'], # arbitrary keywords
  license='LICENSE.txt',
  long_description=open('README.txt', encoding='utf-8').read(),
  classifiers = ['Programming Language :: Python :: 3.7'],
  install_requires=[ 'numpy', 'miniful' ],
)