
from distutils.core import setup
import os
import sys

packages = ['beak', 'beak.render', 'beak.analyze', 'beak.msm']
scripts = ['beak/reimage.py']
package_data = {
        }

setup(name='dabble',
      version='1.0.0a1',
      description='Membrane protein system builder',
      author='Robin Betz',
      author_email='robin@robinbetz.com',
      url='http://robinbetz.com',
      license='GPLv2 or later',
      packages=packages,
      scripts=scripts,
     )

