
from distutils.core import setup
import os
import sys

packages = ['beak', 'beak.render', 'beak.analyze', 'beak.msm', 'beak.visualize']
scripts = ['beak/reimage']
package_data = {
        }

setup(name='beak',
      version='1.0.0a1',
      description='Robins scripts',
      author='Robin Betz',
      author_email='robin@robinbetz.com',
      url='http://robinbetz.com',
      license='GPLv2 or later',
      packages=packages,
      scripts=scripts,
     )

