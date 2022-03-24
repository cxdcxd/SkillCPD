from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['dong_pkg'],
    #scripts=['pbdlib/demos.py'],
	scripts=['note/online_record.py', 'note/test_pub.py'],
	#scripts=['note/online_record.py'],
	#scripts=['note/test_pub.py'],
    package_dir={'': 'src'}
)

setup(**d)
