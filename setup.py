from setuptools import setup

setup(
    name='JarbasModelZoo',
    version='0.2.0a2post1',
    packages=['JarbasModelZoo',
              'JarbasModelZoo.features',
              "JarbasModelZoo.nltk_taggers",
              "JarbasModelZoo.nltk_chunkers"],
    url='https://github.com/OpenJarbas/ModelZoo',
    license='apache-2.0',
    author='jarbasai',
    include_package_data=True,
    author_email='jarbasai@mailfence.com',
    description='pretrained models (down)loader'
)
