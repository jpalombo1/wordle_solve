from setuptools import setup  # type: ignore

setup(
    name="wordle_solve",
    version="1.0",
    description="Mortgage Calculation and Analysis Package",
    author="Joseh Palombo",
    packages=["wordle_solve"],
    setup_requires=["numpy", "pandas", "pandas-stubs", "scipy"],
)
