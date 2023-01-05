from setuptools import setup  # type: ignore

setup(
    name="wordle_solve",
    version="1.0",
    description="Wordle solver and play assist tool.",
    author="Joseh Palombo",
    packages=["wordle_solve"],
    setup_requires=["numpy", "pandas", "pandas-stubs", "scipy"],
)
