### PyPi

1. Make all the code changes, including updating version numbers in `pyproject.toml` and `README.md`

2. Issue the release on github, creating a new tag

3. Fetch all the tags with `git pull`

4. Delete any builds for old versions in `/dist`

5. Build the new distribution using `python setup.py sdist bdist_wheel`

6. Upload to pypi with `twine upload dist/*`

7. Check that the [pypi package page](https://pypi.org/project/cubed-xarray/) shows the expected version.

### Conda-forge

1. Fork the [cubed-xarray feedstock](https://github.com/conda-forge/cubed-xarray-feedstock), creating a branch for the new version

2. The branch should change the `recipe/meta.yaml`, specifically changing the package version number, resetting the build number to 0, and updating the hash

3. The new hash is found using e.g. `openssl sha256 dist/cubed-xarray-0.0.5.tar.gz`

4. Open a PR on the conda-forge feedstock for the changes, and merge it once the bots approve.
