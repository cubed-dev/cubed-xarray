1. Make all the code changes, including updating version numbers in `pyproject.toml` and `README.md`

2. Issue the release on github, creating a new tag

3. Fetch all the tags with `git pull`

4. Delete any builds for old versions in `/dist`

5. Build the new distribution using `python setup.py sdist bdist_wheel`

6. Upload to pypi with `twine upload dist/*`

7. Check that the [pypi package page](https://pypi.org/project/cubed-xarray/) shows the expected version.
