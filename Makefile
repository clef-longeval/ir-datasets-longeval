build-pypi-package:
	rm -Rf dist
	python3 -m build --sdist .
	python3 -m build --wheel .
	twine upload dist/ir_datasets_longeval-*-py3-none-any.whl dist/ir_datasets_longeval-*.tar.gz
