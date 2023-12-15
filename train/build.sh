rm -rf ./dist
python setup.py bdist_wheel
rm -rf ./build ./flash_attention.egg-info